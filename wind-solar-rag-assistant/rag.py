"""Core RAG pipeline: document loading, chunking, indexing, hybrid retrieval and generation."""

import glob
import os
import pickle
import re

import faiss
import numpy as np
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

EMBED_MODEL = "BAAI/bge-small-en-v1.5"
HOSTED_LLMS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]
LOCAL_LLM = "Qwen/Qwen2.5-0.5B-Instruct"

DATA_DIR = "data"
INDEX_DIR = "index"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
TOP_K = 5
MIN_SCORE = 0.50
RRF_K = 60

SYSTEM_PROMPT = (
    "You are an assistant for wind and solar resource assessment engineers.\n"
    "Rules:\n"
    "1. Answer ONLY using the numbered context passages.\n"
    "2. Cite the passage number after each fact, like [1].\n"
    "3. If the context does not contain the answer, reply exactly: "
    "'I could not find this in the documents.'\n"
    "4. Be concise: at most 5 sentences."
)
NOT_FOUND = "I could not find this in the documents."


# ---------------------------------------------------------------- ingestion
def load_documents(folder=DATA_DIR):
    """Load .txt and .pdf files as a list of {text, source, page}."""
    pages = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        name = os.path.basename(path)
        if name.lower().endswith(".pdf"):
            for pno, page in enumerate(PdfReader(path).pages, 1):
                text = (page.extract_text() or "").replace("\n", " ")
                if text.strip():
                    pages.append({"text": text, "source": name, "page": pno})
        elif name.lower().endswith(".txt"):
            with open(path, encoding="utf-8") as f:
                pages.append({"text": f.read().replace("\n", " "), "source": name, "page": 1})
    return pages


def chunk_pages(pages, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split pages into overlapping chunks, breaking on word boundaries."""
    chunks, step = [], chunk_size - overlap
    for p in pages:
        text = " ".join(p["text"].split())
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                space = text.rfind(" ", start, end)
                if space > start + step // 2:
                    end = space
            piece = text[start:end].strip()
            if len(piece) > 50:
                chunks.append({"text": piece, "source": p["source"], "page": p["page"]})
            if end >= len(text):
                break
            start = max(end - overlap, start + 1)
    return chunks


def build_index(chunks, embedder):
    vectors = embedder.encode([c["text"] for c in chunks], normalize_embeddings=True,
                              batch_size=32, show_progress_bar=False)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def tokenize(text):
    return re.findall(r"[a-z0-9]+(?:[-.][a-z0-9]+)*", text.lower())


# ---------------------------------------------------------------- pipeline
class RAGAssistant:
    def __init__(self, index_dir=INDEX_DIR, embedder=None):
        self.embedder = embedder or SentenceTransformer(EMBED_MODEL)
        index_path = os.path.join(index_dir, "index.faiss")
        chunks_path = os.path.join(index_dir, "chunks.pkl")
        if not os.path.exists(index_path):
            raise FileNotFoundError("Index not found. Run `python ingest.py` first.")
        self.index = faiss.read_index(index_path)
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self._llm = None

    # ---- retrieval
    def vector_search(self, query, k=TOP_K):
        q = self.embedder.encode([query], normalize_embeddings=True)
        scores, ids = self.index.search(np.asarray(q, dtype="float32"), k)
        return [(self.chunks[i], float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    def bm25_search(self, query, k=TOP_K):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def hybrid_search(self, query, k=TOP_K, pool=20):
        """Reciprocal Rank Fusion of vector and BM25 results; returns vector similarity as score."""
        vec = self.vector_search(query, pool)
        fused = {}
        for ranked in (vec, self.bm25_search(query, pool)):
            for rank, (c, _) in enumerate(ranked):
                entry = fused.setdefault(id(c), [c, 0.0])
                entry[1] += 1.0 / (RRF_K + rank + 1)
        best = sorted(fused.values(), key=lambda x: -x[1])[:k]
        vec_score = {id(c): s for c, s in vec}
        return [(c, vec_score.get(id(c), 0.0)) for c, _ in best]

    # ---- generation
    def _connect_llm(self):
        token = os.environ.get("HF_TOKEN", "")
        if token:
            from huggingface_hub import InferenceClient
            client = InferenceClient(token=token)
            for name in HOSTED_LLMS:
                try:
                    client.chat_completion(model=name, max_tokens=5,
                                           messages=[{"role": "user", "content": "OK"}])
                    return ("hosted", name, client)
                except Exception:
                    continue
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(LOCAL_LLM)
        mdl = AutoModelForCausalLM.from_pretrained(LOCAL_LLM)
        return ("local", LOCAL_LLM, (tok, mdl))

    @property
    def llm_name(self):
        return self._llm[1] if self._llm else None

    def generate(self, messages, max_tokens=400, temperature=0.1):
        if self._llm is None:
            self._llm = self._connect_llm()
        mode, name, backend = self._llm
        if mode == "hosted":
            out = backend.chat_completion(model=name, messages=messages,
                                          max_tokens=max_tokens, temperature=temperature)
            return out.choices[0].message.content.strip()
        tok, mdl = backend
        inputs = tok.apply_chat_template(messages, add_generation_prompt=True,
                                         return_tensors="pt", return_dict=True)
        out = mdl.generate(**inputs, max_new_tokens=max_tokens, do_sample=False)
        return tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    def answer(self, question, k=TOP_K, hybrid=True, min_score=MIN_SCORE):
        """Return (answer, hits). Refuses without calling the LLM if retrieval confidence is low."""
        hits = (self.hybrid_search if hybrid else self.vector_search)(question, k)
        if not hits or max(s for _, s in hits) < min_score:
            return NOT_FOUND, hits
        context = "\n\n".join(f"[{n}] (source: {c['source']}, page {c['page']})\n{c['text']}"
                              for n, (c, _) in enumerate(hits, 1))
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ]
        return self.generate(messages), hits
