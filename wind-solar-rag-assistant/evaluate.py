"""Evaluate retrieval quality (Hit@k, MRR) across chunk sizes and search methods."""

import json
import os
import pickle
import tempfile

import faiss
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

from rag import EMBED_MODEL, TOP_K, RAGAssistant, build_index, chunk_pages, load_documents

CHUNK_SIZES = [200, 350, 500, 800, 1200]


def score(search_fn, eval_set, k=TOP_K):
    hits, rr = 0, 0.0
    for item in eval_set:
        sources = [c["source"] for c, _ in search_fn(item["question"], k)]
        if item["source"] in sources:
            hits += 1
            rr += 1.0 / (sources.index(item["source"]) + 1)
    n = len(eval_set)
    return hits / n, rr / n


def assistant_for(chunk_size, embedder, pages):
    chunks = chunk_pages(pages, chunk_size, chunk_size // 5)
    tmp = tempfile.mkdtemp()
    faiss.write_index(build_index(chunks, embedder), os.path.join(tmp, "index.faiss"))
    with open(os.path.join(tmp, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)
    return RAGAssistant(tmp, embedder=embedder), len(chunks)


def main():
    with open("eval/questions.json", encoding="utf-8") as f:
        eval_set = json.load(f)
    embedder = SentenceTransformer(EMBED_MODEL)
    pages = load_documents()

    size_rows = []
    for size in CHUNK_SIZES:
        ra, n = assistant_for(size, embedder, pages)
        size_rows.append((size, n, *score(ra.vector_search, eval_set)))

    best_size = max(size_rows, key=lambda r: r[3])[0]
    ra, _ = assistant_for(best_size, embedder, pages)
    method_rows = [(name, *score(fn, eval_set)) for name, fn in [
        ("Vector (FAISS)", ra.vector_search),
        ("Keyword (BM25)", ra.bm25_search),
        ("Hybrid (RRF)", ra.hybrid_search),
    ]]

    lines = [f"Evaluation on {len(eval_set)} questions (k={TOP_K})", "",
             "| Chunk size | Chunks | Hit@5 | MRR |", "|---:|---:|---:|---:|"]
    lines += [f"| {s} | {n} | {h:.0%} | {m:.2f} |" for s, n, h, m in size_rows]
    lines += ["", f"Search method (chunk size {best_size})", "",
              "| Method | Hit@5 | MRR |", "|---|---:|---:|"]
    lines += [f"| {name} | {h:.0%} | {m:.2f} |" for name, h, m in method_rows]
    report = "\n".join(lines)
    print(report)
    os.makedirs("assets", exist_ok=True)
    with open("eval/results.md", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    axes[0].bar([str(r[0]) for r in size_rows], [r[3] for r in size_rows], color="#4C78A8")
    axes[0].set(title="MRR by chunk size (vector search)", xlabel="Chunk size (characters)", ylabel="MRR", ylim=(0, 1.05))
    colors = ["#9AA5B1", "#9AA5B1", "#F58518"]
    axes[1].bar([r[0] for r in method_rows], [r[2] for r in method_rows], color=colors)
    axes[1].set(title=f"MRR by search method (chunk size {best_size})", ylabel="MRR", ylim=(0, 1.05))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2, p.get_height()),
                        ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig("assets/evaluation_results.png", dpi=150)
    print("\nSaved eval/results.md and assets/evaluation_results.png")


if __name__ == "__main__":
    main()
