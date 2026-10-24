"""Build the vector index from documents in data/."""

import argparse
import os
import pickle
import time

import faiss
from sentence_transformers import SentenceTransformer

from rag import CHUNK_OVERLAP, CHUNK_SIZE, DATA_DIR, EMBED_MODEL, INDEX_DIR, build_index, chunk_pages, load_documents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default=DATA_DIR)
    parser.add_argument("--out", default=INDEX_DIR)
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=CHUNK_OVERLAP)
    args = parser.parse_args()

    t0 = time.time()
    pages = load_documents(args.data)
    chunks = chunk_pages(pages, args.chunk_size, args.overlap)
    index = build_index(chunks, SentenceTransformer(EMBED_MODEL))

    os.makedirs(args.out, exist_ok=True)
    faiss.write_index(index, os.path.join(args.out, "index.faiss"))
    with open(os.path.join(args.out, "chunks.pkl"), "wb") as f:
        pickle.dump(chunks, f)

    n_files = len({p["source"] for p in pages})
    print(f"Indexed {len(chunks)} chunks from {n_files} files in {time.time() - t0:.1f}s -> {args.out}/")


if __name__ == "__main__":
    main()
