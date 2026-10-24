Evaluation on 20 questions (k=5)

| Chunk size | Chunks | Hit@5 | MRR |
|---:|---:|---:|---:|
| 200 | 75 | 100% | 0.97 |
| 350 | 43 | 100% | 1.00 |
| 500 | 31 | 100% | 1.00 |
| 800 | 20 | 100% | 0.97 |
| 1200 | 14 | 100% | 0.90 |

Search method (chunk size 350)

| Method | Hit@5 | MRR |
|---|---:|---:|
| Vector (FAISS) | 100% | 1.00 |
| Keyword (BM25) | 100% | 0.95 |
| Hybrid (RRF) | 100% | 1.00 |
