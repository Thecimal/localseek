# Changelog

## 0.1.0

First release.

- Index folders of `.txt`, `.md`, `.pdf`, `.docx`, `.html`, `.epub`, and common source files
- Hybrid search: embeddings plus SQLite FTS5 keyword search, merged with Reciprocal Rank Fusion
- Incremental indexing, with cached embeddings for unchanged text
- Filters by file type, modified date, and path
- `watch` mode and a local search page (`serve`)
- Evaluation script for recall@k and MRR
