# Your Project

## Quickstart
1. `make up` (postgres+pgvector, ollama, phoenix, openwebui)
2. Put PDFs in `data/raw/`.
3. `make ingest` (Docling → HybridChunker → size post-pass → dedup → PGVector)
4. `make api` then POST to `/ask` with `{"question":"..."}`
5. Optional: `make eval` to run RAGAs.

> Ensure Python 3.10+, install deps: `pip install -e .`
