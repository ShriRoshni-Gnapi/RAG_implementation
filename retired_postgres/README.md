# Retired: PostgreSQL + pgvector implementation

This directory holds the previous vector-store implementation, kept for
reference after the project reverted to Qdrant. **Nothing in the active
pipeline imports it.**

- `pg_store.py` — PostgreSQL + pgvector storage layer
- `sql/schema.sql` — `document_chunks` DDL (VECTOR(384), HNSW cosine index)

It was verified working: 2,957 rows ingested, and all 7 benchmark queries
returned scores identical to Qdrant to 4 decimal places.

To use it again you would need to restore the PostgreSQL settings in
`config.py`, point `embed_store.py` / `retrieve_test.py` back at `pg_store`,
and run a pgvector-capable PostgreSQL:

```powershell
docker start pt_rag_pg      # container still exists on port 5433
```

Delete this directory (and that container) if you have no plans to go back.
