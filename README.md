# pt_rag — PT Textbook RAG Pipeline

Turns a physiotherapy textbook PDF into a searchable, citable vector index
backed by **Qdrant** (server mode, `http://localhost:6333`).

```text
PDF → extract → clean → structure → chunk → all-MiniLM-L6-v2 (384-d)
    → Qdrant → retrieve relevant chunks
```

MedGemma is **not** part of this pipeline yet. This stage validates everything
up to and including retrieval.

---

## 1. Start the Qdrant server

Qdrant runs as a server on `localhost:6333`:

```powershell
docker run -d --name pt_rag_qdrant `
  -p 6333:6333 -p 6334:6334 `
  -v pt_rag_qdrant_data:/qdrant/storage `
  qdrant/qdrant
```

Already created? Just start it:

```powershell
docker start pt_rag_qdrant
```

Data lives on the named volume `pt_rag_qdrant_data`, so it survives restarts.

## 2. Verify Qdrant is reachable

```powershell
py -3.14 qdrant_store.py check
```

Or in a browser:

- **Dashboard / collection browser:** <http://localhost:6333/dashboard>
- REST: <http://localhost:6333/collections>

## 3. Collections

Nothing to pre-create. One collection per book is created during ingestion,
named `<COLLECTION_PREFIX><book_key>` — e.g. `ptbook_paz_acute_care`.

Vectors: **384-d, COSINE distance**. Qdrant builds its own HNSW index.

## 4. Environment variables

Copy the template and adjust if needed:

```powershell
Copy-Item .env.example .env
```

| Variable | Default | Meaning |
|---|---|---|
| `QDRANT_URL` | `http://localhost:6333` | Server URL |
| `QDRANT_API_KEY` | *(empty)* | Only for Qdrant Cloud / secured instances |
| `QDRANT_TIMEOUT` | `60` | Request timeout (seconds) |
| `COLLECTION_PREFIX` | `ptbook_` | Collection naming |
| `PT_BOOKS_DIR` | *(auto)* | Override where source PDFs live |

A local Qdrant needs **no credentials**. Real environment variables take
precedence over `.env`, which is git-ignored.

Source PDFs are found in `../books/` if that folder exists, otherwise in the
parent directory.

## 5. Python dependencies

```powershell
py -3.14 -m pip install -r requirements.txt
```

## 6. Running ingestion

One command runs the whole pipeline:

```powershell
cd "c:\Users\Shri Roshni\Downloads\References and Books\pt_rag"

py -3.14 main.py paz_acute_care
```

Stages: preflight (Qdrant reachable) → ingest → embed + store → retrieval test.
It stops at the first failure, so a stopped Qdrant container fails in seconds
rather than after minutes of ingestion.

Works for any book key registered in `config.py`. Run `py -3.14 main.py` with
no arguments to list them.

| Exit code | Meaning |
|---|---|
| `0` | All stages succeeded |
| `1` | A stage failed (the stage is named and the error printed) |
| `2` | Unknown or missing book key — nothing ran |

Individual stages remain available for debugging:

```powershell
py -3.14 qdrant_store.py  check
py -3.14 ingest_book.py   paz_acute_care
py -3.14 embed_store.py   paz_acute_care
py -3.14 retrieve_test.py paz_acute_care
```

`main.py` imports these modules' own functions rather than duplicating logic.

## 7. Running retrieval

Built-in suite of 7 queries (writes a JSON report):

```powershell
py -3.14 retrieve_test.py paz_acute_care
```

Ask your own question — no source editing required:

```powershell
py -3.14 retrieve_test.py paz_acute_care "What are the precautions for mobilizing a patient with DVT?"
py -3.14 retrieve_test.py paz_acute_care "What causes pressure ulcers?" --top-k 10
py -3.14 retrieve_test.py "compare open and closed chain exercise" --all-books
```

Output format:

```text
Question:
    ...

Top Results:

Rank 1
  Chunk   : paz_acute_care_01222
  Score   : 0.6319
  Book    : Acute Care Handbook for Physical Therapists
  Chapter : 2 Body Systems
  Section : 7 Vascular System and Hematology
  Pages   : 180-181 (pdf 196, src=label)
  Content : ...
```

## 8. Inspecting the data

Open the **Qdrant dashboard**: <http://localhost:6333/dashboard>

It lists collections, lets you browse points with their payloads, and run
filtered searches — the equivalent of what pgAdmin gave the SQL build.

From the command line:

```powershell
py -3.14 qdrant_store.py check
```

Or via REST:

```powershell
curl http://localhost:6333/collections
curl http://localhost:6333/collections/ptbook_paz_acute_care
```

## 9. Stored payload

Every point carries the full citation chain plus provenance:

| Field | Notes |
|---|---|
| `chunk_id` | Stable business key (also seeds the deterministic point id) |
| `book_id`, `book`, `author`, `edition` | Citation identity |
| `chapter`, `section`, `subsection`, `section_path` | Structure |
| `page_start`, `page_end` | Printed page labels (e.g. `xii`, `325`) |
| `pdf_page_start`, `pdf_page_end` | PDF indices, always available |
| `page_label_source` | `label` or `index` — prevents misleading citations |
| `content_type`, `subject_area` | Filtering (payload-indexed) |
| `text` | Chunk content |
| `n_chars`, `n_tokens` | Size accounting |
| `source_file`, `corpus_version`, `chunk_config_id`, `extraction_method` | Provenance |

Payload indexes exist on `book_id`, `content_type` and `subject_area`.

## 10. How the architecture works now

```text
                 ingestion (offline)
  PDF ─► PyMuPDF extract ─► clean ─► detect structure (PDF outline)
      ─► token-aware chunk ─► metadata ─► all-MiniLM-L6-v2
      ─► upsert (deterministic point ids)
      ─► Qdrant collection ptbook_<book>  (384-d, COSINE)

                 retrieval (per query)
  question ─► all-MiniLM-L6-v2 ─► 384-d vector
           ─► Qdrant query_points(limit=k)
           ─► top-k chunks + Book/Chapter/Section/Page citation
```

`qdrant_store.py` is the only module that touches the vector database, so the
store can be swapped in one place.

**Similarity metric.** Embeddings are L2-normalised and the collection uses
COSINE distance, so `score` is cosine similarity. Verified: all 7 benchmark
queries scored identically to 4 decimal places under both Qdrant and the
previous pgvector implementation.

---

## Things worth knowing

**The 256-token ceiling is the binding constraint.** `all-MiniLM-L6-v2`
truncates at 256 tokens with no error. Chunking is therefore token-aware, not
character-aware, and both `ingest_book.py` and `embed_store.py` abort if any
chunk exceeds 250 tokens. Measured token density for this medical prose is
0.239 tokens/char on average and up to 0.465 — about 1.6× what a naive
character estimate assumes.

**Re-running is safe.** `embed_store.py` recreates the collection and upserts
with deterministic point ids (`uuid5` of `book_id/chunk_id`), so the point
count stays constant across runs. This matters: an earlier build used Qdrant's
*embedded* (local-path) mode, where `delete_collection()` reported success but
`create_collection()` restored the points from disk — combined with random
point ids, every re-run silently appended a duplicate copy of the book. Running
Qdrant as a **server** avoids that defect entirely.

**References and front matter are excluded from the index**, not deleted. They
remain in the chunk JSON tagged by `content_type`; adjust
`EXCLUDED_CONTENT_TYPES` in `config.py` to change this.

**The PostgreSQL + pgvector implementation is retired**, not deleted. It lives
in `retired_postgres/` and nothing in the active pipeline imports it. Its
container (`pt_rag_pg`, port 5433) also still exists. Remove both when you are
sure you will not go back:

```powershell
Remove-Item -Recurse -Force retired_postgres
docker rm -f pt_rag_pg; docker volume rm pt_rag_pgdata
```

**`data/qdrant/`** is the older *embedded* Qdrant store (~48 MB), superseded by
the server. Nothing reads it:

```powershell
Remove-Item -Recurse -Force data\qdrant
```

## Requirements

- **Python 3.14** — use `py -3.14`; `python` is not on PATH and the
  dependencies are installed under 3.14.
- Docker, for the Qdrant server (sections 1–2).
- The `all-MiniLM-L6-v2` model is already in the local HuggingFace cache, so
  the pipeline runs offline.
