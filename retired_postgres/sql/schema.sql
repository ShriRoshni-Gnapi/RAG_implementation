-- ---------------------------------------------------------------------------
-- pt_rag schema: PostgreSQL + pgvector
--
-- pg_store.py applies all of this automatically (`py -3.14 pg_store.py init`).
-- This file is the reference copy, for running by hand in pgAdmin or psql.
-- ---------------------------------------------------------------------------

-- 1. Create the database (run from the 'postgres' database).
--    CREATE DATABASE cannot run inside a transaction block.
-- CREATE DATABASE pt_rag;

-- 2. Connect to pt_rag, then enable pgvector.
CREATE EXTENSION IF NOT EXISTS vector;

-- Verify the extension is available BEFORE creating the table:
--   SELECT * FROM pg_available_extensions WHERE name = 'vector';
--     -> no row  = server-side pgvector files are not installed
--     -> a row   = installable
--   SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
--     -> confirms it is enabled in THIS database

-- 3. The chunk table.
--    embedding is VECTOR(384) to match all-MiniLM-L6-v2.
--    chunk_id is UNIQUE so ingestion can upsert instead of duplicating.
CREATE TABLE IF NOT EXISTS document_chunks (
    id                BIGSERIAL PRIMARY KEY,
    chunk_id          TEXT        NOT NULL UNIQUE,
    book_id           TEXT        NOT NULL,
    book_title        TEXT        NOT NULL,
    author            TEXT,
    edition           TEXT,
    chapter           TEXT,
    section           TEXT,
    subsection        TEXT,
    section_path      TEXT,
    page_start        TEXT,          -- printed page label, e.g. 'xii' or '325'
    page_end          TEXT,
    pdf_page_start    INTEGER,       -- 0-based PDF index, always available
    pdf_page_end      INTEGER,
    page_label_source TEXT,          -- 'label' | 'index'
    content_type      TEXT,          -- prose | references | front_matter | ...
    subject_area      TEXT,
    content           TEXT        NOT NULL,
    n_chars           INTEGER,
    n_tokens          INTEGER,
    source_file       TEXT,
    corpus_version    TEXT,
    chunk_config_id   TEXT,
    extraction_method TEXT,          -- text | ocr | mixed
    embedding         VECTOR(384) NOT NULL,
    created_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- 4. Metadata indexes (filtering and citation lookups).
CREATE INDEX IF NOT EXISTS idx_document_chunks_book_id
    ON document_chunks (book_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_content_type
    ON document_chunks (content_type);
CREATE INDEX IF NOT EXISTS idx_document_chunks_subject_area
    ON document_chunks (subject_area);

-- 5. ANN index for cosine similarity.
--    Must match the operator used at query time (<=> = cosine distance).
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

-- Alternative for very large tables (build AFTER loading data):
-- CREATE INDEX idx_document_chunks_embedding
--     ON document_chunks USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 100);

-- ---------------------------------------------------------------------------
-- Example query: top-5 nearest chunks to a 384-d query vector.
-- `<=>` is cosine DISTANCE, so similarity = 1 - distance.
-- ---------------------------------------------------------------------------
-- SELECT chunk_id, book_title, chapter, section, page_start, page_end,
--        1 - (embedding <=> $1::vector) AS score,
--        content
-- FROM document_chunks
-- WHERE book_id = $2
-- ORDER BY embedding <=> $1::vector
-- LIMIT 5;

-- ---------------------------------------------------------------------------
-- Handy checks in pgAdmin
-- ---------------------------------------------------------------------------
-- SELECT count(*) FROM document_chunks;
-- SELECT book_id, count(*) FROM document_chunks GROUP BY book_id;
-- SELECT vector_dims(embedding) FROM document_chunks LIMIT 1;   -- expect 384
-- SELECT chunk_id, chapter, section, page_start
--   FROM document_chunks ORDER BY chunk_id LIMIT 10;
