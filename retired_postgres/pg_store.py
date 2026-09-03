"""
PostgreSQL + pgvector storage layer.

This is the ONLY module that talks to the database. Ingestion and retrieval
both go through it, so the vector store can be changed in one place.

Replaces the previous Qdrant layer. Design notes:

  * Cosine distance (`<=>`) with L2-normalised embeddings, so the similarity
    score returned here (1 - distance) is directly comparable to the cosine
    scores the Qdrant implementation produced.
  * chunk_id carries a UNIQUE constraint and inserts use ON CONFLICT DO UPDATE.
    Re-running ingestion therefore overwrites rows in place instead of
    appending duplicates - the failure mode that silently tripled the old
    Qdrant collection.
  * Every statement is parameterised. Identifiers that must be interpolated
    (table/index names) come from config, never from user input, and are
    passed through psycopg2.sql.Identifier.
"""

import psycopg2
from psycopg2 import sql
from psycopg2.extras import execute_values, RealDictCursor

from config import (
    POSTGRES_HOST, POSTGRES_PORT, POSTGRES_DB, POSTGRES_USER,
    POSTGRES_PASSWORD, POSTGRES_SCHEMA, DATABASE_URL,
    CHUNKS_TABLE, EMBEDDING_DIM, VECTOR_INDEX_TYPE,
)


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------

class PgStoreError(RuntimeError):
    """Any configuration/connection/schema problem, reported with a fix."""


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------

def _dsn_parts():
    if not POSTGRES_PASSWORD and not DATABASE_URL:
        raise PgStoreError(
            "POSTGRES_PASSWORD is not set.\n"
            "  Copy .env.example to .env and fill in your PostgreSQL password,\n"
            "  or export POSTGRES_PASSWORD in the environment.\n"
            "  Credentials are never stored in the source tree."
        )
    return dict(host=POSTGRES_HOST, port=POSTGRES_PORT, dbname=POSTGRES_DB,
                user=POSTGRES_USER, password=POSTGRES_PASSWORD)


def connect(dbname=None, timeout=10):
    """
    Open a connection, translating driver errors into actionable messages.

    Never falls back to another database: if PostgreSQL is unreachable the
    pipeline must stop, not quietly write somewhere else.
    """
    try:
        if DATABASE_URL and dbname is None:
            return psycopg2.connect(DATABASE_URL, connect_timeout=timeout)
        parts = _dsn_parts()
        if dbname:
            parts["dbname"] = dbname
        return psycopg2.connect(connect_timeout=timeout, **parts)
    except psycopg2.OperationalError as e:
        msg = str(e).strip()
        low = msg.lower()
        if "could not connect" in low or "connection refused" in low or "timeout" in low:
            raise PgStoreError(
                f"Cannot reach PostgreSQL at {POSTGRES_HOST}:{POSTGRES_PORT}.\n"
                f"  Is the server running?  "
                f"(Windows: Get-Service *postgres*)\n"
                f"  Driver said: {msg}") from e
        if "password" in low or "authentication" in low:
            raise PgStoreError(
                f"Authentication failed for user '{POSTGRES_USER}'.\n"
                f"  Check POSTGRES_USER / POSTGRES_PASSWORD in your .env.\n"
                f"  Driver said: {msg}") from e
        if "does not exist" in low and "database" in low:
            raise PgStoreError(
                f"Database '{dbname or POSTGRES_DB}' does not exist.\n"
                f"  Create it with:  py -3.14 pg_store.py init\n"
                f"  or in pgAdmin:   CREATE DATABASE {POSTGRES_DB};\n"
                f"  Driver said: {msg}") from e
        raise PgStoreError(f"PostgreSQL connection failed: {msg}") from e


# --------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------

def ensure_database():
    """Create the target database if it is missing (connects via 'postgres')."""
    con = connect(dbname="postgres")
    con.autocommit = True
    try:
        with con.cursor() as cur:
            cur.execute("SELECT 1 FROM pg_database WHERE datname = %s",
                        (POSTGRES_DB,))
            if cur.fetchone():
                print(f"  database '{POSTGRES_DB}' already exists")
                return False
            cur.execute(sql.SQL("CREATE DATABASE {}").format(
                sql.Identifier(POSTGRES_DB)))
            print(f"  created database '{POSTGRES_DB}'")
            return True
    finally:
        con.close()


def ensure_extension(con):
    """Enable pgvector, with a real installation hint when it is missing."""
    with con.cursor() as cur:
        cur.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
        if not cur.fetchone():
            raise PgStoreError(
                "The pgvector extension is NOT available on this PostgreSQL server.\n"
                "  'CREATE EXTENSION vector' cannot succeed until the server-side\n"
                "  files (vector.control, vector.dll) are installed into the\n"
                "  PostgreSQL installation directory.\n"
                "  See README.md -> 'Installing pgvector' for the steps.\n"
                "  Verify with:  SELECT * FROM pg_available_extensions "
                "WHERE name = 'vector';")
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
        row = cur.fetchone()
    con.commit()
    return row[0] if row else None


def _table():
    return sql.Identifier(POSTGRES_SCHEMA, CHUNKS_TABLE)


DDL = """
CREATE TABLE IF NOT EXISTS {table} (
    id              BIGSERIAL PRIMARY KEY,
    chunk_id        TEXT        NOT NULL UNIQUE,
    book_id         TEXT        NOT NULL,
    book_title      TEXT        NOT NULL,
    author          TEXT,
    edition         TEXT,
    chapter         TEXT,
    section         TEXT,
    subsection      TEXT,
    section_path    TEXT,
    page_start      TEXT,
    page_end        TEXT,
    pdf_page_start  INTEGER,
    pdf_page_end    INTEGER,
    page_label_source TEXT,
    content_type    TEXT,
    subject_area    TEXT,
    content         TEXT        NOT NULL,
    n_chars         INTEGER,
    n_tokens        INTEGER,
    source_file     TEXT,
    corpus_version  TEXT,
    chunk_config_id TEXT,
    extraction_method TEXT,
    embedding       VECTOR({dim}) NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""


def ensure_schema(con):
    """Create the table and its indexes. Safe to call repeatedly."""
    with con.cursor() as cur:
        cur.execute(sql.SQL(DDL).format(table=_table(),
                                        dim=sql.Literal(EMBEDDING_DIM)))
        # metadata indexes for filtering / citation lookups
        for col in ("book_id", "content_type", "subject_area"):
            cur.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {table} ({col})").format(
                name=sql.Identifier(f"idx_{CHUNKS_TABLE}_{col}"),
                table=_table(), col=sql.Identifier(col)))
    con.commit()


def ensure_vector_index(con):
    """
    Build the ANN index for cosine distance.

    Optional at this corpus size - a sequential scan over a few thousand rows
    is already fast - but it is what makes the design scale to the full
    25-book corpus without changing any query.
    """
    if VECTOR_INDEX_TYPE.lower() == "none":
        print("  vector index: skipped (VECTOR_INDEX_TYPE=none)")
        return
    name = sql.Identifier(f"idx_{CHUNKS_TABLE}_embedding")
    kind = VECTOR_INDEX_TYPE.lower()
    with con.cursor() as cur:
        if kind == "hnsw":
            cur.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {table} "
                "USING hnsw (embedding vector_cosine_ops)"
            ).format(name=name, table=_table()))
        elif kind == "ivfflat":
            cur.execute(sql.SQL(
                "CREATE INDEX IF NOT EXISTS {name} ON {table} "
                "USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)"
            ).format(name=name, table=_table()))
        else:
            raise PgStoreError(
                f"VECTOR_INDEX_TYPE must be hnsw, ivfflat or none "
                f"(got {VECTOR_INDEX_TYPE!r})")
    con.commit()
    print(f"  vector index: {kind} (cosine)")


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

INSERT_COLUMNS = [
    "chunk_id", "book_id", "book_title", "author", "edition",
    "chapter", "section", "subsection", "section_path",
    "page_start", "page_end", "pdf_page_start", "pdf_page_end",
    "page_label_source", "content_type", "subject_area",
    "content", "n_chars", "n_tokens", "source_file",
    "corpus_version", "chunk_config_id", "extraction_method", "embedding",
]


def _vec_literal(vec):
    """pgvector text form: '[0.1,0.2,...]'. Passed as a parameter, then cast."""
    return "[" + ",".join(f"{float(x):.7g}" for x in vec) + "]"


def delete_book(con, book_id):
    """Remove one book's rows so it can be re-ingested cleanly."""
    with con.cursor() as cur:
        cur.execute(sql.SQL("DELETE FROM {table} WHERE book_id = %s").format(
            table=_table()), (book_id,))
        n = cur.rowcount
    con.commit()
    return n


def insert_chunks(con, chunks, vectors, batch_size=500):
    """
    Batch-insert chunks with their embeddings.

    ON CONFLICT (chunk_id) DO UPDATE makes this idempotent: re-running
    ingestion updates each row in place instead of duplicating it.
    """
    if len(chunks) != len(vectors):
        raise PgStoreError(
            f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}")

    rows = []
    for c, v in zip(chunks, vectors):
        if len(v) != EMBEDDING_DIM:
            raise PgStoreError(
                f"Embedding dimension mismatch for {c['chunk_id']}: "
                f"got {len(v)}, table expects VECTOR({EMBEDDING_DIM}). "
                f"Check EMBEDDING_MODEL / EMBEDDING_DIM in config.py.")
        rows.append((
            c["chunk_id"], c["book_id"], c["book"], c.get("author"),
            c.get("edition"), c.get("chapter"), c.get("section"),
            c.get("subsection"), c.get("section_path"),
            c.get("page_start"), c.get("page_end"),
            c.get("pdf_page_start"), c.get("pdf_page_end"),
            c.get("page_label_source"), c.get("content_type"),
            c.get("subject_area"), c["text"], c.get("n_chars"),
            c.get("n_tokens"), c.get("source_file"), c.get("corpus_version"),
            c.get("chunk_config_id"), c.get("extraction_method"),
            _vec_literal(v),
        ))

    updates = sql.SQL(", ").join(
        sql.SQL("{c} = EXCLUDED.{c}").format(c=sql.Identifier(col))
        for col in INSERT_COLUMNS if col != "chunk_id")

    stmt = sql.SQL(
        "INSERT INTO {table} ({cols}) VALUES %s "
        "ON CONFLICT (chunk_id) DO UPDATE SET {updates}"
    ).format(
        table=_table(),
        cols=sql.SQL(", ").join(map(sql.Identifier, INSERT_COLUMNS)),
        updates=updates,
    )

    template = "(" + ", ".join(
        ["%s"] * (len(INSERT_COLUMNS) - 1) + ["%s::vector"]) + ")"

    inserted = 0
    with con.cursor() as cur:
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i + batch_size]
            execute_values(cur, stmt.as_string(con), batch,
                           template=template, page_size=batch_size)
            inserted += len(batch)
    con.commit()
    return inserted


def count_rows(con, book_id=None):
    with con.cursor() as cur:
        if book_id:
            cur.execute(sql.SQL("SELECT count(*) FROM {table} WHERE book_id = %s")
                        .format(table=_table()), (book_id,))
        else:
            cur.execute(sql.SQL("SELECT count(*) FROM {table}").format(table=_table()))
        return cur.fetchone()[0]


# --------------------------------------------------------------------------
# Searching
# --------------------------------------------------------------------------

SEARCH_COLUMNS = [
    "chunk_id", "book_id", "book_title", "author", "edition",
    "chapter", "section", "subsection", "section_path",
    "page_start", "page_end", "pdf_page_start", "pdf_page_end",
    "page_label_source", "content_type", "subject_area",
    "content", "n_chars", "n_tokens",
]


def search(con, query_vector, limit=5, book_id=None, content_types=None):
    """
    Cosine similarity search.

    `<=>` is pgvector's cosine DISTANCE, so similarity = 1 - distance. With
    L2-normalised embeddings this is the same score Qdrant's COSINE metric
    returned, which keeps old and new results directly comparable.
    """
    if query_vector is None or len(query_vector) != EMBEDDING_DIM:
        raise PgStoreError(
            f"Query vector must have {EMBEDDING_DIM} dimensions, "
            f"got {0 if query_vector is None else len(query_vector)}")

    where, params = [], [_vec_literal(query_vector)]
    if book_id:
        where.append(sql.SQL("book_id = %s"))
        params.append(book_id)
    if content_types:
        where.append(sql.SQL("content_type = ANY(%s)"))
        params.append(list(content_types))
    clause = (sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where)) if where \
        else sql.SQL("")
    params.append(limit)

    stmt = sql.SQL(
        "SELECT {cols}, 1 - (embedding <=> %s::vector) AS score "
        "FROM {table}{where} "
        "ORDER BY embedding <=> %s::vector "
        "LIMIT %s"
    ).format(
        cols=sql.SQL(", ").join(map(sql.Identifier, SEARCH_COLUMNS)),
        table=_table(), where=clause,
    )
    # the vector appears twice (score + ORDER BY); rebuild the parameter list
    ordered = [params[0]] + params[1:-1] + [params[0], params[-1]]

    with con.cursor(cursor_factory=RealDictCursor) as cur:
        cur.execute(stmt, ordered)
        return [dict(r) for r in cur.fetchall()]


# --------------------------------------------------------------------------
# CLI: py -3.14 pg_store.py init | check
# --------------------------------------------------------------------------

def init():
    """Create database + extension + schema. Idempotent."""
    print(f"PostgreSQL {POSTGRES_HOST}:{POSTGRES_PORT}  db={POSTGRES_DB} "
          f"user={POSTGRES_USER}")
    print("\n[1/4] Ensuring database ...")
    ensure_database()
    con = connect()
    try:
        print("[2/4] Ensuring pgvector extension ...")
        ver = ensure_extension(con)
        print(f"  pgvector version: {ver}")
        print("[3/4] Ensuring table + metadata indexes ...")
        ensure_schema(con)
        print(f"  table ready: {POSTGRES_SCHEMA}.{CHUNKS_TABLE} "
              f"(embedding VECTOR({EMBEDDING_DIM}))")
        print("[4/4] Ensuring vector index ...")
        ensure_vector_index(con)
        print(f"\nRows currently stored: {count_rows(con)}")
        print("Database is ready.")
    finally:
        con.close()


def check():
    """Report server/extension/table status without changing anything."""
    print(f"host={POSTGRES_HOST} port={POSTGRES_PORT} db={POSTGRES_DB} "
          f"user={POSTGRES_USER}")
    con = connect()
    try:
        with con.cursor() as cur:
            cur.execute("SELECT version()")
            print("server   :", cur.fetchone()[0].split(",")[0])
            cur.execute("SELECT default_version, installed_version "
                        "FROM pg_available_extensions WHERE name='vector'")
            row = cur.fetchone()
            print("pgvector :", "NOT AVAILABLE" if not row else
                  f"available {row[0]}, installed {row[1]}")
            cur.execute("SELECT to_regclass(%s)",
                        (f"{POSTGRES_SCHEMA}.{CHUNKS_TABLE}",))
            exists = cur.fetchone()[0]
            print("table    :", exists or "MISSING")
            if exists:
                print("rows     :", count_rows(con))
                cur.execute(sql.SQL(
                    "SELECT book_id, count(*) FROM {table} GROUP BY book_id "
                    "ORDER BY 2 DESC").format(table=_table()))
                for book, n in cur.fetchall():
                    print(f"           {book}: {n}")
    finally:
        con.close()


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    try:
        {"init": init, "check": check}[cmd]()
    except KeyError:
        print("Usage: py -3.14 pg_store.py [init|check]")
        sys.exit(2)
    except PgStoreError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
