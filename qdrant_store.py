"""
Qdrant storage layer (server mode, http://localhost:6333).

This is the ONLY module that talks to the vector database. Ingestion and
retrieval both go through it, so the store can be swapped in one place.

Design notes:

  * Talks to a Qdrant SERVER over HTTP, like the sibling transcript project.
    The earlier embedded (local path) mode is not used: it had a defect where
    delete_collection() reported success but create_collection() restored the
    points from disk, which silently triplicated the collection.
  * Point IDs are deterministic (uuid5 of book_id/chunk_id), so re-running
    ingestion overwrites each point in place instead of appending duplicates.
  * COSINE distance over L2-normalised embeddings, so `score` is cosine
    similarity in [-1, 1] - the same number the pgvector build reported.

CLI:
    py -3.14 qdrant_store.py check
    py -3.14 qdrant_store.py init
"""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, MatchAny, FilterSelector,
)

from config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_TIMEOUT, COLLECTION_PREFIX,
    EMBEDDING_DIM,
)


class VectorStoreError(RuntimeError):
    """Any connection/collection problem, reported with a fix."""


# --------------------------------------------------------------------------
# Connection
# --------------------------------------------------------------------------

def get_client():
    """
    Open a client against the Qdrant server.

    Never falls back to embedded/local storage: if the server is unreachable
    the pipeline must stop, not quietly write somewhere else.
    """
    try:
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY,
                              timeout=QDRANT_TIMEOUT)
        client.get_collections()          # force a real round-trip
        return client
    except Exception as e:
        raise VectorStoreError(
            f"Cannot reach Qdrant at {QDRANT_URL}.\n"
            f"  Start it with:\n"
            f"    docker start pt_rag_qdrant\n"
            f"  or create it:\n"
            f"    docker run -d --name pt_rag_qdrant -p 6333:6333 -p 6334:6334 \\\n"
            f"      -v pt_rag_qdrant_data:/qdrant/storage qdrant/qdrant\n"
            f"  Dashboard: {QDRANT_URL}/dashboard\n"
            f"  Client said: {type(e).__name__}: {str(e).strip()[:200]}") from e


def collection_name(book_key):
    return f"{COLLECTION_PREFIX}{book_key}"


def point_id(book_key, chunk_id):
    """
    Deterministic point id for a chunk.

    Random uuid4 ids would make every re-run append a fresh copy of the whole
    book. Deriving the id from the chunk_id makes upserts idempotent.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{book_key}/{chunk_id}"))


# --------------------------------------------------------------------------
# Collection lifecycle
# --------------------------------------------------------------------------

# Payload fields kept on every point - the full citation chain plus provenance.
PAYLOAD_FIELDS = [
    "chunk_id", "book_id", "book", "author", "edition",
    "chapter", "section", "subsection", "section_path",
    "page_start", "page_end", "pdf_page_start", "pdf_page_end",
    "page_label_source", "content_type", "subject_area",
    "text", "n_chars", "n_tokens", "source_file",
    "corpus_version", "chunk_config_id", "extraction_method",
]

# Payload indexes speed up metadata filtering.
INDEXED_FIELDS = ["book_id", "content_type", "subject_area"]


def ensure_collection(client, name, dim=EMBEDDING_DIM, recreate=False):
    """Create the collection (and payload indexes) if needed."""
    exists = client.collection_exists(name)
    if exists and recreate:
        client.delete_collection(name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field in INDEXED_FIELDS:
            try:
                client.create_payload_index(
                    collection_name=name, field_name=field,
                    field_schema="keyword")
            except Exception:
                pass          # index already present - not fatal
    else:
        info = client.get_collection(name)
        actual = info.config.params.vectors.size
        if actual != dim:
            raise VectorStoreError(
                f"Collection '{name}' stores {actual}-d vectors but the model "
                f"produces {dim}-d. Delete the collection and re-ingest.")
    return name


def clear_collection(client, name):
    """Remove every point but keep the collection and its indexes."""
    if not client.collection_exists(name):
        return 0
    before = client.count(collection_name=name, exact=True).count
    if before:
        client.delete(collection_name=name,
                      points_selector=FilterSelector(filter=Filter()))
    return before


def count_points(client, name):
    if not client.collection_exists(name):
        return 0
    return client.count(collection_name=name, exact=True).count


def list_collections(client):
    return sorted(c.name for c in client.get_collections().collections
                  if c.name.startswith(COLLECTION_PREFIX))


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def insert_chunks(client, name, book_key, chunks, vectors, batch_size=256):
    """Upsert chunks with their embeddings. Idempotent via deterministic ids."""
    if len(chunks) != len(vectors):
        raise VectorStoreError(
            f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}")

    points = []
    for c, v in zip(chunks, vectors):
        if len(v) != EMBEDDING_DIM:
            raise VectorStoreError(
                f"Embedding dimension mismatch for {c['chunk_id']}: got "
                f"{len(v)}, collection expects {EMBEDDING_DIM}. Check "
                f"EMBEDDING_MODEL / EMBEDDING_DIM in config.py.")
        points.append(PointStruct(
            id=point_id(book_key, c["chunk_id"]),
            vector=[float(x) for x in v],
            payload={k: c.get(k) for k in PAYLOAD_FIELDS},
        ))

    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=name, points=points[i:i + batch_size],
                      wait=True)
    return len(points)


# --------------------------------------------------------------------------
# Searching
# --------------------------------------------------------------------------

def _build_filter(book_id=None, content_types=None):
    must = []
    if book_id:
        must.append(FieldCondition(key="book_id", match=MatchValue(value=book_id)))
    if content_types:
        must.append(FieldCondition(key="content_type",
                                   match=MatchAny(any=list(content_types))))
    return Filter(must=must) if must else None


def search(client, name, query_vector, limit=5, book_id=None,
           content_types=None):
    """
    Cosine similarity search.

    Returns plain dicts with the same keys the pgvector build returned, so the
    calling code and the saved reports stay comparable across both stores.
    """
    if query_vector is None or len(query_vector) != EMBEDDING_DIM:
        raise VectorStoreError(
            f"Query vector must have {EMBEDDING_DIM} dimensions, got "
            f"{0 if query_vector is None else len(query_vector)}")

    res = client.query_points(
        collection_name=name,
        query=[float(x) for x in query_vector],
        limit=limit,
        query_filter=_build_filter(book_id, content_types),
        with_payload=True,
    )
    out = []
    for p in res.points:
        row = dict(p.payload or {})
        # normalise to the field names the rest of the code expects
        row["book_title"] = row.get("book")
        row["content"] = row.get("text")
        row["score"] = p.score
        out.append(row)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def check():
    print(f"Qdrant URL : {QDRANT_URL}")
    client = get_client()
    cols = list_collections(client)
    print(f"reachable  : yes")
    print(f"collections: {len(cols)}")
    for name in cols:
        info = client.get_collection(name)
        print(f"  {name}: {count_points(client, name)} points, "
              f"dim={info.config.params.vectors.size}, "
              f"distance={info.config.params.vectors.distance}")
    if not cols:
        print("  (none yet - run ingestion)")
    client.close()


def init():
    print(f"Qdrant URL : {QDRANT_URL}")
    client = get_client()
    print("reachable  : yes")
    print(f"dashboard  : {QDRANT_URL}/dashboard")
    print("\nNothing to pre-create: collections are made per book during "
          "ingestion.")
    client.close()


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "check"
    try:
        {"check": check, "init": init}[cmd]()
    except KeyError:
        print("Usage: py -3.14 qdrant_store.py [check|init]")
        sys.exit(2)
    except VectorStoreError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
