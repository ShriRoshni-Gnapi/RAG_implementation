"""
Qdrant storage layer (consolidated single-collection mode: pt_books, http://localhost:6333).

This is the ONLY module that talks to the vector database. Ingestion, migration,
and retrieval all go through it.

Design notes:
  * Stores all books in ONE consolidated collection: `pt_books` (configurable via QDRANT_COLLECTION).
  * Fast single-query vector retrieval across all books (or book-filtered via payload filter `book_id`).
  * Point IDs are deterministic (uuid5 of book_id/chunk_id), so re-running ingestion/upsert
    overwrites each point in place idempotently without duplicating points.
  * COSINE distance over L2-normalised 384-d embeddings.
  * Keyword payload indexes on `book_id`, `content_type`, and `subject_area`.

CLI:
    py -3.14 qdrant_store.py check
    py -3.14 qdrant_store.py init
"""

import uuid
from typing import Optional, List, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, FieldCondition,
    MatchValue, MatchAny, FilterSelector,
)

from config import (
    QDRANT_URL, QDRANT_API_KEY, QDRANT_TIMEOUT, QDRANT_COLLECTION,
    COLLECTION_PREFIX, EMBEDDING_DIM,
)


class VectorStoreError(RuntimeError):
    """Any connection/collection problem, reported with a fix."""


# What Qdrant's client raises when a request fails after the connection was made:
# UnexpectedResponse and ResponseHandlingException (timeouts, resets, HTTP errors)
# both derive from ApiException; a raw socket failure is an OSError. A stubbed or
# older client without those classes falls back to OSError alone.
try:
    from qdrant_client.http.exceptions import ApiException
    QDRANT_ERRORS = (ApiException, OSError)
except ImportError:
    QDRANT_ERRORS = (OSError,)


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


def collection_name(book_key: Optional[str] = None) -> str:
    """Return the active collection name (pt_books)."""
    return QDRANT_COLLECTION


def point_id(book_key: str, chunk_id: str) -> str:
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
    "chunk_id", "book_id", "book", "author", "edition", "year",
    "chapter", "section", "subsection", "section_path",
    "page_start", "page_end", "pdf_page_start", "pdf_page_end",
    "page_label_source", "content_type", "subject_area",
    "text", "n_chars", "n_tokens", "source_file",
    "corpus_version", "chunk_config_id", "extraction_method",
]

# Payload indexes speed up metadata filtering.
INDEXED_FIELDS = ["book_id", "content_type", "subject_area"]


def ensure_collection(client: QdrantClient, name: Optional[str] = None,
                      dim: int = EMBEDDING_DIM, recreate: bool = False) -> str:
    """Create the collection (and payload indexes) if needed."""
    target_name = name or QDRANT_COLLECTION
    exists = client.collection_exists(target_name)
    if exists and recreate:
        client.delete_collection(target_name)
        exists = False
    if not exists:
        client.create_collection(
            collection_name=target_name,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        for field in INDEXED_FIELDS:
            try:
                client.create_payload_index(
                    collection_name=target_name, field_name=field,
                    field_schema="keyword")
            except Exception:
                pass          # index already present - not fatal
    else:
        info = client.get_collection(target_name)
        actual = info.config.params.vectors.size
        if actual != dim:
            raise VectorStoreError(
                f"Collection '{target_name}' stores {actual}-d vectors but the model "
                f"produces {dim}-d. Delete the collection and re-ingest.")
        # Ensure payload indexes exist even on existing collection
        for field in INDEXED_FIELDS:
            try:
                client.create_payload_index(
                    collection_name=target_name, field_name=field,
                    field_schema="keyword")
            except Exception:
                pass
    return target_name


def clear_collection(client: QdrantClient, name: Optional[str] = None) -> int:
    """Remove every point but keep the collection and its indexes."""
    target_name = name or QDRANT_COLLECTION
    if not client.collection_exists(target_name):
        return 0
    before = client.count(collection_name=target_name, exact=True).count
    if before:
        client.delete(collection_name=target_name,
                      points_selector=FilterSelector(filter=Filter()))
    return before


def count_points(client: QdrantClient, name: Optional[str] = None,
                 book_id: Optional[str] = None) -> int:
    """
    Count points in collection. If book_id is provided, counts points
    matching that specific book using the payload filter.
    """
    target_name = name or QDRANT_COLLECTION
    if not client.collection_exists(target_name):
        return 0
    if book_id:
        filter_query = Filter(
            must=[FieldCondition(key="book_id", match=MatchValue(value=book_id))]
        )
        return client.count(collection_name=target_name, count_filter=filter_query,
                            exact=True).count
    return client.count(collection_name=target_name, exact=True).count


def list_collections(client: QdrantClient) -> List[str]:
    """List collections in Qdrant."""
    return sorted(c.name for c in client.get_collections().collections)


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------

def insert_chunks(client: QdrantClient, name: Optional[str], book_key: str,
                  chunks: List[Dict[str, Any]], vectors: Any, batch_size: int = 256) -> int:
    """Upsert chunks with their embeddings. Idempotent via deterministic ids."""
    target_name = name or QDRANT_COLLECTION
    if len(chunks) != len(vectors):
        raise VectorStoreError(
            f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}")

    ensure_collection(client, target_name, dim=EMBEDDING_DIM, recreate=False)

    points = []
    for c, v in zip(chunks, vectors):
        if len(v) != EMBEDDING_DIM:
            raise VectorStoreError(
                f"Embedding dimension mismatch for {c.get('chunk_id')}: got "
                f"{len(v)}, collection expects {EMBEDDING_DIM}. Check "
                f"EMBEDDING_MODEL / EMBEDDING_DIM in config.py.")
        payload = {k: c.get(k) for k in PAYLOAD_FIELDS}
        if not payload.get("book_id"):
            payload["book_id"] = book_key
        cid = c.get("chunk_id", str(len(points)))
        points.append(PointStruct(
            id=point_id(book_key, cid),
            vector=[float(x) for x in v],
            payload=payload,
        ))

    for i in range(0, len(points), batch_size):
        client.upsert(collection_name=target_name, points=points[i:i + batch_size],
                      wait=True)
    return len(points)


# --------------------------------------------------------------------------
# Searching
# --------------------------------------------------------------------------

def _build_filter(book_id: Optional[str] = None,
                  content_types: Optional[List[str]] = None) -> Optional[Filter]:
    must = []
    if book_id:
        must.append(FieldCondition(key="book_id", match=MatchValue(value=book_id)))
    if content_types:
        must.append(FieldCondition(key="content_type",
                                   match=MatchAny(any=list(content_types))))
    return Filter(must=must) if must else None


def search(client: QdrantClient, name: Optional[str], query_vector: List[float],
           limit: int = 5, book_id: Optional[str] = None,
           content_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """
    Cosine similarity search against a single collection (pt_books).

    If book_id is supplied, applies a fast payload keyword filter.
    Returns plain dicts with normalized keys (book_title, content, score).
    """
    target_name = name or QDRANT_COLLECTION
    if query_vector is None or len(query_vector) != EMBEDDING_DIM:
        raise VectorStoreError(
            f"Query vector must have {EMBEDDING_DIM} dimensions, got "
            f"{0 if query_vector is None else len(query_vector)}")

    # A request that fails after the connection was made (timeout, reset, HTTP error) is a
    # VectorStoreError too, so a caller treats it like any other "Qdrant is unavailable"
    # and not as a bug. Only the exception's type is kept in the message.
    try:
        exists = client.collection_exists(target_name)
        if exists:
            res = client.query_points(
                collection_name=target_name,
                query=[float(x) for x in query_vector],
                limit=limit,
                query_filter=_build_filter(book_id, content_types),
                with_payload=True,
            )
    except QDRANT_ERRORS as e:
        raise VectorStoreError(f"Qdrant request failed ({type(e).__name__}).") from e

    if not exists:
        raise VectorStoreError(
            f"Collection '{target_name}' does not exist in Qdrant.\n"
            f"  Run migration/ingestion: py -3.14 migrate_to_single_collection.py"
        )
    out = []
    for p in res.points:
        row = dict(p.payload or {})
        # normalise to the field names the rest of the code expects
        row["book_title"] = row.get("book") or row.get("book_title")
        row["content"] = row.get("text") or row.get("content")
        row["score"] = p.score
        out.append(row)
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def check():
    print(f"Qdrant URL  : {QDRANT_URL}")
    print(f"Collection  : {QDRANT_COLLECTION}")
    client = get_client()
    try:
        exists = client.collection_exists(QDRANT_COLLECTION)
        print(f"reachable   : yes")
        print(f"{QDRANT_COLLECTION} exists: {'yes' if exists else 'NO (run migration)'}")
        if exists:
            total = count_points(client, QDRANT_COLLECTION)
            info = client.get_collection(QDRANT_COLLECTION)
            print(f"total points: {total:,}")
            print(f"vector dim  : {info.config.params.vectors.size}")
            print(f"distance    : {info.config.params.vectors.distance}")
        all_cols = list_collections(client)
        print(f"all collections in Qdrant: {len(all_cols)}")
        for c in all_cols:
            n = count_points(client, c)
            print(f"  - {c:36}: {n:>7,} points")
    finally:
        client.close()


def init():
    print(f"Qdrant URL : {QDRANT_URL}")
    client = get_client()
    try:
        ensure_collection(client, QDRANT_COLLECTION, dim=EMBEDDING_DIM, recreate=False)
        print(f"✓ Collection '{QDRANT_COLLECTION}' is ready with payload indexes.")
        print(f"Dashboard: {QDRANT_URL}/dashboard")
    finally:
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
