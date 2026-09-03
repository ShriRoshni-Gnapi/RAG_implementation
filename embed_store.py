"""
Embed chunks and store them in Qdrant (server at http://localhost:6333).

Only the storage layer changed: chunking, the embedding model and the metadata
schema are exactly as before. All vector-store access goes through
qdrant_store.py.

Run:  py -3.14 embed_store.py paz_acute_care
"""

import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

import qdrant_store
from qdrant_store import VectorStoreError
from config import (
    CHUNKS_DIR, EMBEDDING_MODEL, EMBEDDING_DIM, EMBED_BATCH_SIZE,
    EXCLUDED_CONTENT_TYPES, TOKEN_SAFETY_LIMIT, REPORTS_DIR,
    QDRANT_URL, DISTANCE_METRIC,
)


_CACHED_MODEL = None


def get_embedding_model():
    """Load the SentenceTransformer model once and reuse across multiple book runs in the same process."""
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        print(f"\nLoading model  : {EMBEDDING_MODEL}")
        _CACHED_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _CACHED_MODEL


def main(book_key):
    path = CHUNKS_DIR / f"{book_key}_chunks.json"
    if not path.exists():
        raise SystemExit(
            f"Chunk file not found: {path}\n"
            f"  Run ingestion first:  py -3.14 ingest_book.py {book_key}")

    chunks = json.load(open(path, encoding="utf-8"))
    if not chunks:
        raise SystemExit(f"ABORT: {path} contains no chunks (empty document).")

    indexable = [c for c in chunks if c["content_type"] not in EXCLUDED_CONTENT_TYPES]
    skipped = len(chunks) - len(indexable)
    print(f"chunks in file : {len(chunks)}")
    print(f"excluded       : {skipped} ({', '.join(EXCLUDED_CONTENT_TYPES)})")
    print(f"to index       : {len(indexable)}")
    if not indexable:
        raise SystemExit("ABORT: every chunk was excluded - nothing to index.")

    # Never embed something the model would silently truncate.
    over = [c["chunk_id"] for c in indexable if c["n_tokens"] > TOKEN_SAFETY_LIMIT]
    if over:
        raise SystemExit(f"ABORT: {len(over)} chunks over {TOKEN_SAFETY_LIMIT} tokens")

    model = get_embedding_model()
    dim = model.get_sentence_embedding_dimension()
    print(f"dimension      : {dim}   max_seq_length: {model.max_seq_length}")
    if dim != EMBEDDING_DIM:
        raise SystemExit(
            f"ABORT: model produces {dim}-d vectors but EMBEDDING_DIM is "
            f"{EMBEDDING_DIM}. Update config.py and re-create the collection.")

    name = qdrant_store.collection_name(book_key)
    print(f"\nQdrant         : {QDRANT_URL}")
    print(f"Collection     : {name} ({DISTANCE_METRIC})")

    client = qdrant_store.get_client()
    try:
        # Rebuild from scratch so a re-run can never leave stale points behind.
        qdrant_store.ensure_collection(client, name, dim, recreate=True)
        print("collection ready (0 points)")

        texts = [c["text"] for c in indexable]
        print(f"\nEmbedding {len(texts)} chunks (batch={EMBED_BATCH_SIZE}, CPU) ...")
        t0 = time.time()
        vectors = model.encode(
            texts, batch_size=EMBED_BATCH_SIZE,
            show_progress_bar=True, convert_to_numpy=True,
            normalize_embeddings=True,
        )
        elapsed = time.time() - t0
        print(f"embedded in {elapsed:.1f}s  ({len(texts)/elapsed:.1f} chunks/s)")
        print(f"vector shape   : {vectors.shape}")

        print("\nUpserting into Qdrant ...")
        t1 = time.time()
        sent = qdrant_store.insert_chunks(client, name, book_key,
                                          indexable, vectors)
        insert_s = time.time() - t1
        stored = qdrant_store.count_points(client, name)
        print(f"upserted points: {sent} in {insert_s:.1f}s")
        print(f"points in coll : {stored}")

        if stored != len(indexable):
            raise SystemExit(
                f"ABORT: stored {stored} points but expected {len(indexable)}. "
                f"The collection may contain stale or duplicated data.")

        report = {
            "book_id": book_key,
            "vector_store": "qdrant",
            "qdrant_url": QDRANT_URL,
            "collection": name,
            "distance": DISTANCE_METRIC,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": dim,
            "max_seq_length": model.max_seq_length,
            "chunks_in_file": len(chunks),
            "chunks_excluded": skipped,
            "chunks_indexed": len(indexable),
            "points_stored": stored,
            "embed_seconds": round(elapsed, 1),
            "chunks_per_second": round(len(texts) / elapsed, 1),
            "upsert_seconds": round(insert_s, 1),
        }
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with open(REPORTS_DIR / f"{book_key}_embed_report.json", "w",
                  encoding="utf-8") as f:
            json.dump(report, f, indent=1)
    finally:
        client.close()

    print(f"\nDone. {stored} vectors in collection '{name}'.")


# Public name for pipeline orchestration (see main.py). Same function -
# main() is kept so the existing CLI behaviour is unchanged.
embed_store = main


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: py -3.14 embed_store.py <book_key>")
        print("To see available book keys, run: py -3.14 main.py --list")
        sys.exit(2 if len(sys.argv) < 2 else 0)
    try:
        main(sys.argv[1])
    except VectorStoreError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

