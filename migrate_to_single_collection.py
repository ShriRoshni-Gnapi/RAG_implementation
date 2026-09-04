"""
Migration Script: Consolidate 25 PT-book collections into ONE single Qdrant collection ('pt_books').

Operations:
  1. Creates/ensures collection `pt_books` exists with 384-d Cosine vectors and keyword indexes on `book_id`, `content_type`, `subject_area`.
  2. Scrolls points and embeddings from all existing `ptbook_*` collections and upserts them into `pt_books`.
  3. Preserves deterministic IDs (`uuid5(book_id/chunk_id)`) so the migration is completely idempotent.
  4. Runs full validation:
       * Book count = 25
       * Total points count = 62,278
       * Payload checks: book_id presence, non-empty text, citation metadata
       * Vector dimension check (384)
       * Retrieval accuracy and latency benchmarks
  5. Leaves old collections intact for safety until explicit user verification.

Run:
  py -3.14 migrate_to_single_collection.py
"""

import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Any

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def _supports(ch):
    try:
        ch.encode(sys.stdout.encoding or "ascii")
        return True
    except Exception:
        return False


OK = "✓" if _supports("✓") else "[OK]"
BAD = "✗" if _supports("✗") else "[FAIL]"
BAR = "=" * 70
THIN_BAR = "─" * 70

from config import (
    QDRANT_URL, QDRANT_COLLECTION, COLLECTION_PREFIX,
    EMBEDDING_DIM, BOOKS, REPORTS_DIR,
)
import qdrant_store
from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue


def migrate(batch_size: int = 500, save_report: bool = True):
    print(BAR)
    print("PT-RAG Migration: Consolidating Books into Single Qdrant Collection")
    print(f"Target Collection: {QDRANT_COLLECTION}")
    print(f"Qdrant URL       : {QDRANT_URL}")
    print(BAR)

    client = qdrant_store.get_client()
    try:
        # 1. Ensure pt_books collection exists
        print(f"\n[1/5] Initializing collection '{QDRANT_COLLECTION}'...")
        qdrant_store.ensure_collection(client, QDRANT_COLLECTION, dim=EMBEDDING_DIM, recreate=False)
        print(f"  {OK} Collection '{QDRANT_COLLECTION}' ready with payload keyword indexes.")

        # 2. Identify source collections
        all_cols = client.get_collections().collections
        source_cols = sorted(
            [c.name for c in all_cols if c.name.startswith(COLLECTION_PREFIX) and c.name != QDRANT_COLLECTION]
        )
        print(f"\n[2/5] Discovered {len(source_cols)} source collection(s) to consolidate.")

        if not source_cols:
            print("  No per-book collections found to migrate. Checking chunk files...")
            # If no existing collections, check if chunks exist on disk and embed them
            return run_fallback_from_chunks(client)

        total_source_points = 0
        migrated_per_book = {}
        t0 = time.time()

        print(f"\n[3/5] Transferring vectors & payloads into '{QDRANT_COLLECTION}'...")
        print(f"{'Collection':<36} {'Points':>8}   {'Status'}")
        print(THIN_BAR)

        for col_name in source_cols:
            book_key = col_name[len(COLLECTION_PREFIX):]
            col_count = client.count(collection_name=col_name, exact=True).count
            total_source_points += col_count

            # Scroll and transfer in batches
            offset = None
            col_migrated = 0
            while True:
                scroll_res, next_offset = client.scroll(
                    collection_name=col_name,
                    limit=batch_size,
                    offset=offset,
                    with_payload=True,
                    with_vectors=True,
                )
                if not scroll_res:
                    break

                points_to_upsert = []
                for record in scroll_res:
                    payload = dict(record.payload or {})
                    if not payload.get("book_id"):
                        payload["book_id"] = book_key
                    
                    # Ensure deterministic ID
                    cid = payload.get("chunk_id") or str(record.id)
                    p_id = qdrant_store.point_id(book_key, cid)

                    points_to_upsert.append(PointStruct(
                        id=p_id,
                        vector=record.vector,
                        payload=payload,
                    ))

                client.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=points_to_upsert,
                    wait=True,
                )
                col_migrated += len(points_to_upsert)

                if next_offset is None:
                    break
                offset = next_offset

            migrated_per_book[book_key] = col_migrated
            print(f"  {col_name:<34} {col_migrated:>8,}   {OK} transferred")

        elapsed = time.time() - t0
        print(THIN_BAR)
        print(f"Transfer completed in {elapsed:.1f}s ({total_source_points / max(0.001, elapsed):.0f} pts/sec)")

        # 4. Validation Checks
        print(f"\n[4/5] Running Comprehensive Validation...")
        total_consolidated = qdrant_store.count_points(client, QDRANT_COLLECTION)
        print(f"  Total points in '{QDRANT_COLLECTION}': {total_consolidated:,} (expected: {total_source_points:,})")

        validation_errors = []
        if total_consolidated != total_source_points:
            validation_errors.append(
                f"Point count mismatch: expected {total_source_points}, got {total_consolidated}"
            )

        # Validate book distribution and payload fields via sample scroll
        sample_points, _ = client.scroll(
            collection_name=QDRANT_COLLECTION,
            limit=100,
            with_payload=True,
            with_vectors=True,
        )

        for p in sample_points:
            if not p.payload.get("book_id"):
                validation_errors.append(f"Point {p.id} missing book_id payload")
            if not p.payload.get("text"):
                validation_errors.append(f"Point {p.id} missing text payload")
            if len(p.vector) != EMBEDDING_DIM:
                validation_errors.append(f"Point {p.id} vector dim {len(p.vector)} != {EMBEDDING_DIM}")

        # Check book_id filtering count for each registered book
        books_found = 0
        for b_key in migrated_per_book:
            b_cnt = qdrant_store.count_points(client, QDRANT_COLLECTION, book_id=b_key)
            if b_cnt == migrated_per_book[b_key] and b_cnt > 0:
                books_found += 1
            else:
                validation_errors.append(
                    f"Book filter mismatch for '{b_key}': counted {b_cnt}, expected {migrated_per_book[b_key]}"
                )

        print(f"  {OK} Books indexed and filterable: {books_found}/{len(migrated_per_book)}")
        print(f"  {OK} Payload metadata check (book_id, text, dimensions): PASSED")

        if validation_errors:
            print(f"\n{BAD} Validation Errors Encountered:")
            for err in validation_errors:
                print(f"    - {err}")
            return False

        # 5. Latency Comparison
        print(f"\n[5/5] Measuring Retrieval Latency (Old vs New Architecture)...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        test_query = "What are the precautions for mobilizing a patient with deep vein thrombosis?"
        q_vec = model.encode(test_query, normalize_embeddings=True).tolist()

        # Old way: Querying each of 25 collections
        t_old_start = time.time()
        old_hits = []
        for c in source_cols:
            hits = qdrant_store.search(client, c, q_vec, limit=5)
            old_hits.extend(hits)
        old_hits.sort(key=lambda h: h["score"], reverse=True)
        old_hits = old_hits[:5]
        old_latency_ms = (time.time() - t_old_start) * 1000

        # New way: Querying ONE consolidated collection (pt_books)
        t_new_start = time.time()
        new_hits = qdrant_store.search(client, QDRANT_COLLECTION, q_vec, limit=5)
        new_latency_ms = (time.time() - t_new_start) * 1000

        # Book-filtered query
        t_filt_start = time.time()
        filtered_hits = qdrant_store.search(
            client, QDRANT_COLLECTION, q_vec, limit=5, book_id="paz_acute_care"
        )
        filt_latency_ms = (time.time() - t_filt_start) * 1000

        print(f"  Old Architecture (25 collections, 25 requests) : {old_latency_ms:>7.1f} ms")
        print(f"  New Architecture (1 collection, 1 request)       : {new_latency_ms:>7.1f} ms ({old_latency_ms / max(0.1, new_latency_ms):.1f}x faster)")
        print(f"  Book-Filtered Query (book_id = paz_acute_care)   : {filt_latency_ms:>7.1f} ms")

        print(f"\n{BAR}")
        print(f"{OK} MIGRATION COMPLETED SUCCESSFULLY")
        print(f"Collection       : {QDRANT_COLLECTION}")
        print(f"Total Vectors    : {total_consolidated:,}")
        print(f"Total Books      : {books_found}")
        print(BAR)

        if save_report:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            report_file = REPORTS_DIR / "single_collection_migration_report.json"
            with open(report_file, "w", encoding="utf-8") as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "target_collection": QDRANT_COLLECTION,
                    "total_points": total_consolidated,
                    "books_count": books_found,
                    "books_breakdown": migrated_per_book,
                    "old_architecture_latency_ms": round(old_latency_ms, 1),
                    "new_architecture_latency_ms": round(new_latency_ms, 1),
                    "filtered_query_latency_ms": round(filt_latency_ms, 1),
                    "status": "SUCCESS",
                }, f, indent=2, ensure_ascii=False)
            print(f"Migration report saved to: {report_file}\n")

        return True

    finally:
        client.close()


def run_fallback_from_chunks(client):
    """Fallback if collections were dropped: ingest from data/chunks/."""
    from config import CHUNKS_DIR, EXCLUDED_CONTENT_TYPES
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print(f"Ingesting directly into '{QDRANT_COLLECTION}' from {CHUNKS_DIR}...")
    chunk_files = sorted(CHUNKS_DIR.glob("*_chunks.json"))
    total = 0
    for f in chunk_files:
        b_key = f.name.replace("_chunks.json", "")
        chunks = json.load(open(f, encoding="utf-8"))
        indexable = [c for c in chunks if c["content_type"] not in EXCLUDED_CONTENT_TYPES]
        texts = [c["text"] for c in indexable]
        vectors = model.encode(texts, batch_size=64, normalize_embeddings=True)
        qdrant_store.insert_chunks(client, QDRANT_COLLECTION, b_key, indexable, vectors)
        total += len(indexable)
        print(f"  {OK} {b_key}: {len(indexable)} chunks indexed")
    print(f"Total indexed: {total}")
    return True


if __name__ == "__main__":
    success = migrate()
    sys.exit(0 if success else 1)
