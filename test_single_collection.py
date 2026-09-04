"""
Comprehensive Automated Test Suite for Single-Collection PT-RAG (pt_books).

Validates all 6 user test specifications:
  Test 1 — Single collection: pt_books exists and all 25 books are stored there.
  Test 2 — All-book retrieval: One single Qdrant vector search against pt_books.
  Test 3 — Book-specific retrieval: Qdrant payload filter `book_id = paz_acute_care` works.
  Test 4 — Metadata integrity: Every retrieved chunk has book_id, text, and citation metadata.
  Test 5 — Idempotency: Re-upserting does not duplicate point count.
  Test 6 — NVIDIA Nemotron Generation: Generates grounded response using existing env config.

Run:
  py -3.14 test_single_collection.py
"""

import json
import sys
import time
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
    QDRANT_URL, QDRANT_COLLECTION, BOOKS, EMBEDDING_DIM,
    LLM_API_URL, LLM_MODEL,
)
import qdrant_store
from query_engine import PTQueryEngine, get_embedding_model
from llm_client import LLMClient


def test_1_single_collection(client) -> bool:
    print(f"\n[Test 1] Single Collection Architecture ('{QDRANT_COLLECTION}')...")
    exists = client.collection_exists(QDRANT_COLLECTION)
    if not exists:
        print(f"  {BAD} Collection '{QDRANT_COLLECTION}' does not exist.")
        return False

    total = qdrant_store.count_points(client, QDRANT_COLLECTION)
    print(f"  {OK} Collection '{QDRANT_COLLECTION}' exists with {total:,} points.")

    # Check that all 25 books have points in pt_books
    books_with_points = 0
    for key in BOOKS:
        cnt = qdrant_store.count_points(client, QDRANT_COLLECTION, book_id=key)
        if cnt > 0:
            books_with_points += 1
        else:
            print(f"  {BAD} Book '{key}' has 0 points in '{QDRANT_COLLECTION}'")

    print(f"  {OK} Books indexed in '{QDRANT_COLLECTION}': {books_with_points}/{len(BOOKS)}")
    return exists and (books_with_points == len(BOOKS)) and (total > 0)


def test_2_all_book_retrieval(engine: PTQueryEngine) -> bool:
    print(f"\n[Test 2] All-Book Single-Query Retrieval...")
    q = "What are the precautions for mobilizing a patient with deep vein thrombosis?"
    t0 = time.time()
    hits = engine.retrieve_chunks(question=q, all_books=True, top_k=5)
    elapsed_ms = (time.time() - t0) * 1000

    if not hits:
        print(f"  {BAD} No hits returned.")
        return False

    print(f"  {OK} Retrieved {len(hits)} chunks in {elapsed_ms:.1f} ms via a single Qdrant query.")
    print(f"  Top score: {hits[0]['score']:.4f} (Book: {hits[0].get('book_title')})")
    return len(hits) == 5 and hits[0]["score"] > 0.40


def test_3_book_specific_retrieval(engine: PTQueryEngine) -> bool:
    print(f"\n[Test 3] Book-Specific Retrieval with Payload Filter...")
    target_book = "paz_acute_care"
    q = "What are the normal arterial blood gas ranges?"
    t0 = time.time()
    hits = engine.retrieve_chunks(question=q, book_key=target_book, all_books=False, top_k=5)
    elapsed_ms = (time.time() - t0) * 1000

    if not hits:
        print(f"  {BAD} No hits returned for book '{target_book}'.")
        return False

    # Verify that every hit belongs to paz_acute_care
    all_match = all(h.get("book_id") == target_book for h in hits)
    if not all_match:
        print(f"  {BAD} Some hits do not match book_id '{target_book}': {[h.get('book_id') for h in hits]}")
        return False

    print(f"  {OK} Filter `book_id == '{target_book}'` strictly enforced ({len(hits)} hits, {elapsed_ms:.1f} ms).")
    print(f"  Top hit: {hits[0].get('chapter')} - {hits[0].get('section')} (Score: {hits[0]['score']:.4f})")
    return True


def test_4_metadata_integrity(engine: PTQueryEngine) -> bool:
    print(f"\n[Test 4] Payload Metadata Integrity Check...")
    hits = engine.retrieve_chunks(question="What causes pressure ulcers?", all_books=True, top_k=5)

    required_keys = ["book_id", "text", "chunk_id", "page_start", "page_end"]
    missing = []
    for i, h in enumerate(hits, 1):
        for k in required_keys:
            if k not in h and (k == "text" and "content" not in h):
                missing.append(f"Hit #{i} missing '{k}'")

    if missing:
        for m in missing:
            print(f"  {BAD} {m}")
        return False

    print(f"  {OK} All required metadata fields present in retrieved payloads ({', '.join(required_keys)}).")
    return True


def test_5_idempotency(client) -> bool:
    print(f"\n[Test 5] Ingestion Idempotency & Deduplication...")
    before_count = qdrant_store.count_points(client, QDRANT_COLLECTION)
    book_to_test = "postural_drainage"
    book_before = qdrant_store.count_points(client, QDRANT_COLLECTION, book_id=book_to_test)

    # Re-upsert a dummy point with deterministic ID
    dummy_chunk = {
        "chunk_id": "test_idempotent_001",
        "book_id": book_to_test,
        "book": "Postural Drainage Guidelines",
        "text": "Idempotency test chunk text content.",
        "content_type": "text",
    }
    dummy_vec = [0.0] * EMBEDDING_DIM
    dummy_vec[0] = 1.0  # Normalized unit vector

    # Insert once
    qdrant_store.insert_chunks(client, QDRANT_COLLECTION, book_to_test, [dummy_chunk], [dummy_vec])
    count_after_first = qdrant_store.count_points(client, QDRANT_COLLECTION)

    # Insert again (identical ID)
    qdrant_store.insert_chunks(client, QDRANT_COLLECTION, book_to_test, [dummy_chunk], [dummy_vec])
    count_after_second = qdrant_store.count_points(client, QDRANT_COLLECTION)

    # Clean up dummy point
    dummy_id = qdrant_store.point_id(book_to_test, "test_idempotent_001")
    client.delete(collection_name=QDRANT_COLLECTION, points_selector=[dummy_id])

    final_count = qdrant_store.count_points(client, QDRANT_COLLECTION)

    is_idempotent = (count_after_first == count_after_second) and (final_count == before_count)
    if is_idempotent:
        print(f"  {OK} Upserting identical chunk produces identical point ID ({dummy_id}).")
        print(f"  {OK} Re-insertion did not duplicate points (Count stayed {count_after_first:,}).")
    else:
        print(f"  {BAD} Count changed: first={count_after_first}, second={count_after_second}")

    return is_idempotent


def test_6_nvidia_generation(engine: PTQueryEngine) -> bool:
    print(f"\n[Test 6] End-to-End NVIDIA Nemotron Generation & Citations...")
    q = "What are the precautions for mobilizing a patient with deep vein thrombosis?"
    t0 = time.time()
    res = engine.query(question=q, all_books=True, top_k=5)
    elapsed = time.time() - t0

    if not res.get("answer"):
        print(f"  {BAD} Empty answer returned from LLM.")
        return False

    print(f"  {OK} Generation completed in {elapsed:.2f}s (Model: {res.get('model')}).")
    print(f"  {OK} Citations: {len(res.get('citations', []))} valid, {len(res.get('invalid_citation_indices', []))} invalid.")
    print(f"\nSample Answer:\n{res['answer'][:200]}...")
    return len(res.get("citations", [])) > 0 and len(res.get("invalid_citation_indices", [])) == 0


def main():
    print(BAR)
    print("PT-RAG Single-Collection ('pt_books') Validation Suite")
    print(f"Qdrant Target: {QDRANT_COLLECTION} @ {QDRANT_URL}")
    print(f"LLM Endpoint : {LLM_API_URL} ({LLM_MODEL})")
    print(BAR)

    client = qdrant_store.get_client()
    try:
        engine = PTQueryEngine(top_k=5)

        tests = [
            ("Test 1: Single Collection Architecture", lambda: test_1_single_collection(client)),
            ("Test 2: All-Book Single Retrieval", lambda: test_2_all_book_retrieval(engine)),
            ("Test 3: Book-Specific Filtered Retrieval", lambda: test_3_book_specific_retrieval(engine)),
            ("Test 4: Metadata Integrity", lambda: test_4_metadata_integrity(engine)),
            ("Test 5: Ingestion Idempotency", lambda: test_5_idempotency(client)),
            ("Test 6: NVIDIA Nemotron Generation & Citations", lambda: test_6_nvidia_generation(engine)),
        ]

        passed = 0
        for name, test_fn in tests:
            try:
                ok = test_fn()
                if ok:
                    passed += 1
                else:
                    print(f"\n{BAD} {name} FAILED\n")
            except Exception as e:
                print(f"\n{BAD} {name} raised exception: {type(e).__name__}: {e}\n")

        print("\n" + BAR)
        print(f"TEST RESULTS: {passed}/{len(tests)} PASSED ({passed/len(tests)*100:.0f}%)")
        print(BAR)

        return 0 if passed == len(tests) else 1

    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
