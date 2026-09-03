"""
Retrieval test against Qdrant (server at http://localhost:6333).

Validates: Book -> Extraction -> Chunking -> Embeddings -> Qdrant -> Retrieval.
MedGemma is not involved.

Run the test suite for a specific book:
    py -3.14 retrieve_test.py <book_key>

Ask a question against a specific book:
    py -3.14 retrieve_test.py <book_key> "What causes pressure ulcers?"

Ask a question across ALL ingested books:
    py -3.14 retrieve_test.py "What causes pressure ulcers?" --all-books

Options:
    --top-k N     number of results (default 5)
    --all-books   search every ingested book
"""

import json
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")

from sentence_transformers import SentenceTransformer

import qdrant_store
from qdrant_store import VectorStoreError
from config import EMBEDDING_MODEL, REPORTS_DIR, BOOKS, QDRANT_URL

_CACHED_MODEL = None


def get_embedding_model():
    global _CACHED_MODEL
    if _CACHED_MODEL is None:
        _CACHED_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _CACHED_MODEL

QUERIES = [
    "What are the precautions for mobilizing a patient with deep vein thrombosis?",
    "How is heart rate used to determine exercise intensity in cardiac patients?",
    "What are the physiological effects of prolonged bed rest?",
    "How do you interpret arterial blood gas values?",
    "What should a physical therapist consider when treating a patient with a burn injury?",
    "What are the physical therapy considerations after coronary artery bypass graft surgery?",
    # deliberately outside this book's scope - checks score separation
    "What is the Lachman test for anterior cruciate ligament instability?",
]

TOP_K = 5


def preview(text, n=220):
    t = " ".join((text or "").split())
    return t[:n] + ("..." if len(t) > n else "")


def show(hit, rank):
    print(f"Rank {rank}")
    print(f"  Chunk   : {hit['chunk_id']}")
    print(f"  Score   : {hit['score']:.4f}")
    print(f"  Book    : {hit['book_title']}")
    print(f"  Chapter : {hit['chapter']}")
    print(f"  Section : {hit['section']}")
    if hit.get("subsection"):
        print(f"  Subsect : {hit['subsection']}")
    print(f"  Pages   : {hit['page_start']}-{hit['page_end']} "
          f"(pdf {hit['pdf_page_start']}, src={hit['page_label_source']})")
    print(f"  Content : {preview(hit['content'])}")
    print()


def run_queries(queries, book_key=None, top_k=TOP_K, all_books=False, save_to=None):
    client = qdrant_store.get_client()
    try:
        if all_books:
            names = qdrant_store.list_collections(client)
            if not names:
                raise SystemExit(
                    "No collections found.\n"
                    "  Run ingestion first:  py -3.14 main.py <book_key>")
        else:
            if not book_key:
                raise SystemExit(
                    "No book specified.\n"
                    "  Specify a book:  py -3.14 retrieve_test.py <book_key>\n"
                    "  Or search all:   py -3.14 retrieve_test.py --all-books")
            names = [qdrant_store.collection_name(book_key)]
            if not client.collection_exists(names[0]):
                raise SystemExit(
                    f"Collection '{names[0]}' does not exist.\n"
                    f"  Run ingestion first:  py -3.14 main.py {book_key}")

        total = sum(qdrant_store.count_points(client, n) for n in names)
        if total == 0:
            target = "collections" if all_books else f"collection '{names[0]}'"
            raise SystemExit(
                f"{target} contain no points.\n"
                f"  Run ingestion first:  py -3.14 main.py {book_key or '<book_key>'}")

        print("=" * 100)
        print(f"RETRIEVAL TEST  |  qdrant @ {QDRANT_URL}")
        print(f"collections={', '.join(names)}  points={total}  "
              f"model={EMBEDDING_MODEL}  top_k={top_k}")
        print("=" * 100)

        model = get_embedding_model()
        results_out = []
        for qi, q in enumerate(queries, 1):
            t0 = time.time()
            qv = model.encode(q, normalize_embeddings=True).tolist()
            hits = []
            for n in names:
                hits.extend(qdrant_store.search(client, n, qv, limit=top_k))
            hits.sort(key=lambda h: h["score"], reverse=True)
            hits = hits[:top_k]
            ms = (time.time() - t0) * 1000

            print(f"\n[{qi}] Question:")
            print(f"    {q}")
            print(f"    ({ms:.0f} ms)")
            print("-" * 100)
            print("Top Results:\n")
            if not hits:
                print("  (no results)\n")
            for rank, h in enumerate(hits, 1):
                show(h, rank)

            results_out.append({
                "query": q,
                "top_score": round(hits[0]["score"], 4) if hits else None,
                "latency_ms": round(ms),
                "hits": [{
                    "rank": r,
                    "score": round(h["score"], 4),
                    "chunk_id": h["chunk_id"],
                    "book": h["book_title"],
                    "chapter": h["chapter"],
                    "section": h["section"],
                    "subsection": h["subsection"],
                    "page_start": h["page_start"],
                    "page_end": h["page_end"],
                    "preview": preview(h["content"], 300),
                } for r, h in enumerate(hits, 1)],
            })

        if len(results_out) > 1:
            print("\n" + "=" * 100)
            print("TOP-SCORE SUMMARY")
            print("=" * 100)
            for r in results_out:
                print(f"  {r['top_score']:.4f}  {r['latency_ms']:>4} ms  "
                      f"{r['query'][:72]}")

        if save_to:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(save_to, "w", encoding="utf-8") as f:
                json.dump({
                    "vector_store": "qdrant",
                    "qdrant_url": QDRANT_URL,
                    "collections": names,
                    "points": total,
                    "embedding_model": EMBEDDING_MODEL,
                    "top_k": top_k,
                    "results": results_out,
                }, f, indent=1, ensure_ascii=False)
            print(f"\nSaved -> {save_to}")
        return results_out
    finally:
        client.close()


def main(book_key=None, question=None, top_k=TOP_K, all_books=False):
    if not all_books and not book_key:
        raise SystemExit(
            "ABORT: No book specified.\n"
            "  Specify a book:  py -3.14 retrieve_test.py <book_key>\n"
            "  Or search all:   py -3.14 retrieve_test.py \"<question>\" --all-books\n"
            "  To list books:   py -3.14 main.py --list"
        )

    if question is not None:
        q = question.strip()
        if not q:
            raise SystemExit("ABORT: empty query. Provide a question to search for.")
        run_queries([q], book_key, top_k, all_books, save_to=None)
        return

    save_path = (
        REPORTS_DIR / "all_books_retrieval_results.json"
        if all_books
        else REPORTS_DIR / f"{book_key}_retrieval_results.json"
    )
    run_queries(QUERIES, book_key, top_k, all_books, save_to=save_path)


# Public name for pipeline orchestration (see main.py). Same function -
# main() is kept so the existing CLI behaviour is unchanged.
retrieve_test = main


def _parse(argv):
    """
    Accepts:
        retrieve_test.py <book_key>
        retrieve_test.py <book_key> "a question"
        retrieve_test.py "a question" --all-books
    plus --top-k N.
    """
    book, question, top_k, all_books = None, None, TOP_K, False
    args, i = [], 0
    while i < len(argv):
        a = argv[i]
        if a in ("--top-k", "-k"):
            i += 1
            if i >= len(argv):
                raise SystemExit("--top-k needs a number")
            top_k = int(argv[i])
        elif a == "--all-books":
            all_books = True
        elif a in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            args.append(a)
        i += 1

    if not args and not all_books:
        print(__doc__)
        print("Registered book keys (run with: py -3.14 retrieve_test.py <book_key>):")
        for k in BOOKS:
            print(f"  - {k}")
        sys.exit(0)

    if len(args) == 1:
        if args[0] in BOOKS:
            book = args[0]
        else:
            question = args[0]
            if not all_books:
                raise SystemExit(
                    f"Ambiguous query '{question}': no book specified.\n"
                    f"  Specify a book:  py -3.14 retrieve_test.py <book_key> \"{question}\"\n"
                    f"  Or search all:   py -3.14 retrieve_test.py \"{question}\" --all-books\n"
                    f"  To list books:   py -3.14 main.py --list"
                )
    elif len(args) >= 2:
        book = args[0]
        question = " ".join(args[1:])

    return book, question, top_k, all_books


if __name__ == "__main__":
    try:
        b, q, k, allb = _parse(sys.argv[1:])
        main(b, q, k, allb)
    except VectorStoreError as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

