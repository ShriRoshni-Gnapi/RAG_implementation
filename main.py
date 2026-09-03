"""
Single entry point for the whole PT-RAG pipeline.

    py -3.14 main.py <book_key>
    py -3.14 main.py paz_acute_care

Runs three stages in order, stopping at the first failure:

    1. ingest_book   PDF -> extract -> clean -> structure -> chunk
    2. embed_store   chunks -> embeddings -> Qdrant collection
    3. retrieve_test retrieval queries -> report

Each stage is the existing script's own function, imported rather than
reimplemented, so the individual CLIs keep working unchanged:

    py -3.14 ingest_book.py <book_key>
    py -3.14 embed_store.py <book_key>
    py -3.14 retrieve_test.py <book_key>

Exit code is 0 on success, 1 on any stage failure, 2 on a bad book key.
"""

import sys
import time
import traceback

from config import BOOKS, REPORTS_DIR

# --------------------------------------------------------------------------
# Console setup. Python 3.14 defaults to UTF-8, but if this ever runs under a
# legacy code page the tick/cross glyphs would raise UnicodeEncodeError, so
# fall back to ASCII markers instead of crashing the pipeline on cosmetics.
# --------------------------------------------------------------------------
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


OK = "✓" if _supports("✓") else "[OK]"      # check mark
BAD = "✗" if _supports("✗") else "[FAIL]"   # ballot X
BAR = "=" * 40


def banner(book_key):
    print(BAR)
    print("PT-RAG Pipeline")
    print(f"Book: {book_key}")
    print(f"Title: {BOOKS[book_key]['title']}")
    print(BAR)


def validate(book_key):
    """Fail fast on an unknown book key, before any expensive work starts."""
    if book_key in BOOKS:
        return
    print(f"{BAD} Unknown book key: {book_key!r}\n")
    print("Registered book keys in config.py:")
    for key, meta in BOOKS.items():
        print(f"  - {key:22} {meta['title']}")
    print("\nAdd a new book by creating an entry in BOOKS in config.py.")
    sys.exit(2)


def run_stage(number, total, label, func, book_key):
    """
    Run one stage. Returns True on success, False on failure.

    Catches BaseException, not just Exception: the ingestion and storage
    safety gates abort with SystemExit, and that must be reported as a
    stage failure rather than silently ending the process.
    """
    print(f"\n[{number}/{total}] {label}...")
    print("-" * 40)
    t0 = time.time()
    try:
        func(book_key)
    except KeyboardInterrupt:
        print(f"\n{BAD} {label} interrupted by user")
        raise
    except SystemExit as e:
        # SystemExit(0)/SystemExit(None) means a clean stop, anything else
        # is one of our abort gates.
        if e.code in (0, None):
            print(f"{OK} {label} completed ({time.time() - t0:.1f}s)")
            return True
        print(f"\n{BAD} {label} failed")
        print(f"Error: {e.code}")
        return False
    except Exception as e:
        print(f"\n{BAD} {label} failed")
        print(f"Error: {type(e).__name__}: {e}")
        print("\n--- traceback ---")
        traceback.print_exc()
        return False
    print(f"{OK} {label} completed ({time.time() - t0:.1f}s)")
    return True


def preflight():
    """
    Check the Qdrant server before doing any expensive work.

    Ingestion takes minutes; discovering the vector store is down afterwards
    wastes all of it. This fails in seconds instead.
    """
    print("\n[0/3] Checking Qdrant...")
    print("-" * 40)
    try:
        import qdrant_store
        from config import QDRANT_URL
        print(f"target: {QDRANT_URL}")
        client = qdrant_store.get_client()
        try:
            cols = qdrant_store.list_collections(client)
            print(f"{OK} Qdrant reachable ({len(cols)} existing collection(s))")
        finally:
            client.close()
        return True
    except Exception as e:
        print(f"\n{BAD} Vector store preflight failed")
        print(f"Error: {e}")
        return False


def main(book_key):
    validate(book_key)
    banner(book_key)

    if not preflight():
        print(f"\n{BAR}")
        print("Pipeline FAILED at preflight: Qdrant is not ready")
        print(f"{BAR}")
        return 1

    # Imported lazily inside each stage so heavy dependencies
    # (torch / sentence-transformers) load only when that stage runs, and any
    # import error is attributed to the stage that needed it.
    def stage_ingest(key):
        from ingest_book import ingest_book
        ingest_book(key)

    def stage_embed(key):
        from embed_store import embed_store
        embed_store(key)

    def stage_retrieve(key):
        from retrieve_test import retrieve_test
        retrieve_test(key)

    stages = [
        ("Ingesting book", stage_ingest),
        ("Embedding and storing", stage_embed),
        ("Running retrieval tests", stage_retrieve),
    ]

    t0 = time.time()
    for i, (label, func) in enumerate(stages, 1):
        if not run_stage(i, len(stages), label, func, book_key):
            # Stop here. Never run a later stage on the output of a failed one.
            print(f"\n{BAR}")
            print(f"Pipeline FAILED at stage {i}/{len(stages)}: {label}")
            print(f"{BAR}")
            return 1

    print(f"\n{BAR}")
    print("Pipeline completed successfully")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(f"{BAR}")
    print("\nReports written to:")
    for suffix in ("_ingest_report.json", "_sample_chunks.txt",
                   "_embed_report.json", "_retrieval_results.json"):
        p = REPORTS_DIR / f"{book_key}{suffix}"
        mark = OK if p.exists() else BAD
        size = f"{p.stat().st_size:,} bytes" if p.exists() else "MISSING"
        print(f"  {mark} {p.name:42} {size}")
    return 0


def list_books():
    """Show everything discovered in the books folder."""
    from config import BOOKS_DIR, unsupported_files
    print(f"Books folder: {BOOKS_DIR}\n")
    print(f"{'book_key':<32} {'tier':<5} {'subject':<20} title")
    print("-" * 100)
    for key, m in BOOKS.items():
        mark = "" if m.get("curated") else "  (auto)"
        print(f"{key:<32} {m['structure_tier']:<5} {m['subject_area']:<20} "
              f"{m['title'][:38]}{mark}")
    print(f"\n{len(BOOKS)} ingestible book(s).")
    un = unsupported_files()
    if un:
        print(f"\n{len(un)} file(s) present but NOT ingestible "
              f"(PDF only - no DOC/DOCX extractor yet):")
        for f in un:
            print(f"  - {f}")


def run_all(keys=None):
    """
    Ingest every discovered book.

    One book failing does not stop the rest: each is reported and the run
    continues, with a summary at the end and a non-zero exit if any failed.
    """
    keys = keys or list(BOOKS)
    if not keys:
        print(f"{BAD} No books found. Put PDFs in the books folder.")
        return 2

    print(BAR)
    print("PT-RAG Pipeline - ALL BOOKS")
    print(f"Books: {len(keys)}")
    print(BAR)

    if not preflight():
        print(f"\n{BAR}\nFAILED at preflight: Qdrant is not ready\n{BAR}")
        return 1

    t0 = time.time()
    results = []
    for i, key in enumerate(keys, 1):
        print(f"\n\n{'#' * 100}")
        print(f"# BOOK {i}/{len(keys)}: {key} - {BOOKS[key]['title'][:60]}")
        print(f"{'#' * 100}")
        bt = time.time()
        ok = True
        for label, func in (("Ingesting book", _stage_ingest),
                            ("Embedding and storing", _stage_embed)):
            if not run_stage(1 if label.startswith("Ing") else 2, 2,
                             label, func, key):
                ok = False
                break
        results.append((key, ok, time.time() - bt))
        print(f"\n{OK if ok else BAD} {key}: "
              f"{'done' if ok else 'FAILED'} in {time.time() - bt:.1f}s")

    ok_n = sum(1 for _, ok, _ in results if ok)
    print(f"\n\n{BAR}")
    print(f"ALL-BOOKS SUMMARY  ({ok_n}/{len(results)} succeeded)")
    print(f"Total time: {time.time() - t0:.1f}s")
    print(BAR)
    for key, ok, secs in results:
        print(f"  {OK if ok else BAD} {key:<32} {secs:>7.1f}s")
    failed = [k for k, ok, _ in results if not ok]
    if failed:
        print(f"\n{len(failed)} book(s) failed: {', '.join(failed)}")
        return 1
    print("\nQuery across every book with:")
    print("  py -3.14 retrieve_test.py \"your question\" --all-books")
    return 0


def verify():
    """
    Cross-check every book: chunks on disk vs points actually in Qdrant.

    Worth having because a collection can end up empty *after* a successful
    ingestion (a later re-create, a manual command, a wiped volume) and nothing
    in the pipeline would notice. This is the check that catches that drift.
    """
    import json
    import qdrant_store
    from config import CHUNKS_DIR, EXCLUDED_CONTENT_TYPES

    client = qdrant_store.get_client()
    try:
        print(f"{'book_key':<32} {'expected':>9} {'stored':>8}  status")
        print("-" * 70)
        problems, total = [], 0
        for key in BOOKS:
            path = CHUNKS_DIR / f"{key}_chunks.json"
            if not path.exists():
                print(f"{key:<32} {'-':>9} {'-':>8}  not ingested")
                continue
            chunks = json.load(open(path, encoding="utf-8"))
            expected = sum(1 for c in chunks
                           if c["content_type"] not in EXCLUDED_CONTENT_TYPES)
            name = qdrant_store.collection_name(key)
            stored = qdrant_store.count_points(client, name)
            total += stored
            if stored == expected and expected > 0:
                state = f"{OK} ok"
            elif stored == 0:
                state = f"{BAD} EMPTY - re-run embed_store"
                problems.append(key)
            else:
                state = f"{BAD} MISMATCH"
                problems.append(key)
            print(f"{key:<32} {expected:>9,} {stored:>8,}  {state}")
        print("-" * 70)
        print(f"{'TOTAL':<32} {'':>9} {total:>8,}")
        if problems:
            print(f"\n{len(problems)} book(s) need attention: {', '.join(problems)}")
            print("Fix with:  py -3.14 embed_store.py <book_key>")
            return 1
        print(f"\n{OK} All {len(BOOKS)} books consistent.")
        return 0
    finally:
        client.close()


def _stage_ingest(key):
    from ingest_book import ingest_book
    ingest_book(key)


def _stage_embed(key):
    from embed_store import embed_store
    embed_store(key)


def _usage():
    print("Usage:")
    print("  py -3.14 main.py <book_key>     run one book")
    print("  py -3.14 main.py --all          run every book in the books folder")
    print("  py -3.14 main.py --list         show discovered books\n")
    print("Registered book keys:")
    for key, meta in BOOKS.items():
        print(f"  - {key:32} {meta['title'][:44]}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        _usage()
        sys.exit(2)
    arg = sys.argv[1]
    if arg in ("--list", "-l"):
        list_books()
        sys.exit(0)
    if arg in ("--all", "-a"):
        sys.exit(run_all())
    if arg in ("--verify", "-v"):
        sys.exit(verify())
    if arg in ("-h", "--help"):
        _usage()
        sys.exit(0)
    sys.exit(main(arg))
