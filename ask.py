"""
Interactive Physiotherapy Reference Assistant & LLM QA CLI (NVIDIA Nemotron / MedGemma).

Usage:
  1. Ask a question across all books:
     py -3.14 ask.py "What are the precautions for mobilizing a patient with DVT?"

  2. Ask a question against a specific book:
     py -3.14 ask.py --book paz_acute_care "What is the normal range for arterial blood gases?"

  3. Start an interactive terminal assistant session:
     py -3.14 ask.py --interactive

  4. Test LLM and Qdrant connectivity:
     py -3.14 ask.py --check-api

  5. JSON output mode for programmatic integrations:
     py -3.14 ask.py "..." --json
"""

import argparse
import json
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import (
    LLM_API_URL,
    LLM_API_TOKEN,
    LLM_MODEL,
    QDRANT_URL,
    BOOKS,
)
from llm_client import LLMClient, LLMError
from query_engine import PTQueryEngine
import qdrant_store

BAR = "═" * 70
THIN_BAR = "─" * 70


def check_api_connectivity():
    """Run connection diagnostics for Qdrant and the LLM server."""
    print(BAR)
    print("PT-RAG API & Environment Health Check")
    print(BAR)

    # 1. Check Qdrant
    print(f"\n[1/2] Checking Vector Store (Qdrant)...")
    print(f"  URL:        {QDRANT_URL}")
    print(f"  Collection: {qdrant_store.QDRANT_COLLECTION}")
    try:
        client = qdrant_store.get_client()
        total_points = qdrant_store.count_points(client, qdrant_store.QDRANT_COLLECTION)
        print(f"  ✓ Connected: {total_points:,} total vectors indexed in '{qdrant_store.QDRANT_COLLECTION}'.")
        client.close()
    except Exception as e:
        print(f"  ✗ Qdrant Connection Failed: {e}")
        print("  Hint: Ensure Qdrant container is running: docker start pt_rag_qdrant")


    # 2. Check LLM server
    print(f"\n[2/2] Checking Inference Server (NVIDIA Nemotron / LLM)...")
    print(f"  Endpoint: {LLM_API_URL}")
    print(f"  Model:    {LLM_MODEL}")
    try:
        llm = LLMClient()
        status = llm.check_connection()
        print(f"  ✓ Connected: Server responsive ({status['latency_ms']:.0f} ms latency)")
        print(f"  Model reported: {status['model']}")
    except Exception as e:
        print(f"  ✗ LLM Server Connection Failed: {e}")
        print("  Hint: Check NVIDIA_API_URL and server availability in .env.")

    print(f"\n{BAR}\n")


def display_result(res: dict, json_mode: bool = False):
    """Format and print the assistant answer with clean sources and timing."""
    if json_mode:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return

    print("\n" + BAR)
    print(f"QUESTION: {res['question']}")
    print(BAR + "\n")

    print(res["answer"])
    print()

    # Sources Block
    if res.get("all_retrieved"):
        print("REFERENCED TEXTBOOKS & SOURCES")
        print(THIN_BAR)
        seen = set()
        for entry in res["all_retrieved"]:
            formatted = entry.get("formatted_text", "")
            if formatted and formatted not in seen:
                seen.add(formatted)
                print(formatted)
                print()
        print(THIN_BAR)
        print("Clinical decision-making requires individual patient assessment.")

    print(f"\n{THIN_BAR}")
    print(
        f"Model: {res['model']} | "
        f"Retrieval: {res['retrieval_ms']}ms | "
        f"Generation: {res['generation_ms']}ms | "
        f"Total: {res['total_ms']}ms"
    )
    print(BAR + "\n")


def interactive_mode(book_key: str = None, top_k: int = 5):
    """Start an interactive question answering terminal session."""
    engine = PTQueryEngine(top_k=top_k)
    scope = f"Book: {BOOKS[book_key]['title']}" if book_key and book_key in BOOKS else "All Ingested Textbooks"

    print(BAR)
    print("PT-RAG Interactive Clinical Assistant")
    print(f"Scope: {scope}")
    print(f"Model: {LLM_MODEL}")
    print("Type your question and press Enter. Type 'exit' or 'quit' to end.")
    print(BAR + "\n")

    while True:
        try:
            q = input("\n[Question] > ").strip()
            if not q:
                continue
            if q.lower() in ("exit", "quit", "q"):
                print("Goodbye!")
                break

            print("\nSearching textbooks & generating clinical response...")
            res = engine.query(
                question=q,
                book_key=book_key,
                all_books=(book_key is None),
                top_k=top_k,
            )
            display_result(res)

        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break
        except Exception as e:
            print(f"\n[Error] {type(e).__name__}: {e}\n")


def run_reference_mode(path: str, top_k: int, json_mode: bool, summaries: bool = True,
                       summary_only: bool = False) -> int:
    """Read a SOAP note and print its matching textbook references (with detailed summaries)."""
    note = sys.stdin.read() if path == "-" else open(path, encoding="utf-8").read()
    if not note.strip():
        print("Error: the SOAP note is empty.")
        return 2
    res = PTQueryEngine(top_k=top_k).find_references(
        note, top_k=top_k, summaries=summaries, summary_only=summary_only)
    print(json.dumps(res, indent=2, ensure_ascii=False) if json_mode else res["references"])
    return 0


def main():
    parser =argparse.ArgumentParser(description="Physiotherapy RAG + NVIDIA Nemotron Query Interface")
    parser.add_argument("question", nargs="?", help="Clinical question to ask")
    parser.add_argument("--book", "-b", help="Specific book key to search (e.g. paz_acute_care)")
    parser.add_argument("--all-books", "-a", action="store_true", default=True, help="Search across all ingested books (default)")
    parser.add_argument("--top-k", "-k", type=int, default=5, help="Number of chunks to retrieve (default: 5)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Start interactive Q&A assistant loop")
    parser.add_argument("--check-api", "-c", action="store_true", help="Test LLM and Qdrant connectivity")
    parser.add_argument("--json", "-j", action="store_true", help="Output response in JSON format")
    parser.add_argument("--soap-file", "-s", metavar="PATH",
                        help="SOAP note text file ('-' reads stdin); prints the matching textbook references")
    parser.add_argument("--summary-only", action="store_true",
                        help="With --soap-file: print only each citation line and its Detailed Summary "
                             "(no original sentences)")
    parser.add_argument("--no-summary", action="store_true",
                        help="With --soap-file: print only the citations and original sentences, without Detailed Summary")

    args = parser.parse_args()

    if args.soap_file:
        if args.summary_only and args.no_summary:
            parser.error("--summary-only and --no-summary cannot be used together")
        return run_reference_mode(args.soap_file, args.top_k, args.json,
                                  summaries=not args.no_summary, summary_only=args.summary_only)

    if args.check_api:
        check_api_connectivity()
        return 0

    if args.interactive:
        interactive_mode(book_key=args.book, top_k=args.top_k)
        return 0

    if not args.question:
        if len(sys.argv) == 1:
            interactive_mode(book_key=args.book, top_k=args.top_k)
            return 0
        parser.print_help()
        return 2

    engine = PTQueryEngine(top_k=args.top_k)
    res = engine.query(
        question=args.question,
        book_key=args.book,
        all_books=(args.book is None) or args.all_books,
        top_k=args.top_k,
    )
    display_result(res, json_mode=args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
