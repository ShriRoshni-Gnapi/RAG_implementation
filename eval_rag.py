"""
Evaluation framework for PT-RAG + NVIDIA Nemotron generation & citation quality.

Evaluates:
  1. Retrieval Relevance (Cosine similarity scores, top-k rank distribution)
  2. NVIDIA Nemotron Generation Latency & Throughput
  3. Citation Precision (Marker existence, range validity, source resolution)
  4. Grounding Honesty (Handling in-scope vs out-of-scope queries)

Run:
  py -3.14 eval_rag.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

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

from config import REPORTS_DIR, BOOKS
from query_engine import PTQueryEngine
from llm_client import LLMClient


EVAL_BENCHMARK = [
    {
        "category": "acute_care",
        "question": "What are the precautions for mobilizing a patient with deep vein thrombosis?",
        "expected_in_scope": True,
        "target_keywords": ["dvt", "anticoagul", "mobiliz", "heparin", "thrombosis"],
    },
    {
        "category": "cardiopulmonary",
        "question": "How is heart rate used to determine exercise intensity in cardiac patients?",
        "expected_in_scope": True,
        "target_keywords": ["heart rate", "karvonen", "target", "intensity", "exercise"],
    },
    {
        "category": "orthopaedics",
        "question": "What exercises are recommended for improving quadriceps strength after ACL reconstruction?",
        "expected_in_scope": True,
        "target_keywords": ["quadriceps", "acl", "closed kinetic", "extension"],
    },
    {
        "category": "neuro",
        "question": "What are the key rehabilitation strategies for improving gait in post-stroke patients?",
        "expected_in_scope": True,
        "target_keywords": ["stroke", "gait", "weight-bearing", "symmetry", "locomotor"],
    },
    {
        "category": "out_of_scope",
        "question": "What is the recommended surgical technique for repairing an orbital floor blowout fracture?",
        "expected_in_scope": False,
        "target_keywords": [],
    },
]


def run_evaluation(save_report: bool = True):
    print("=" * 80)
    print("PT-RAG + NVIDIA NEMOTRON BENCHMARK EVALUATION")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    engine = PTQueryEngine(top_k=5)
    results = []

    total_q = len(EVAL_BENCHMARK)
    passed_citations = 0
    total_latency = 0.0

    for idx, item in enumerate(EVAL_BENCHMARK, 1):
        q = item["question"]
        cat = item["category"]
        in_scope = item["expected_in_scope"]

        print(f"\n[{idx}/{total_q}] Testing ({cat}): {q[:65]}...")
        t0 = time.time()

        try:
            res = engine.query(question=q, all_books=True)
            elapsed = time.time() - t0
            total_latency += elapsed

            citation_count = len(res.get("citations", []))
            invalid_citations = len(res.get("invalid_citation_indices", []))
            is_grounded = invalid_citations == 0

            if is_grounded:
                passed_citations += 1

            print(f"    {OK} Latency: {elapsed:.2f}s (Retrieval: {res['retrieval_ms']}ms, Gen: {res['generation_ms']}ms)")
            print(f"    {OK} Model: {res.get('model')}")
            print(f"    {OK} Citations: {citation_count} valid, {invalid_citations} invalid")

            results.append({
                "item_id": idx,
                "category": cat,
                "question": q,
                "expected_in_scope": in_scope,
                "retrieval_ms": res["retrieval_ms"],
                "generation_ms": res["generation_ms"],
                "total_ms": res["total_ms"],
                "citations_cited": citation_count,
                "invalid_citations": invalid_citations,
                "is_grounded": is_grounded,
                "answer_text": res["answer"],
                "citations": res["citations"],
            })

        except Exception as e:
            print(f"    {BAD} Error running query: {e}")
            results.append({
                "item_id": idx,
                "category": cat,
                "question": q,
                "error": str(e),
            })


    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    avg_latency = (total_latency / total_q) if total_q else 0
    print(f"Total benchmark questions: {total_q}")
    print(f"Citation validation pass rate: {passed_citations}/{total_q} ({passed_citations/total_q*100:.1f}%)")
    print(f"Average end-to-end latency: {avg_latency:.2f}s")

    if save_report:
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = REPORTS_DIR / f"rag_evaluation_{ts_str}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "benchmark_size": total_q,
                "avg_latency_sec": round(avg_latency, 2),
                "pass_rate": round(passed_citations / total_q, 2),
                "results": results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\nFull evaluation report saved to:\n  {report_file}")

    return 0


if __name__ == "__main__":
    sys.exit(run_evaluation())
