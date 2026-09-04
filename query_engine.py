"""
Core RAG Query Engine connecting Retrieval, Context Assembly, NVIDIA Nemotron / LLM, and Citations.

Flow:
  User Question
       ↓
  SentenceTransformers (all-MiniLM-L6-v2)
       ↓
  Qdrant Vector Search (Single book or all 25 collections)
       ↓
  Score Filtering & Context Assembly ([1], [2], ...)
       ↓
  NVIDIA Nemotron / LLM Inference (via OpenAI-compatible vLLM API)
       ↓
  Citation Validation & Provenance Rendering
       ↓
  Grounded Clinical Answer + Verifiable Source Citations
"""

import os
import time
from typing import Dict, List, Optional, Any, Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import qdrant_store
from config import (
    EMBEDDING_MODEL,
    BOOKS,
    get_llm_config,
    get_qdrant_config,
)
from llm_client import LLMClient, get_llm_client
from prompts import SYSTEM_PROMPT, assemble_context, build_user_prompt
from citations import (
    validate_citations,
    render_citation_entry,
    format_citations_block,
    DISCLAIMER_TEXT,
)

_CACHED_EMBED_MODEL = None


def get_embedding_model():
    """Lazy load the sentence transformer model."""
    global _CACHED_EMBED_MODEL
    if _CACHED_EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer
        _CACHED_EMBED_MODEL = SentenceTransformer(EMBEDDING_MODEL)
    return _CACHED_EMBED_MODEL


class PTQueryEngine:
    """
    Orchestrates vector retrieval from Qdrant and synthesis through LLM.
    """

    def __init__(
        self,
        top_k: int = 5,
        min_score: float = 0.20,
        client: Optional[LLMClient] = None,
    ):
        self.top_k = top_k
        self.min_score = min_score
        self.client = client or get_llm_client()

    def retrieve_chunks(
        self,
        question: str,
        book_key: Optional[str] = None,
        all_books: bool = True,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Embed the question and search Qdrant pt_books in a single vector query
        (with optional book_id payload filter).
        """
        limit = top_k or self.top_k
        model = get_embedding_model()
        q_vec = model.encode(question, normalize_embeddings=True).tolist()

        target_book = None if (all_books and book_key is None) else book_key

        client = qdrant_store.get_client()
        try:
            hits = qdrant_store.search(
                client=client,
                name=None,  # defaults to QDRANT_COLLECTION (pt_books)
                query_vector=q_vec,
                limit=limit,
                book_id=target_book,
            )
            return hits
        finally:
            client.close()


    def query(
        self,
        question: str,
        book_key: Optional[str] = None,
        all_books: bool = True,
        top_k: Optional[int] = None,
        intent: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute full RAG flow: retrieval -> prompt building -> LLM generation -> citation verification.
        """
        t0 = time.time()

        # 1. Retrieval
        hits = self.retrieve_chunks(
            question=question,
            book_key=book_key,
            all_books=all_books,
            top_k=top_k,
        )
        retrieval_ms = (time.time() - t0) * 1000

        # Filter below min_score if we have hits
        filtered_hits = [h for h in hits if h.get("score", 0) >= self.min_score]
        if not filtered_hits and hits:
            # If all are below threshold, keep at least the top hit
            filtered_hits = hits[:1]

        # 2. Assemble context
        context_text = assemble_context(filtered_hits)
        user_prompt = build_user_prompt(
            question=question,
            context_text=context_text,
            intent=intent,
        )

        # 3. LLM Inference
        t_gen_start = time.time()
        gen_result = self.client.generate(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=user_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        generation_ms = (time.time() - t_gen_start) * 1000
        raw_answer = gen_result.get("text", "")

        # 4. Citation extraction & validation
        cleaned_answer, valid_idx, invalid_idx, audit = validate_citations(
            raw_answer, filtered_hits
        )

        # 5. Citation rendering
        cited_entries = []
        for idx in valid_idx:
            if 1 <= idx <= len(filtered_hits):
                cited_entries.append(render_citation_entry(filtered_hits[idx - 1], idx))

        all_entries = [
            render_citation_entry(h, i) for i, h in enumerate(filtered_hits, 1)
        ]

        total_ms = (time.time() - t0) * 1000

        # Build formatted presentation text
        sources_text = format_citations_block(
            filtered_hits,
            valid_idx if valid_idx else list(range(1, len(filtered_hits) + 1)),
        )

        formatted_output = f"""{cleaned_answer}

{sources_text}"""

        return {
            "question": question,
            "answer": cleaned_answer,
            "raw_answer": raw_answer,
            "citations": cited_entries,
            "all_retrieved": all_entries,
            "valid_citation_indices": valid_idx,
            "invalid_citation_indices": invalid_idx,
            "audit": audit,
            "model": gen_result.get("model", self.client.model),
            "retrieval_ms": round(retrieval_ms, 1),
            "generation_ms": round(generation_ms, 1),
            "total_ms": round(total_ms, 1),
            "formatted_output": formatted_output,
        }


def ask_question(
    question: str,
    book_key: Optional[str] = None,
    all_books: bool = True,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Convenience one-call helper."""
    engine = PTQueryEngine(top_k=top_k)
    return engine.query(question=question, book_key=book_key, all_books=all_books)
