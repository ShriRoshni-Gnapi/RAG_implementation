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
import re
import time
from typing import Dict, List, Optional, Any, Callable

import qdrant_store
from config import (
    EMBEDDING_MODEL,
    BOOKS,
    get_llm_config,
    get_qdrant_config,
)
from llm_client import LLMClient, get_llm_client
from prompts import (
    SYSTEM_PROMPT,
    REFERENCE_SYSTEM_PROMPT,
    NO_REFERENCE_TEXT,
    assemble_context,
    build_user_prompt,
    build_reference_user_prompt,
)
from citations import (
    extract_source_labels,
    format_reference_list,
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


# The embedding model reads only ~256 tokens; 500 characters of clinical prose
# stays under that (config.py measured up to 0.42 tokens/char).
NOTE_SEGMENT_CHARS = 500


def split_note(text: str, max_chars: int = NOTE_SEGMENT_CHARS) -> List[str]:
    """
    Cut a long note into pieces the embedding model reads in full.

    Searching with the whole note would silently use only its first ~256
    tokens, so the rest of the note would never influence retrieval.
    """
    pieces, cur = [], ""
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", text):
        sentence = sentence.strip()
        while len(sentence) > max_chars:          # one very long sentence
            if cur:
                pieces.append(cur)
                cur = ""
            pieces.append(sentence[:max_chars])
            sentence = sentence[max_chars:].strip()
        if not sentence:
            continue
        if cur and len(cur) + len(sentence) + 1 > max_chars:
            pieces.append(cur)
            cur = sentence
        else:
            cur = f"{cur} {sentence}".strip()
    if cur:
        pieces.append(cur)
    return pieces


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


    def retrieve_for_note(
        self,
        note: str,
        top_k: Optional[int] = None,
        max_sources: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Search every segment of the note, merge by chunk (best score wins) and
        keep the strongest passages. Unlike query(), a passage below min_score
        is never kept as a fallback: no strong match means no reference.
        """
        best: Dict[str, Dict[str, Any]] = {}
        for piece in split_note(note):
            for hit in self.retrieve_chunks(piece, all_books=True, top_k=top_k):
                key = hit.get("chunk_id") or hit.get("content") or ""
                if key not in best or hit["score"] > best[key]["score"]:
                    best[key] = hit
        ranked = sorted(best.values(), key=lambda h: h["score"], reverse=True)
        return [h for h in ranked if h["score"] >= self.min_score][:max_sources]

    def find_references(
        self,
        soap_note: str,
        top_k: Optional[int] = None,
        max_sources: int = 12,
    ) -> Dict[str, Any]:
        """
        SOAP note -> combined list of relevant textbook references.

        The model only chooses which retrieved passages are relevant. The
        printed list is built from stored metadata and passage text, so a
        book title, author, chapter or page can never be invented.
        """
        t0 = time.time()
        hits = self.retrieve_for_note(soap_note, top_k=top_k, max_sources=max_sources)
        retrieval_ms = (time.time() - t0) * 1000

        raw_answer, model = "", self.client.model
        if hits:
            # max_chars sized so no retrieved passage is dropped from the prompt
            context_text = assemble_context(hits, max_chars=len(hits) * 3000)
            gen = self.client.generate(
                system_prompt=REFERENCE_SYSTEM_PROMPT,
                user_prompt=build_reference_user_prompt(soap_note, context_text),
                temperature=0.0,
            )
            raw_answer = gen.get("text", "")
            model = gen.get("model", model)

        indices = extract_source_labels(raw_answer, len(hits))
        return {
            "references": format_reference_list(hits, indices) if hits else NO_REFERENCE_TEXT,
            "source_indices": indices,
            "retrieved_count": len(hits),
            "raw_answer": raw_answer,
            "model": model,
            "retrieval_ms": round(retrieval_ms, 1),
            "total_ms": round((time.time() - t0) * 1000, 1),
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
