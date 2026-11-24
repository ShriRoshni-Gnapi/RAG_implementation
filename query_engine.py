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
from llm_client import LLMClient, LLMError, get_llm_client
from summaries import CALL_FAILED_PREFIX, parse_summary, summary_failure
from prompts import (
    MAX_SENTENCES_PER_SOURCE,
    SUMMARY_SYSTEM_PROMPT,
    build_summary_user_prompt,
    SYSTEM_PROMPT,
    REFERENCE_SYSTEM_PROMPT,
    NO_REFERENCE_TEXT,
    assemble_context,
    build_user_prompt,
    build_reference_user_prompt,
    assemble_numbered_context,
)
from passages import hit_text, sentences_for, split_text
from citations import (
    build_reference_entries,
    extract_source_sentences,
    selection_reply_usable,
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


# Room for two to four short summary paragraphs, plus any thinking text the
# server's model emits before them.
SUMMARY_MAX_TOKENS = 1500

# Stored with every generated set so a caller can tell which prompts and filters
# produced it. Change it whenever the prompts, sentence filters or summary checks change.
GENERATOR_VERSION = "pt-rag-references-1"

# "S:", "O:", "A:", "P:" or "Subjective:" ... at the start of a line.
_SECTION_MARK = re.compile(
    r"(?m)^[ \t]*(?:(?i:(subjective|objective|assessment|plan))[ \t]*[:\-–]|([SOAP])[ \t]*:)[ \t]*"
)


def parse_soap_sections(note: str) -> List[tuple]:
    """
    [(label, text)] for a SOAP note, label in S/O/A/P, or '' for text outside
    any marker. A note with no markers is one '' section, so it still works.
    """
    marks = list(_SECTION_MARK.finditer(note))
    if not marks:
        return [("", note.strip())] if note.strip() else []
    sections = []
    lead = note[:marks[0].start()].strip()
    if lead:
        sections.append(("", lead))
    for i, m in enumerate(marks):
        label = (m.group(1) or m.group(2))[0].upper()
        end = marks[i + 1].start() if i + 1 < len(marks) else len(note)
        body = note[m.end():end].strip()
        if body:
            sections.append((label, body))
    return sections


def _wrap(unit: str, max_chars: int) -> List[str]:
    """Cut one over-long unit at word boundaries, never inside a word."""
    out, cur = [], ""
    for word in unit.split():
        if cur and len(cur) + len(word) + 1 > max_chars:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return out + ([cur] if cur else [])


def split_note(text: str, max_chars: int = NOTE_SEGMENT_CHARS) -> List[str]:
    """
    Cut a section into search queries the embedding model reads in full.

    Searching with the whole text would silently use only its first ~256
    tokens. Pieces end at sentence or clause boundaries (never inside a
    word), and each piece repeats the last unit of the one before it, so a
    finding is not separated from the context next to it.
    """
    units = []
    for line in text.splitlines():
        for sentence in split_text(line):
            units.extend(u.strip() for u in re.split(r";\s+", sentence) if u.strip())

    pieces: List[str] = []
    cur: List[str] = []

    def size(parts: List[str]) -> int:
        return sum(len(p) for p in parts) + max(0, len(parts) - 1)

    for unit in (w for u in units for w in _wrap(u, max_chars)):
        if cur and size(cur + [unit]) > max_chars:
            pieces.append(" ".join(cur))
            cur = [cur[-1]] if size([cur[-1], unit]) <= max_chars else []
        cur.append(unit)
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def build_queries(sections: List[tuple], max_chars: int = NOTE_SEGMENT_CHARS):
    """
    (pieces, weights) for every section. Each section's pieces share a total
    weight of 1, so a long Objective section cannot drown out a short
    Subjective or Assessment section.
    """
    pieces: List[str] = []
    weights: List[float] = []
    for _, body in sections:
        section_pieces = split_note(body, max_chars)
        pieces.extend(section_pieces)
        weights.extend([1.0 / len(section_pieces)] * len(section_pieces))
    return pieces, weights


def fuse_ranked_hits(
    result_lists: List[List[Dict[str, Any]]],
    weights: List[float],
    rrf_k: int = 60,
) -> List[Dict[str, Any]]:
    """
    Reciprocal-rank fusion of the per-query results.

    A passage that several parts of the note retrieve outranks one that a
    single phrase retrieved, so passages that fit the note as a whole come
    first. Each hit keeps its best cosine score in 'score' and gets 'fused'.
    Ties break on score then chunk id so the order is repeatable.
    """
    fused: Dict[str, float] = {}
    best: Dict[str, Dict[str, Any]] = {}
    for hits, weight in zip(result_lists, weights):
        for rank, hit in enumerate(sorted(hits, key=lambda h: h["score"], reverse=True), 1):
            key = hit.get("chunk_id") or hit_text(hit)
            fused[key] = fused.get(key, 0.0) + weight / (rrf_k + rank)
            if key not in best or hit["score"] > best[key]["score"]:
                best[key] = hit
    ranked = sorted(best, key=lambda k: (-fused[k], -best[k]["score"], k))
    return [{**best[k], "fused": round(fused[k], 6)} for k in ranked]


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


    def _search_pieces(self, pieces: List[str], top_k: int) -> List[List[Dict[str, Any]]]:
        """One Qdrant search per query piece (one connection, one batch encode)."""
        vectors = get_embedding_model().encode(pieces, normalize_embeddings=True).tolist()
        client = qdrant_store.get_client()
        try:
            return [
                qdrant_store.search(client=client, name=None, query_vector=v, limit=top_k)
                for v in vectors
            ]
        finally:
            client.close()

    def retrieve_for_note(
        self,
        sections: List[tuple],
        top_k: Optional[int] = None,
        max_sources: int = 12,
    ) -> List[Dict[str, Any]]:
        """
        Search every piece of every section, fuse the rankings and keep the
        strongest passages. A passage under min_score (the engine's existing
        floor, not re-tuned for this mode) is never kept as a fallback.
        """
        pieces, weights = build_queries(sections)
        if not pieces:
            return []
        results = self._search_pieces(pieces, top_k or self.top_k)
        ranked = fuse_ranked_hits(results, weights)
        return [h for h in ranked if h["score"] >= self.min_score][:max_sources]

    def _summarize(self, entry: Dict[str, Any], passages: List[List[Any]]):
        """
        (paragraphs, rejected, raw reply, passage sentences) for one citation:
        the model sees the complete retrieved passage(s) behind it, numbered,
        and every statement it writes is checked against that text. A failed
        call gives no summary. The raw reply and the numbered passage are
        returned so a summary can be checked, or a missing one explained.
        """
        texts = [s.text for idx in entry["sources"] for s in passages[idx - 1]]
        numbered = "\n".join(f"({k}) {t}" for k, t in enumerate(texts, 1))
        chosen = set(entry["sentences"])
        focus = [k for k, t in enumerate(texts, 1) if t in chosen]     # the sentences quoted for this note
        try:
            reply = self.client.generate(
                system_prompt=SUMMARY_SYSTEM_PROMPT,
                user_prompt=build_summary_user_prompt(numbered, focus),
                temperature=0.0,
                max_tokens=SUMMARY_MAX_TOKENS,
            ).get("text", "")
        except LLMError as e:
            return [], [{"paragraph": "", "reason": f"{CALL_FAILED_PREFIX}{type(e).__name__}"}], "", texts
        paragraphs, rejected = parse_summary(reply, texts)
        return paragraphs, rejected, reply, texts

    def find_references(
        self,
        soap_note: str,
        top_k: Optional[int] = None,
        max_sources: int = 12,
        summaries: bool = False,
        summary_only: bool = False,
        sections: Optional[List[tuple]] = None,
    ) -> Dict[str, Any]:
        """
        SOAP note -> combined list of relevant textbook references.

        `sections` is an already-split note, [("S", text), ("O", text), ("A", text)].
        When given, the note is not parsed for "S:" / "O:" markers, so text that
        happens to start a line with one is never treated as a section break.

        Extractive: the model only picks which numbered sentences of which
        retrieved passages are relevant. The book details and the quoted
        sentences are stored text, so nothing from outside the books and no
        invented title, author, chapter or page can appear.

        With summaries=True each citation also gets a 'Detailed Summary' of
        its passage. That is the only model-written text; it is checked
        against the passage (see summaries.py) and left out when it cannot be
        supported. summary_only=True (which implies summaries) prints just the
        citation line and the summary.
        """
        summaries = summaries or summary_only
        t0 = time.time()
        # Plan text is not used: the task reads S, O and A and gives no treatment.
        if sections is None:
            sections = parse_soap_sections(soap_note)
        sections = [(label, body) for label, body in sections if label != "P" and body.strip()]
        hits = self.retrieve_for_note(sections, top_k=top_k, max_sources=max_sources)
        retrieval_ms = (time.time() - t0) * 1000

        # Only usable sentences are offered (complete, no glued heading, not a
        # question, not a run-on), so only those can be printed. The whole
        # passage is kept for the summary.
        rows = []
        for h in hits:
            whole = sentences_for(hit_text(h))
            usable = [s for s in whole if s.usable]
            if usable:                                         # passages with no usable sentence are dropped
                rows.append((h, usable, whole))
        hits = [h for h, _, _ in rows]
        sentences = [s for _, s, _ in rows]
        passages = [w for _, _, w in rows]

        raw_answer, model = "", self.client.model
        if hits:
            note_for_model = "\n".join(body for _, body in sections)
            gen = self.client.generate(
                system_prompt=REFERENCE_SYSTEM_PROMPT,
                user_prompt=build_reference_user_prompt(
                    note_for_model, assemble_numbered_context(hits, sentences)),
                temperature=0.0,
            )
            raw_answer = gen.get("text", "")
            model = gen.get("model", model)

        # A reply that is neither a source list nor "no match" is a failure, not "no references".
        failures: List[str] = []
        if hits and not selection_reply_usable(raw_answer):
            failures.append("the model's source list could not be read")

        picks = extract_source_sentences(
            raw_answer, [len(s) for s in sentences], max_per_source=MAX_SENTENCES_PER_SOURCE)
        entries = build_reference_entries(hits, sentences, picks)
        shown = sorted({i for e in entries for i in e["sources"]})
        summary_notes: List[Dict[str, Any]] = []
        if summaries:
            for n, entry in enumerate(entries, 1):
                (entry["summary"], notes, entry["summary_raw"],
                 entry["summary_passage"]) = self._summarize(entry, passages)
                summary_notes.extend({"citation": n, **note} for note in notes)
                problem = summary_failure(entry["summary"], notes)
                if problem:
                    failures.append(f"citation {n}: {problem}")
        return {
            "references": format_reference_list(entries, with_summary=summaries, summary_only=summary_only),
            # Why this result is incomplete (fixed phrases and citation numbers only, never note text).
            # A caller that stores results must not store one with failures.
            "failures": failures,
            "summary_notes": summary_notes,
            "source_indices": sorted(i for i, p in picks.items() if p),
            "dropped_sources": sorted(i for i, p in picks.items() if not p),
            "reference_entries": entries,
            "candidates": [
                {"source": i, "chunk_id": h.get("chunk_id"), "book": h.get("book_title"),
                 "score": round(h["score"], 4), "fused": h.get("fused"),
                 "picked": i in picks and bool(picks[i]), "printed": i in shown}
                for i, h in enumerate(hits, 1)
            ],
            "retrieved_count": len(hits),
            "generator": GENERATOR_VERSION,
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
