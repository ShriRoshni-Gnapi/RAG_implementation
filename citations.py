"""
Citation extraction, validation, and source attribution for PT-RAG.

Implements:
  * Numbered indirection: model emits markers [1], [2]; system renders citations.
  * Validation: invalid or out-of-range markers are caught and audited.
  * Provenance: attaches full book metadata, section hierarchy, page numbers,
    and source file references.
"""

import re
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple, Optional

from config import BOOKS_DIR, BOOKS
from prompts import NO_REFERENCE_TEXT

DISCLAIMER_TEXT = (
    "Clinical decision-making requires individual patient assessment."
)


def extract_citation_markers(text: str) -> List[int]:
    """
    Extract all integer citation numbers from patterns like [1], [1, 2], [1, 3, 4].
    Returns a sorted, unique list of integer IDs found in the text.
    """
    if not text:
        return []

    pattern = r"\[(\d+(?:\s*,\s*\d+)*)\]"
    found: Set[int] = set()

    for match in re.finditer(pattern, text):
        nums_str = match.group(1)
        for part in nums_str.split(","):
            part = part.strip()
            if part.isdigit():
                found.add(int(part))

    return sorted(list(found))


def validate_citations(
    text: str, hits: List[Dict[str, Any]]
) -> Tuple[str, List[int], List[int], Dict[str, Any]]:
    """
    Validate and clean any stray citation markers from the output text.

    Returns:
      (cleaned_text, valid_indices, invalid_indices, audit_metadata)
    """
    num_hits = len(hits)
    extracted = extract_citation_markers(text)

    valid_indices = [idx for idx in extracted if 1 <= idx <= num_hits]
    invalid_indices = [idx for idx in extracted if idx < 1 or idx > num_hits]

    # Clean any bracketed citation numbers from prose
    cleaned_text = re.sub(r"\[\s*\d+(?:\s*,\s*\d+)*\s*\]", "", text)
    # Clean up double spaces left by removed brackets
    cleaned_text = re.sub(r"  +", " ", cleaned_text)
    # Clean space before punctuation
    cleaned_text = re.sub(r" ([.,;:!?])", r"\1", cleaned_text)

    audit = {
        "total_extracted": len(extracted),
        "valid_count": len(valid_indices),
        "invalid_count": len(invalid_indices),
        "valid_indices": valid_indices,
        "invalid_indices": invalid_indices,
        "citation_precision": (len(valid_indices) / len(extracted)) if extracted else 1.0,
    }

    return cleaned_text.strip(), valid_indices, invalid_indices, audit


def render_citation_entry(hit: Dict[str, Any], index: Optional[int] = None) -> Dict[str, Any]:
    """
    Format a rich bibliographical reference entry for a single hit.
    """
    book_id = hit.get("book_id", "")
    book_meta = BOOKS.get(book_id, {})

    book = hit.get("book_title") or book_meta.get("title") or hit.get("book") or "Unknown Book"
    author = hit.get("author") or book_meta.get("author")
    edition = hit.get("edition") or book_meta.get("edition")
    year = hit.get("year") or book_meta.get("year")

    # Build bibliographic heading
    biblio_parts = [book]
    extra = []
    if author:
        extra.append(author)
    if edition:
        extra.append(f"{edition} ed.")
    if year:
        extra.append(str(year))
    if extra:
        biblio_parts.append(f"({', '.join(extra)})")
    biblio_str = " ".join(biblio_parts)

    # Section hierarchy
    ch = hit.get("chapter")
    sec = hit.get("section")
    subsec = hit.get("subsection")

    hierarchy_parts = []
    if ch and ch != "None":
        hierarchy_parts.append(f"Ch. {ch}")
    if sec and sec != "None":
        hierarchy_parts.append(f"Sec. {sec}")
    if subsec and subsec != "None":
        hierarchy_parts.append(f"Subsec. {subsec}")
    hierarchy_str = " · ".join(hierarchy_parts)

    # Pages
    p_start = hit.get("page_start")
    p_end = hit.get("page_end")
    if p_start is not None and p_end is not None:
        page_str = f"p. {p_start}" if p_start == p_end else f"pp. {p_start}–{p_end}"
    else:
        page_str = "p. [unspecified]"

    pdf_page_start = hit.get("pdf_page_start", p_start)
    source_file = hit.get("source_file") or book_meta.get("source_file", "")

    formatted_lines = [f"• {biblio_str}"]
    if hierarchy_str:
        formatted_lines.append(f"    {hierarchy_str} · {page_str}")
    else:
        formatted_lines.append(f"    {page_str}")

    return {
        "index": index or 0,
        "book": book,
        "author": author,
        "edition": edition,
        "year": year,
        "chapter": ch,
        "section": sec,
        "subsection": subsec,
        "page_label": page_str,
        "pdf_page_start": pdf_page_start,
        "source_file": source_file,
        "formatted_text": "\n".join(formatted_lines),
        "chunk_id": hit.get("chunk_id"),
        "score": hit.get("score"),
        "preview": (hit.get("content") or hit.get("text") or "")[:200],
    }


def extract_source_sentences(text: str, sentence_counts: List[int]) -> Dict[int, List[int]]:
    """
    Reference mode: {source number: picked sentence numbers} from the model's
    reply ("Source 3: 2, 4").

    A source or sentence number outside what was shown is dropped, so a label
    the model made up can never reach the printed list. A source named with
    no valid sentence numbers maps to [] and is printed in full instead.
    """
    parts = re.split(r"Source\s*(\d+)", text or "", flags=re.I)
    out: Dict[int, List[int]] = {}
    for num, body in zip(parts[1::2], parts[2::2]):
        idx = int(num)
        if not (1 <= idx <= len(sentence_counts)) or idx in out:
            continue
        picked = {int(n) for n in re.findall(r"\d+", body)}
        out[idx] = sorted(k for k in picked if 1 <= k <= sentence_counts[idx - 1])
    return out


def _join_sentences(sentences: List[str], picked: List[int]) -> str:
    """Picked sentences exactly as stored; '[...]' marks a skipped stretch."""
    pieces, prev = [], None
    for k in picked:
        if prev is not None and k != prev + 1:
            pieces.append("[...]")
        pieces.append(sentences[k - 1])
        prev = k
    return " ".join(pieces)


def format_reference_list(
    hits: List[Dict[str, Any]],
    sentences: List[List[str]],
    picks: Dict[int, List[int]],
) -> str:
    """
    Combined reference list written entirely from stored data: book, author,
    chapter, section and pages from metadata, and the excerpt as the stored
    sentences the model picked, unchanged. The model never writes any text
    that is printed. A source with no valid sentence numbers shows its full
    stored passage.
    """
    if not picks:
        return NO_REFERENCE_TEXT

    blocks = []
    for n, idx in enumerate(sorted(picks), 1):
        entry = render_citation_entry(hits[idx - 1], idx)
        chosen = picks[idx] or list(range(1, len(sentences[idx - 1]) + 1))
        excerpt = _join_sentences(sentences[idx - 1], chosen)
        head = entry["formatted_text"].replace("• ", "", 1)
        blocks.append(f"{n}. {head}\n    Excerpt: {excerpt}")
    return "\n\n".join(blocks)


def format_citations_block(
    hits: List[Dict[str, Any]], cited_indices: Optional[List[int]] = None
) -> str:
    """
    Format the complete REFERENCED TEXTBOOKS block for terminal or UI display.
    """
    entries = []
    seen_sources = set()

    for idx, hit in enumerate(hits, 1):
        entry = render_citation_entry(hit, idx)
        formatted = entry["formatted_text"]
        if formatted not in seen_sources:
            seen_sources.add(formatted)
            entries.append(formatted)

    if not entries:
        return ""

    header = "REFERENCED TEXTBOOKS & SOURCES\n" + ("─" * 60)
    body = "\n\n".join(entries)
    disclaimer = f"\n{'─' * 60}\n{DISCLAIMER_TEXT}"
    return f"{header}\n{body}\n{disclaimer}"
