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
from passages import Sentence
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

    Every source and sentence number is checked against what was actually
    shown, so an id the model made up can never reach the printed list. Ranges
    such as "2-4" are expanded. A source named with no valid sentence number
    maps to [] and the caller drops it: nothing is shown for it.
    """
    parts = re.split(r"Source\s*(\d+)", text or "", flags=re.I)
    out: Dict[int, List[int]] = {}
    for num, body in zip(parts[1::2], parts[2::2]):
        idx = int(num)
        if not (1 <= idx <= len(sentence_counts)) or idx in out:
            continue
        picked: Set[int] = set()
        for lo, hi, single in re.findall(r"(\d+)\s*[-–]\s*(\d+)|(\d+)", body):
            if single:
                picked.add(int(single))
            elif int(lo) <= int(hi) <= int(lo) + 50:
                picked.update(range(int(lo), int(hi) + 1))
        out[idx] = sorted(k for k in picked if 1 <= k <= sentence_counts[idx - 1])
    return out


# --------------------------------------------------------------------------
# Reference list: grouping, metadata and printing (all from stored data)
# --------------------------------------------------------------------------

_CUT = "[...]"
_CHAPTER_LABEL = re.compile(r"^(?:chapter|ch\.?)\s+\w+|^\d{1,3}\b", re.I)
_JUNK_LABEL = re.compile(r"^(?:scan|page|img|image)\d+$", re.I)
_MISSING = (None, "", "None")


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _clean_label(value: Any) -> Optional[str]:
    """A stored outline label, or None when it is missing or scanner junk."""
    if value in _MISSING or _JUNK_LABEL.match(str(value).strip()):
        return None
    return str(value).strip()


def chapter_and_section(hit: Dict[str, Any], title: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Chapter and section as stored. The outline path is used to find the real
    chapter (a book's root entry is often stored in the chapter field); labels
    that are missing or scanner junk ('scan0023') are not shown.
    """
    path = [p.strip() for p in (hit.get("section_path") or "").split(">") if p.strip()]
    for i, label in enumerate(path):
        if _CHAPTER_LABEL.match(label):
            rest = [p for p in path[i + 1:] if _clean_label(p)]
            return label, " > ".join(rest) or None

    chapter = _clean_label(hit.get("chapter"))
    if chapter and _norm(chapter) == _norm(title):
        chapter = None                                  # root entry = the book itself
    rest = [v for v in (_clean_label(hit.get("section")), _clean_label(hit.get("subsection"))) if v]
    return chapter, " > ".join(dict.fromkeys(rest)) or None


def page_text(hit: Dict[str, Any]) -> str:
    """
    Pages as stored. The stored range belongs to the whole section the passage
    sits in, not to the passage itself, so a range is labelled as such.
    """
    start, end = hit.get("page_start"), hit.get("page_end")
    if start in _MISSING and end in _MISSING:
        return "Pages: not recorded"
    start, end = (start if start not in _MISSING else end), (end if end not in _MISSING else start)
    if str(start) == str(end):
        return f"Page: {start}"
    return f"Pages: {start}–{end} (section range)"


def _excerpt(sentences: List[Sentence], picked: List[int]) -> str:
    """Picked sentences exactly as stored; '[...]' marks skipped or cut text."""
    pieces: List[str] = []
    prev = None
    for k in picked:
        s = sentences[k - 1]
        if prev is not None and k != prev + 1:
            pieces.append(_CUT)
        text = s.text
        if s.cut_start and not (pieces and pieces[-1] == _CUT):
            text = f"{_CUT} {text}"
        if s.cut_end:
            text = f"{text} {_CUT}"
        pieces.append(text)
        prev = k
    return " ".join(pieces)


def _is_repeat(sentence: str, printed: List[str]) -> bool:
    """True when this sentence already appears in text printed for the same book."""
    norm = _norm(sentence)
    return not norm or any(norm in p for p in printed)


def build_reference_entries(
    hits: List[Dict[str, Any]],
    sentences: List[List[Sentence]],
    picks: Dict[int, List[int]],
) -> List[Dict[str, Any]]:
    """
    Reference entries from the validated picks.

    Passages from the same book, chapter and section share one entry (one
    heading, one excerpt per passage, in book order). Different sections of
    the same chapter stay separate entries. A sentence already printed for
    the same book (overlapping chunks) is not printed twice.
    """
    groups: Dict[tuple, Dict[str, Any]] = {}
    for idx in sorted(i for i, p in picks.items() if p):
        hit = hits[idx - 1]
        meta = BOOKS.get(hit.get("book_id", ""), {})
        title = hit.get("book_title") or hit.get("book") or meta.get("title") or "Unknown book"
        chapter, section = chapter_and_section(hit, title)
        key = (hit.get("book_id") or title, chapter, section)
        group = groups.setdefault(key, {
            "book": title,
            "author": hit.get("author") or meta.get("author"),
            "chapter": chapter,
            "section": section,
            "pages": page_text(hit),
            "passages": [],
        })
        group["passages"].append((hit.get("chunk_id") or "", idx))

    entries: List[Dict[str, Any]] = []
    printed: Dict[str, List[str]] = {}
    for key, group in groups.items():
        seen = printed.setdefault(str(key[0]), [])
        excerpts, chunk_ids, sources = [], [], []
        for chunk_id, idx in sorted(group.pop("passages")):
            kept = []
            for k in picks[idx]:
                if not _is_repeat(sentences[idx - 1][k - 1].text, seen):
                    kept.append(k)
                    seen.append(_norm(sentences[idx - 1][k - 1].text))
            if kept:
                excerpts.append(_excerpt(sentences[idx - 1], kept))
                chunk_ids.append(chunk_id)
                sources.append(idx)
        if excerpts:
            entries.append({**group, "excerpts": excerpts, "chunk_ids": chunk_ids, "sources": sources})
    return sorted(entries, key=lambda e: min(e["sources"]))


def format_reference_list(entries: List[Dict[str, Any]]) -> str:
    """
    One combined reference list: title, author (if known), chapter, section
    (if known), pages, and the original textbook sentences. Nothing in it is
    written by the model.
    """
    if not entries:
        return NO_REFERENCE_TEXT

    blocks = []
    for n, e in enumerate(entries, 1):
        lines = [f"{n}. {e['book']}"]
        for label, value in (("Author", e["author"]), ("Chapter", e["chapter"]), ("Section", e["section"])):
            if value:
                lines.append(f"    {label}: {value}")
        lines.append(f"    {e['pages']}")
        lines.extend(f"    Excerpt: {x}" for x in e["excerpts"])
        blocks.append("\n".join(lines))
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
