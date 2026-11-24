"""
Passage text handling for SOAP reference mode.

Stored passages come from PDF extraction, so they carry page furniture
(running heads and page numbers) and chunk edges that cut a sentence in half.
This module turns one stored passage into numbered sentences.

It never calls a model and never rewrites a word. Page furniture is only used
to find where a page break fell; a sentence touching a page break or a chunk
edge is flagged as cut, so the caller can leave it out instead of printing
half a sentence. Every sentence returned is a slice of the stored text
(whitespace collapsed to single spaces), with one deliberate exception: the
book's own reference numbers ("factors.27,47,135") are removed.
"""

import re
from typing import Any, Dict, List, NamedTuple, Set


class Sentence(NamedTuple):
    """One sentence of a stored passage."""
    text: str
    cut_start: bool = False   # the sentence began before this passage / before a page break
    cut_end: bool = False     # the sentence continues after this passage / after a page break

    @property
    def complete(self) -> bool:
        return not (self.cut_start or self.cut_end)


def hit_text(hit: Dict[str, Any]) -> str:
    """The stored passage text of a retrieved chunk, whitespace normalised."""
    return " ".join((hit.get("content") or hit.get("text") or "").split())


# --------------------------------------------------------------------------
# Page furniture
# --------------------------------------------------------------------------

_PAGE_BREAK = "\x01"

# One ALL-CAPS token: "DYSFUNCTION:", "NONOPERATIVE", "(CONTINUED)".
_CAPS_TOKEN = r"[A-Z][A-Z0-9'’:,/&()\-]*"
_CAPS_RUN = rf"(?:{_CAPS_TOKEN}\s+){{2,}}"
# "712 PATELLOFEMORAL DYSFUNCTION: NONOPERATIVE MANAGEMENT" (number first)
_NUMBER_THEN_HEAD = re.compile(rf"(?:(?<=\s)|^)\d{{1,4}}\s+({_CAPS_RUN})")
# "NONOPERATIVE MANAGEMENT 713" (number last)
_HEAD_THEN_NUMBER = re.compile(rf"(?:(?<=\s)|^)({_CAPS_RUN})\d{{1,4}}(?=\s|$)")
# A real running head has long words; this keeps "12 ACL PCL tears" intact.
_MIN_HEAD_LETTERS = 12


def _mark_page_furniture(text: str) -> str:
    """Replace page number + running head by a page-break marker."""
    def drop(m: "re.Match[str]") -> str:
        head = m.group(1)
        return _PAGE_BREAK if sum(c.isalpha() for c in head) >= _MIN_HEAD_LETTERS else m.group(0)

    text = _NUMBER_THEN_HEAD.sub(drop, text)
    return _HEAD_THEN_NUMBER.sub(drop, text)


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

_ABBREVIATIONS = frozenset({
    "fig", "figs", "tab", "dr", "mr", "mrs", "ms", "prof", "vs", "approx", "ca",
    "no", "nos", "vol", "pp", "p", "ed", "eds", "al", "cf", "ref", "refs", "eq",
    "st", "jr", "sr", "inc", "ltd", "co", "dept", "e.g", "i.e",
})

# Sentence end: . ! ? (closing quote/bracket), optionally followed by the book's
# reference numbers (".27,47"). Reference numbers only count after punctuation
# that does not follow a digit, so the decimal in "Level 3.2 (Figure 5)" is not
# read as a sentence end, while "6/10. Reports" still is.
_SENTENCE_END = (
    r"""(?:(?<!\d)[.!?]["'”’)\]]*\d{1,3}(?:\s?[,–\-]\s?\d{1,3})*|[.!?]["'”’)\]]*)"""
)
_BOUNDARY = re.compile(rf"({_SENTENCE_END})\s+(?=[A-Z(\[\d])")
_TERMINAL = re.compile(rf"{_SENTENCE_END}$")
_BULLET = re.compile(r"\s*•\s*")


def _ends_with_abbreviation(before: str) -> bool:
    m = re.search(r"([A-Za-z][A-Za-z.]*)$", before)
    if not m:
        return False
    word = m.group(1).lower().rstrip(".")
    return word in _ABBREVIATIONS or len(word.replace(".", "")) == 1   # 'Fig.', 'J.'


def _split_sentences_plain(text: str) -> List[str]:
    """Sentence split that respects abbreviations, decimals and reference numbers."""
    out, start = [], 0
    for m in _BOUNDARY.finditer(text):
        punct_at = m.start(1)
        if text[punct_at] == "." and _ends_with_abbreviation(text[:punct_at]):
            continue
        out.append(text[start:m.end(1)].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if sum(c.isalpha() for c in s) >= 3]


def split_text(text: str) -> List[str]:
    """Plain sentence split of arbitrary text (used for the SOAP note)."""
    out: List[str] = []
    for part in _BULLET.split(" ".join((text or "").split())):
        out.extend(_split_sentences_plain(part))
    return out


# The book's reference numbers are raised numbers in print, so extraction leaves
# them glued to the punctuation: "factors.27,47,135", "quadriceps,231 so",
# "(Figure 12-5).31". Only digits that follow a letter or closing bracket plus
# punctuation, with no space, match; "0.5", "30.5", "3.2" and "(Fig. 4-44)" do not.
_REFERENCE_NUMBERS = re.compile(
    r"""(?<=[A-Za-z)\]"'”’])([.,;:])\d{1,3}(?:\s?[,–\-]\s?\d{1,3})*(?=\s|$)"""
)


def strip_reference_numbers(text: str) -> str:
    """Remove the book's own reference numbers, keeping the punctuation they followed."""
    return _REFERENCE_NUMBERS.sub(r"\1", text)


def sentences_for(text: str) -> List[Sentence]:
    """
    A stored passage as sentences, each flagged when it is cut off.

    A sentence is cut at the start when it begins in lower case at the start
    of the passage or right after a page break, and cut at the end when it
    has no closing punctuation at the end of the passage or right before a
    page break. Nothing is joined or rewritten.
    """
    text = _mark_page_furniture(" ".join((text or "").split()))

    frags: List[str] = []
    after_break: Set[int] = set()       # fragments that start right after a page break
    before_break: Set[int] = set()      # fragments that end right before a page break
    for b, block in enumerate(re.split(rf"\s*{_PAGE_BREAK}\s*", text)):
        parts = split_text(block)
        if not parts:
            continue
        if b and frags:
            before_break.add(len(frags) - 1)
        if b:
            after_break.add(len(frags))
        frags.extend(parts)

    last = len(frags) - 1
    return [
        Sentence(
            text=strip_reference_numbers(f),
            cut_start=f[:1].islower() and (i == 0 or i in after_break),
            cut_end=(i == last or i in before_break) and not _TERMINAL.search(f),
        )
        for i, f in enumerate(frags)
    ]
