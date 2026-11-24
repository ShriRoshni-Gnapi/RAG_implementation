"""
Passage text handling for SOAP reference mode.

Stored passages come from PDF extraction, so they carry page furniture
(running heads and page numbers), words split at line wraps, and chunk edges
that cut a sentence in half. This module turns one stored passage into
numbered sentences for selection and printing.

It never calls a model and never adds words. It only
  * removes recognisable page furniture (page number + running head),
  * rejoins a word split by a line wrap when the joined word appears intact
    elsewhere in the retrieved passages, and
  * marks sentences whose start or end is cut off.
Everything else is the stored text, unchanged.
"""

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Set


class Sentence(NamedTuple):
    """One sentence of a stored passage."""
    text: str
    cut_start: bool = False   # the passage resumes mid-sentence here
    cut_end: bool = False     # the passage stops mid-sentence here


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
# Words split by a line wrap
# --------------------------------------------------------------------------

def build_vocab(texts: Iterable[str]) -> Counter:
    """Word counts over the retrieved passages, used to confirm a rejoin."""
    vocab: Counter = Counter()
    for t in texts:
        vocab.update(w.lower() for w in re.findall(r"[A-Za-z]{2,}", t))
    return vocab


_WORD = re.compile(r"[A-Za-z]{2,}")


def repair_split_words(text: str, vocab: Counter, known_min: int = 2) -> str:
    """
    Rejoin 'patel lofemoral' -> 'patellofemoral'.

    Two neighbouring words (one space apart, the second all lower-case) are
    joined only when the joined word occurs at least `known_min` times in the
    retrieved passages AND the two pieces are not both ordinary words there.
    Pairs such as 'in to' or 'knee cap' are left alone, and a word that does
    not appear intact anywhere is never guessed.
    """
    words = list(_WORD.finditer(text))
    out: List[str] = []
    pos = i = 0
    while i < len(words):
        a = words[i]
        b = words[i + 1] if i + 1 < len(words) else None
        if b and text[a.end():b.start()] == " " and b.group().islower():
            first, second = a.group(), b.group()
            both_words = vocab.get(first.lower(), 0) >= known_min and vocab.get(second, 0) >= known_min
            if vocab.get((first + second).lower(), 0) >= known_min and not both_words:
                out.append(text[pos:a.start()] + first + second)
                pos = b.end()
                i += 2
                continue
        i += 1
    out.append(text[pos:])
    return "".join(out)


# --------------------------------------------------------------------------
# Sentence splitting
# --------------------------------------------------------------------------

_ABBREVIATIONS = frozenset({
    "fig", "figs", "tab", "dr", "mr", "mrs", "ms", "prof", "vs", "approx", "ca",
    "no", "nos", "vol", "pp", "p", "ed", "eds", "al", "cf", "ref", "refs", "eq",
    "st", "jr", "sr", "inc", "ltd", "co", "dept", "e.g", "i.e",
})

# Sentence end: . ! ? (closing quote/bracket) (reference numbers like .27,47)
# followed by a space and something that can start a sentence.
_BOUNDARY = re.compile(
    r"""([.!?]["'”’)\]]*(?:\d{1,3}(?:\s?[,–\-]\s?\d{1,3})*)?)\s+(?=[A-Z(\[\d])"""
)
_TERMINAL = re.compile(r"""[.!?]["'”’)\]]*(?:\d{1,3}(?:\s?[,–\-]\s?\d{1,3})*)?$""")
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


def sentences_for(text: str, vocab: Optional[Counter] = None) -> List[Sentence]:
    """
    A stored passage as numbered sentences.

    Page furniture is removed. A sentence interrupted by a page break is
    rejoined. A sentence that starts or stops mid-thought is flagged so the
    printed excerpt shows '[...]' there.
    """
    text = _mark_page_furniture(" ".join((text or "").split()))
    if vocab:
        text = repair_split_words(text, vocab)

    frags: List[str] = []
    resumed_after_break: Set[int] = set()
    for b, block in enumerate(re.split(rf"\s*{_PAGE_BREAK}\s*", text)):
        parts = split_text(block)
        if not parts:
            continue
        starts_lower = parts[0][:1].islower()
        if b and frags and starts_lower and not _TERMINAL.search(frags[-1]):
            frags[-1] = f"{frags[-1]} {parts.pop(0)}"       # page break inside a sentence
        elif b and starts_lower:
            resumed_after_break.add(len(frags))
        frags.extend(parts)

    last = len(frags) - 1
    return [
        Sentence(
            text=f,
            cut_start=f[:1].islower() and (i == 0 or i in resumed_after_break),
            cut_end=(i == last and not _TERMINAL.search(f)),
        )
        for i, f in enumerate(frags)
    ]
