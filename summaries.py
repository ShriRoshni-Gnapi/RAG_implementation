"""
Checks for the model-written 'Detailed Summary' of a cited passage.

The summary is the only text in reference mode that the model writes, so it is
held to the passage. The model marks the passage sentences each statement rests
on with numbers in square brackets, either after every sentence or at the end
of a paragraph ("... on stairs [1]. ... is common [2, 3]."). The text before
each bracket is one unit, and a unit is kept only when

  * the numbers in its bracket exist,
  * every number it states appears in the passage,
  * every name or acronym it uses appears in the passage,
  * it shares real vocabulary with the sentences it cites, and
  * it is not a copy of an original sentence.

A unit that fails any check is dropped, never repaired, and a paragraph with no
unit left is dropped. If nothing survives, no summary is printed. The brackets
are removed from the text. These checks catch invented figures, names and
unsupported topics. They cannot prove that a paraphrase is faithful, or that an
added explanation of a term comes from the passage and not from the model.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from prompts import INSUFFICIENT_TEXT

# "[2, 5]", "[S2-S4]" anywhere in a paragraph.
_TAG = re.compile(r"\[\s*([Ss\d,\s\-–]+?)\s*\]")
_LEAD_PUNCT = re.compile(r"^\s*([.,;:!?]+)")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_WORD = re.compile(r"[A-Za-z]+")
_ACRONYM_OR_NAME = re.compile(r"\b(?:[A-Z]{2,}[a-z]?|[A-Z][a-z]{2,})\b")
_SENTENCE_START = re.compile(r"(?:^|(?<=[.!?])\s+)\S+")

# "This suggests that ...", "These symptoms indicate ...", "This process ensures ...":
# the model drawing its own conclusion from the passage, not stating what the
# passage says. Only sentences opening "This"/"These" are caught; others that
# reach a conclusion ("The lack of correlation suggests ...") are not.
_CONCLUSION = re.compile(
    r"^(?:This|These)\b[^.]{0,60}?\b"
    r"(?:suggests?|means?|highlights?|ensures?|indicates?|implies|imply|emphasi[sz]es?|demonstrates?)\b")

MIN_UNIT_CHARS = 25
MIN_SHARED_STEMS = 2
_STEM = 5
_COMMON = frozenset({
    "which", "their", "there", "these", "those", "about", "where", "while", "would", "could",
    "should", "other", "being", "after", "before", "between", "because", "during", "often",
    "using", "within", "without", "passage", "author", "states", "explains", "describes",
})


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.lower()).split())


def _stems(text: str) -> set:
    return {w[:_STEM] for w in (x.lower() for x in _WORD.findall(text)) if len(w) >= _STEM and w not in _COMMON}


def _evidence_ids(tag: str, count: int) -> List[int]:
    ids: List[int] = []
    for lo, hi, single in re.findall(r"[Ss]?(\d+)\s*[-–]\s*[Ss]?(\d+)|[Ss]?(\d+)", tag):
        if single:
            ids.append(int(single))
        elif int(lo) <= int(hi) <= int(lo) + 50:
            ids.extend(range(int(lo), int(hi) + 1))
    return [k for k in dict.fromkeys(ids) if 1 <= k <= count]


def _ungrounded_term(text: str, passage_words: set) -> Optional[str]:
    """A name or acronym in `text` (not a sentence's first word) missing from the passage."""
    starts = {m.start() for m in _SENTENCE_START.finditer(text)}
    for m in _ACRONYM_OR_NAME.finditer(text):
        if m.start() in starts:
            continue
        term = m.group().lower()
        if not any(w.startswith(term) or (len(w) >= 3 and term.startswith(w)) for w in passage_words):
            return m.group()
    return None


def _reject_reason(text: str, ids: List[int], sentences: List[str], passage: str,
                   passage_words: set, passage_numbers: set) -> Optional[str]:
    if not ids:
        return "no valid sentence numbers"
    if len(text) < MIN_UNIT_CHARS:
        return "too short"
    if _CONCLUSION.match(text):
        return "a conclusion, not a statement from the passage"
    norm = _norm(text)
    if any(len(s) > 40 and _norm(s) in norm for s in sentences):
        return "copies an original sentence"
    missing = [n for n in _NUMBER.findall(text) if n not in passage_numbers]
    if missing:
        return f"number not in the passage: {missing[0]}"
    term = _ungrounded_term(text, passage_words)
    if term:
        return f"name or term not in the passage: {term}"
    cited = _stems(" ".join(sentences[k - 1] for k in ids))
    mine = _stems(text)
    if len(cited & mine) < min(MIN_SHARED_STEMS, max(1, len(mine))):
        return "not supported by the sentences it cites"
    return None


def parse_summary(reply: str, sentences: List[str]) -> Tuple[List[str], List[Dict[str, Any]]]:
    """
    (paragraphs, rejected) from the model's summary reply.

    `sentences` are the numbered passage sentences the model was shown.
    `rejected` lists each dropped unit (text before a bracket) with its
    reason, for debugging.
    """
    if not (reply or "").strip():
        return [], [{"paragraph": "", "reason": "empty reply from the model"}]
    if reply.strip().upper().startswith(INSUFFICIENT_TEXT):
        return [], [{"paragraph": "", "reason": "the model said the passage is insufficient"}]

    passage = " ".join(sentences)
    passage_words = {w.lower() for w in _WORD.findall(passage)}
    passage_numbers = set(_NUMBER.findall(passage))
    kept: List[str] = []
    rejected: List[Dict[str, Any]] = []
    for line in reply.splitlines():
        line = line.strip().lstrip("-*• ").strip()
        if not line:
            continue
        good: List[str] = []
        for text, tag in _units(line):
            reason = _reject_reason(text, _evidence_ids(tag, len(sentences)), sentences,
                                    passage, passage_words, passage_numbers)
            if reason:
                rejected.append({"paragraph": text, "reason": reason})
            else:
                good.append(text if re.search(r"[.!?]$", text) else f"{text}.")
        if good:
            kept.append(" ".join(good))
    return kept, rejected


def _units(line: str) -> List[Tuple[str, str]]:
    """
    [(text, bracket numbers)] for one paragraph line. The text before each
    bracket is a unit; punctuation right after a bracket goes back to the unit
    it ends; a bracket with no text before it (a closing "[1, 2, 3]") adds
    nothing; text after the last bracket is a unit with no numbers.
    """
    pieces = _TAG.split(line)                    # text, numbers, text, numbers, ..., text
    units: List[List[str]] = []
    for i in range(0, len(pieces) - 1, 2):
        text, tag = pieces[i], pieces[i + 1]
        lead = _LEAD_PUNCT.match(text) if units else None
        if lead:
            units[-1][0] += lead.group(1)
            text = text[lead.end():]
        if text.strip():
            units.append([text.strip(), tag])
    tail = pieces[-1]
    lead = _LEAD_PUNCT.match(tail) if units else None
    if lead:
        units[-1][0] += lead.group(1)
        tail = tail[lead.end():]
    if tail.strip():
        units.append([tail.strip(), ""])
    return [(text, tag) for text, tag in units]
