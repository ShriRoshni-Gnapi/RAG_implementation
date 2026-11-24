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
    heading: bool = False     # a chapter title or running head is glued into the text

    @property
    def complete(self) -> bool:
        return not (self.cut_start or self.cut_end)

    @property
    def usable(self) -> bool:
        """
        Safe to offer and print: complete, free of glued headings, not a
        question (history-taking prompts in the books), not a run-on that
        merges list items or headings into one 'sentence', and not a sentence
        that depends on the one before it.
        """
        return (self.complete and not self.heading
                and not _QUESTION.search(self.text) and not _is_run_on(self.text)
                and not _depends_on_previous(self.text))


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
# A real running head is long and has at least one long word ("DYSFUNCTION").
# A run of acronyms ("ACL PCL MCL LCL", 12 letters) has none, so it is text.
_MIN_HEAD_LETTERS = 12
_MIN_HEAD_WORD_LETTERS = 6


def _is_head(run: str) -> bool:
    """True for an ALL-CAPS run that reads as a running head, not a list of acronyms."""
    lengths = [sum(c.isalpha() for c in word) for word in run.split()]
    return sum(lengths) >= _MIN_HEAD_LETTERS and max(lengths, default=0) >= _MIN_HEAD_WORD_LETTERS


def _mark_page_furniture(text: str) -> str:
    """Replace page number + running head by a page-break marker."""
    def drop(m: "re.Match[str]") -> str:
        return _PAGE_BREAK if _is_head(m.group(1)) else m.group(0)

    text = _NUMBER_THEN_HEAD.sub(drop, text)
    return _HEAD_THEN_NUMBER.sub(drop, text)


# A long ALL-CAPS run or a "CHAPTER FIVE" start inside a sentence is a heading
# that the extraction glued to the text ("CHAPTER FIVE The Knee 241 ...
# REHABILITATION RATIONALE Patellofemoral dysfunction, or disorder, ...").
_CAPS_RUN_ANY = re.compile(rf"(?:{_CAPS_TOKEN}(?:\s+|$)){{2,}}")
_CHAPTER_START = re.compile(r"^(?:CHAPTER|PART|SECTION|APPENDIX)\s+\w+\b")


# A Title Case heading glued to the front of a sentence, with no punctuation
# between: "Etiology of Symptoms The cause of ..." (a heading, then a sentence
# starting "The") and "Common Impairments and Functional Limitations/Disabilities
# Impairments that may ..." (four or more Title Case words joined by a word such
# as "and" or "of", then lower case). Without a joining word, a run of Title Case
# words is a name inside a real sentence ("Magee Orthopedic Physical Assessment
# describes ...", "Anterior Cruciate Ligament Injury Prevention Programs reduce
# ..."), so it is left alone; so are three-word names ("The McConnell Test is ...").
_TITLE_WORD = r"[A-Z][A-Za-z]+(?:/[A-Z][A-Za-z]+)?"
_CONNECTOR = r"(?:of|and|in|for|to|on|with)"
_SENTENCE_STARTERS = r"(?:The|A|An|In|It|This|These|When|If|During|There)"
_HEADING_THEN_SENTENCE = re.compile(
    rf"^{_TITLE_WORD}\s+{_CONNECTOR}\s+{_TITLE_WORD}\s+{_SENTENCE_STARTERS}\s+[a-z]")
_TITLE_RUN = re.compile(rf"^(?:{_TITLE_WORD}(?:\s+{_CONNECTOR})?\s+){{4,}}[a-z]")
_JOINER = re.compile(rf"\s{_CONNECTOR}\s")


def _is_title_heading(text: str) -> bool:
    m = _TITLE_RUN.match(text)
    return bool(m and _JOINER.search(m.group()[:-1]))


def _has_heading(text: str) -> bool:
    if _CHAPTER_START.match(text) or _HEADING_THEN_SENTENCE.match(text) or _is_title_heading(text):
        return True
    return any(_is_head(m.group()) for m in _CAPS_RUN_ANY.finditer(text))


# A sentence that begins "However", "This", "These" or "It" usually leans on the
# sentence before it, so it is not offered on its own ("This signals lateral
# patellar instability.", "These include treatment modalities ...", "It is
# performed with the patient supine ..."). Two openings stand on their own and
# are kept: "This test/sign/maneuver ..." (the citation names the test) and an
# empty "It" with a clause after it ("It is important to measure both sides ...").
_DEPENDENT_START = re.compile(r"^(?:However|This|These|It)\b")
_SELF_CONTAINED_THIS = re.compile(
    r"^This\s+(?:test|sign|maneuver|manoeuvre|procedure|technique|position|exercise|method|"
    r"measurement|scale|score|classification|grade)\b")
_EMPTY_IT = re.compile(
    r"^It\s+(?:is|was|may be|can be|should be|must be)\s+(?:important|imperative|essential|necessary|"
    r"critical|vital|useful|helpful|recommended|advisable|appropriate|difficult|possible|likely|"
    r"unlikely|common)\s+(?:to|that)\b")


def _depends_on_previous(text: str) -> bool:
    if not _DEPENDENT_START.match(text):
        return False
    return not (_SELF_CONTAINED_THIS.match(text) or _EMPTY_IT.match(text))


# --------------------------------------------------------------------------
# Questions and run-ons
# --------------------------------------------------------------------------

_QUESTION = re.compile(r"""\?["'”’)\]]*$""")

# A real sentence is rarely longer than this. A run-on that merges list items
# or headings also shows several lower-case word -> Capitalised word jumps
# with no punctuation between them ("activities Pain", "squatting Patellofemoral").
MAX_SENTENCE_CHARS = 400
_RUN_ON_MIN_CHARS = 200
_RUN_ON_JUMPS = 3
_CASE_JUMP = re.compile(r"(?<=[a-z]) [A-Z][a-z]{2,}\b")


def _is_run_on(text: str) -> bool:
    if len(text) > MAX_SENTENCE_CHARS:
        return True
    return len(text) > _RUN_ON_MIN_CHARS and len(_CASE_JUMP.findall(text)) >= _RUN_ON_JUMPS


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
            heading=_has_heading(f),
        )
        for i, f in enumerate(frags)
    ]


# Every pattern and limit above that decides which sentences are offered. They are
# hashed into the generator version (query_engine.generator_fingerprint), so a stored
# reference set can be recognised as made by different rules. A new pattern or limit
# must be added here; a test fails if one is missing.
_RULE_NAMES = (
    "_ABBREVIATIONS", "_BOUNDARY", "_TERMINAL", "_BULLET", "_REFERENCE_NUMBERS", "_NUMBER_THEN_HEAD",
    "_HEAD_THEN_NUMBER", "_MIN_HEAD_LETTERS", "_MIN_HEAD_WORD_LETTERS", "_CAPS_RUN_ANY", "_CHAPTER_START",
    "_HEADING_THEN_SENTENCE", "_TITLE_RUN", "_JOINER", "_DEPENDENT_START", "_SELF_CONTAINED_THIS", "_EMPTY_IT",
    "_QUESTION", "MAX_SENTENCE_CHARS", "_RUN_ON_MIN_CHARS", "_RUN_ON_JUMPS", "_CASE_JUMP",
)


def rules() -> tuple:
    """The sentence rules' patterns and limits, in a stable form for hashing."""
    out = []
    for name in _RULE_NAMES:
        value = globals()[name]
        out.append((name, value.pattern if isinstance(value, re.Pattern)
                    else sorted(value) if isinstance(value, frozenset) else value))
    return tuple(out)
