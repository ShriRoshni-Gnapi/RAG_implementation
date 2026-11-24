"""
Tests for SOAP reference mode (ask.py --soap-file).

Run:   python3 -m unittest test_references -v
Live:  PT_RAG_LIVE=1 python3 -m unittest test_references.LiveModelCheck -v

Unit tests use a MOCKED model and MOCKED search results. They prove the
plumbing: query building, ranking, validation of ids, grouping, de-duplication,
sentence handling and printing. They do NOT prove that the real model picks the
right sentences or that the real Qdrant data ranks well; only the live check
(which needs Qdrant, the embedding model and the LLM server) can show that.
"""

import os
import re
import sys
import types
import unittest

try:                                    # unit tests need no Qdrant server
    import qdrant_client  # noqa: F401
except ImportError:
    for _name in ("qdrant_client", "qdrant_client.models"):
        _m = types.ModuleType(_name)
        for _a in ("QdrantClient", "Distance", "VectorParams", "PointStruct", "Filter",
                   "FieldCondition", "MatchValue", "MatchAny", "FilterSelector"):
            setattr(_m, _a, object)
        sys.modules[_name] = _m

import query_engine as qe
from citations import (
    build_reference_entries, chapter_and_section, citation_line, extract_source_sentences,
    format_reference_list, page_text,
)
from llm_client import LLMError
from passages import sentences_for, split_text
from prompts import NO_REFERENCE_TEXT, SUMMARY_SYSTEM_PROMPT
from summaries import parse_summary

RIGHT_KNEE_NOTE = """S: 54-year-old reports right knee pain for 3 weeks, worse going up and down stairs and after prolonged sitting with knees bent. Pain rated 6/10 at worst and 2/10 at rest. Reports occasional swelling after long walks and a feeling of the knee giving way on stairs. No history of trauma. Difficulty rising from a low chair.
O: Observation: mild effusion over the right knee, no redness. Gait: antalgic with reduced stance time on the right. Palpation: tenderness along the medial and lateral patellar borders. Range of motion: right knee flexion 0-110 degrees, left 0-135 degrees; extension lacks 5 degrees on the right. Strength: quadriceps 4/5 right, 5/5 left; hip abductors 4/5 right. Patellar tracking: lateral deviation during active extension. Special tests: patellar grind test positive, Lachman test negative, McMurray test negative. Single-leg squat shows dynamic knee valgus.
A: Findings are consistent with patellofemoral pain with quadriceps and hip abductor weakness, reduced knee flexion range, and altered patellar tracking. No signs of ligamentous instability. Functional limitation with stair negotiation and sitting tolerance."""

S1 = "Patellofemoral pain is often worse on stairs."
S2 = "Weakness of the hip abductors is common."
S3 = "The patient may report pain after prolonged sitting with the knees flexed."
S4 = "Check foot posture."
PF_TEXT = " ".join([S1, S2, S3, S4])
OA_TEXT = "The pain is worse on movement and locomotion. She is unable to walk far because of stiffness."
OTHER_TEXT = "Stairs are used in ankle fracture rehabilitation. Swelling after hip surgery is common."


def hit(chunk_id, text, score=0.6, **kw):
    base = {"chunk_id": chunk_id, "score": score, "book_id": "bk", "book_title": "Book A",
            "author": "A. Author", "chapter": "Chapter 21: The Knee", "section": None,
            "subsection": None, "section_path": None, "page_start": "687", "page_end": "758",
            "content": text}
    base.update(kw)
    return base


class FakeLLM:
    """Selection calls get `reply`; summary calls get `summary_reply` (or raise `summary_error`)."""
    model = "fake-llm"

    def __init__(self, reply, summary_reply=None, summary_error=None):
        self.reply, self.summary_reply, self.summary_error = reply, summary_reply, summary_error
        self.calls = []

    def generate(self, **kw):
        self.calls.append(kw)
        if kw.get("system_prompt") == SUMMARY_SYSTEM_PROMPT:
            if self.summary_error:
                raise self.summary_error
            return {"text": self.summary_reply or "", "model": "fake-llm"}
        return {"text": self.reply, "model": "fake-llm"}


def make_engine(results_for, reply, **llm_kw):
    """Engine whose search returns results_for(piece) and whose model returns reply."""
    llm = FakeLLM(reply, **llm_kw)
    engine = qe.PTQueryEngine(client=llm)
    engine.queries = []

    def fake_search(pieces, top_k):
        engine.queries.extend(pieces)
        return [results_for(p) for p in pieces]

    engine._search_pieces = fake_search
    return engine, llm


def quoted(*sentences):
    return "\n".join(f'"{s}"' for s in sentences)


# --------------------------------------------------------------------------
# The 15 required scenarios
# --------------------------------------------------------------------------

class RequiredScenarios(unittest.TestCase):

    def test_01_complete_note_with_relevant_reference(self):
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1, 2")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        self.assertEqual(
            out["references"],
            "1. Book A — A. Author, Chapter: Chapter 21: The Knee, pp. 687–758\n" + quoted(S1, S2))
        self.assertEqual(out["source_indices"], [1])

    def test_02_long_note_information_near_the_end(self):
        filler = " ".join(f"Patient reports general fatigue on day {i}." for i in range(60))
        note = f"S: {filler}\nO: Patellar grind test positive."
        tail = "The patellar grind test is positive in chondromalacia."
        tail_hit = hit("tail", f"{tail} Compress the patella.")
        engine, _ = make_engine(
            lambda p: [tail_hit] if "grind test" in p else [hit("noise", OTHER_TEXT, 0.4)],
            "Source 1: 1")
        out = engine.find_references(note)
        self.assertGreater(len(engine.queries), 3)
        self.assertTrue(all(len(q) <= qe.NOTE_SEGMENT_CHARS for q in engine.queries))
        self.assertTrue(any("grind test" in q for q in engine.queries))   # the end was searched
        self.assertIn(f'"{tail}"', out["references"])

    def test_03_only_subjective(self):
        note = "S: Right knee pain on stairs for 3 weeks. Swelling after long walks."
        self.assertEqual([l for l, _ in qe.parse_soap_sections(note)], ["S"])
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1")
        self.assertIn(f'"{S1}"', engine.find_references(note)["references"])
        self.assertTrue(engine.queries)

    def test_04_only_objective(self):
        note = "Objective: Knee flexion 0-110 degrees. Quadriceps 4/5. Patellar grind test positive."
        self.assertEqual([l for l, _ in qe.parse_soap_sections(note)], ["O"])
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 2")
        self.assertIn(f'"{S2}"', engine.find_references(note)["references"])

    def test_05_only_assessment(self):
        note = "A: Findings consistent with patellofemoral pain with hip abductor weakness."
        self.assertEqual([l for l, _ in qe.parse_soap_sections(note)], ["A"])
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1")
        self.assertIn(f'"{S1}"', engine.find_references(note)["references"])

    def test_06_duplicate_passages_from_multiple_queries(self):
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1, 2")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        self.assertGreater(len(engine.queries), 2)                 # retrieved by every piece
        self.assertEqual(out["retrieved_count"], 1)
        self.assertEqual(out["references"].count(S1), 1)

    def test_06b_overlapping_chunks_do_not_repeat_a_sentence(self):
        a = hit("bk_00001", f"Intro text here. {S2}")
        b = hit("bk_00002", f"{S2} Another point follows.")
        sents = [[s for s in sentences_for(h["content"])] for h in (a, b)]
        entries = build_reference_entries([a, b], sents, {1: [1, 2], 2: [1, 2]})
        text = format_reference_list(entries)
        self.assertEqual(text.count(S2), 1)
        self.assertIn('"Another point follows."', text)

    def test_07_similar_terms_but_unrelated_passage_is_left_out(self):
        engine, _ = make_engine(
            lambda p: [hit("a1", PF_TEXT), hit("z9", OTHER_TEXT, book_id="zz", book_title="Ankle Book")],
            "Source 1: 1")      # mocked model judges Source 2 unrelated
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        self.assertNotIn("Ankle Book", ref)
        self.assertNotIn("ankle fracture", ref)

    def test_08_invalid_source_ids(self):
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)],
                                "Source 99: 1\nSource 0: 2\nSource 1: 1\nsource 1: 3")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        self.assertEqual(out["source_indices"], [1])
        self.assertEqual(out["references"].count('"'), 2)           # one quoted sentence
        self.assertNotIn(S3, out["references"])                      # 'source 1: 3' ignored: first label wins

    def test_09_invalid_sentence_numbers(self):
        engine, _ = make_engine(
            lambda p: [hit("a1", PF_TEXT), hit("b2", OA_TEXT, book_id="oa", book_title="OA Book")],
            "Source 1: 99, 0, 2\nSource 2: 77")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        self.assertIn(f'"{S2}"', out["references"])
        self.assertNotIn(S1, out["references"])
        self.assertNotIn("OA Book", out["references"])               # no valid sentence -> nothing shown
        self.assertEqual(out["dropped_sources"], [2])

    def test_10_missing_author_chapter_and_page_metadata(self):
        h = hit("a1", PF_TEXT, author=None, page_start=None, page_end=None, chapter=None)
        engine, _ = make_engine(lambda p: [h], "Source 1: 1")
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        self.assertEqual(ref, "1. Book A\n" + quoted(S1))            # nothing invented, no placeholders
        self.assertNotIn("None", ref)
        self.assertNotIn("not recorded", ref)

    def test_11_headers_and_footers_inside_text(self):
        text = ("Do not sit with the knees flexed for long periods. Muscles can control 712 PATELLOFEMORAL "
                "DYSFUNCTION: NONOPERATIVE MANAGEMENT knee function without symptoms. "
                "Next point. NONOPERATIVE MANAGEMENT PATELLOFEMORAL 713 A 12 ACL PCL tears were noted.")
        sents = sentences_for(text)
        complete = [s.text for s in sents if s.complete]
        self.assertEqual(complete, ["Do not sit with the knees flexed for long periods.", "Next point.",
                                    "A 12 ACL PCL tears were noted."])      # real "12 ACL PCL" text kept
        self.assertNotIn("PATELLOFEMORAL", " ".join(s.text for s in sents))   # furniture never in a sentence
        cut = [s.text for s in sents if not s.complete]
        self.assertEqual(cut, ["Muscles can control", "knee function without symptoms."])   # halves of the broken sentence

    def test_12_broken_words_stay_verbatim_and_incomplete_sentences_are_not_offered(self):
        text = ("symptoms. Recurrent swelling may relate to patel lofemoral dysfunction. "
                "Fluid extrava sates here. Another that stops mid")
        sents = sentences_for(text)
        self.assertEqual([(s.cut_start, s.cut_end) for s in sents],
                         [(True, False), (False, False), (False, False), (False, True)])
        engine, llm = make_engine(lambda p: [hit("a1", text)], "Source 1: 1, 2")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        prompt = llm.calls[0]["user_prompt"]
        self.assertNotIn("symptoms.", prompt)                          # cut at the start: not offered
        self.assertNotIn("stops mid", prompt)                          # cut at the end: not offered
        self.assertEqual(out["references"].split("\n")[1:],
                         ['"Recurrent swelling may relate to patel lofemoral dysfunction."',
                          '"Fluid extrava sates here."'])               # the book's own spelling, not rewritten

    def test_13_multiple_passages_same_chapter(self):
        kw = dict(section_path="Ch 5 > Knee > Gait", page_start="10", page_end="20")
        a = hit("bk_00002", "Second passage sentence here. Extra.", **kw)
        b = hit("bk_00001", "First passage sentence here. Extra.", **kw)
        c = hit("bk_00009", "Other section sentence here. Extra.", section_path="Ch 5 > Hip > Gait",
                page_start="30", page_end="40")
        d = hit("bk_00005", "Same section later pages here. Extra.", page_start="50", page_end="60",
                section_path="Ch 5 > Knee > Gait")
        sents = [sentences_for(h["content"]) for h in (a, b, c, d)]
        entries = build_reference_entries([a, b, c, d], sents, {1: [1], 2: [1], 3: [1], 4: [1]})
        self.assertEqual([e["sentences"] for e in entries],
                         [["First passage sentence here.", "Second passage sentence here."],
                          ["Other section sentence here."],
                          ["Same section later pages here."]])        # a change of section or pages = new citation
        self.assertEqual([e["pages"] for e in entries], ["pp. 10–20", "pp. 30–40", "pp. 50–60"])

    def test_14_no_relevant_references(self):
        engine, llm = make_engine(lambda p: [hit("a1", PF_TEXT)], NO_REFERENCE_TEXT)
        out = engine.find_references(RIGHT_KNEE_NOTE)
        self.assertEqual(out["references"], NO_REFERENCE_TEXT)
        self.assertEqual(out["reference_entries"], [])
        engine2, llm2 = make_engine(lambda p: [], "unused")            # nothing retrieved: model not called
        self.assertEqual(engine2.find_references(RIGHT_KNEE_NOTE)["references"], NO_REFERENCE_TEXT)
        self.assertEqual(llm2.calls, [])
        self.assertEqual(engine2.find_references("   ")["references"], NO_REFERENCE_TEXT)

    def test_15_repeated_input_gives_identical_output(self):
        def results(p):
            return [hit("b", OA_TEXT, 0.5, book_id="oa"), hit("a", PF_TEXT, 0.5)]   # equal scores
        runs = []
        for _ in range(3):
            engine, _ = make_engine(results, "Source 1: 1\nSource 2: 1")
            out = engine.find_references(RIGHT_KNEE_NOTE)
            runs.append((out["references"], out["candidates"], out["source_indices"]))
        self.assertEqual(runs[0], runs[1])
        self.assertEqual(runs[1], runs[2])


# --------------------------------------------------------------------------
# Output format (academic-style citations)
# --------------------------------------------------------------------------

class OutputFormat(unittest.TestCase):

    def test_citation_line_has_only_known_parts(self):
        full = {"book": "T", "author": "A. Au", "chapter": "Ch 1", "section": "Gait", "pages": "pp. 1–2"}
        self.assertEqual(citation_line(1, full), "1. T — A. Au, Chapter: Ch 1, Section: Gait, pp. 1–2")
        self.assertEqual(citation_line(2, {**full, "author": None, "section": None}),
                         "2. T — Chapter: Ch 1, pp. 1–2")
        self.assertEqual(citation_line(3, {**full, "chapter": None, "section": None, "pages": "p. 7"}),
                         "3. T — A. Au, p. 7")
        self.assertEqual(citation_line(4, {"book": "T", "author": None, "chapter": None,
                                           "section": None, "pages": None}), "4. T")

    def test_only_citation_lines_and_quoted_sentences_with_no_omission_markers(self):
        engine, _ = make_engine(
            lambda p: [hit("a1", PF_TEXT), hit("b2", OA_TEXT, book_id="oa", book_title="OA Book", author=None,
                                              page_start="186", page_end="186", chapter="Chapter 7")],
            "Source 1: 1, 3\nSource 2: 2")
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        lines = ref.split("\n")
        self.assertEqual(lines, [
            "1. Book A — A. Author, Chapter: Chapter 21: The Knee, pp. 687–758", f'"{S1}"', f'"{S3}"',
            "2. OA Book — Chapter: Chapter 7, p. 186", '"She is unable to walk far because of stiffness."'])
        for banned in ("[...]", "...", "…", "Excerpt", "Summary", "Author:", "Pages:", "Subjective"):
            self.assertNotIn(banned, ref)
        for line in lines:
            self.assertTrue(re.match(r"^\d+\. ", line) or (line.startswith('"') and line.endswith('"')), line)

    def test_cut_sentences_are_never_printed(self):
        text = "starts mid thought here. A complete sentence stands here. Ends before the next chunk"
        engine, llm = make_engine(lambda p: [hit("a1", text)], "Source 1: 1, 2, 3")
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        self.assertEqual(ref.split("\n")[1:], ['"A complete sentence stands here."'])
        self.assertNotIn("starts mid thought", llm.calls[0]["user_prompt"])
        # a passage with nothing complete is dropped instead of printed
        engine2, llm2 = make_engine(lambda p: [hit("a1", "only a fragment that never ends")], "Source 1: 1")
        self.assertEqual(engine2.find_references(RIGHT_KNEE_NOTE)["references"], NO_REFERENCE_TEXT)
        self.assertEqual(llm2.calls, [])

    def test_page_text(self):
        self.assertEqual(page_text(hit("a", "x", page_start="186", page_end="186")), "p. 186")
        self.assertEqual(page_text(hit("a", "x", page_start="687", page_end="758")), "pp. 687–758")
        self.assertEqual(page_text(hit("a", "x", page_start="7", page_end=None)), "p. 7")
        self.assertIsNone(page_text(hit("a", "x", page_start=None, page_end=None)))


# --------------------------------------------------------------------------
# Detailed summary (the only model-written text, held to the passage)
# --------------------------------------------------------------------------

GOOD_SUMMARY = ("The passage links patellofemoral pain to stair use and notes that weakness of the hip "
                "abductor muscles is common in these patients.")
HEAD = "1. Book A — A. Author, Chapter: Chapter 21: The Knee, pp. 687–758"


def run_with_summary(summary_reply, text=PF_TEXT, **llm_kw):
    engine, llm = make_engine(lambda p: [hit("a1", text)], "Source 1: 1, 2",
                              summary_reply=summary_reply, **llm_kw)
    return engine.find_references(RIGHT_KNEE_NOTE, summaries=True), llm


class DetailedSummary(unittest.TestCase):

    def test_expected_layout(self):
        out, llm = run_with_summary(f"{GOOD_SUMMARY} [1, 2]")
        self.assertEqual(out["references"], "\n".join([
            HEAD, "Original Sentences:", f'"{S1}"', f'"{S2}"', "Detailed Summary:", GOOD_SUMMARY]))
        self.assertEqual(out["reference_entries"][0]["summary"], [GOOD_SUMMARY])
        self.assertEqual(out["summary_notes"], [])
        # the model saw the COMPLETE passage (all four sentences, not only the two quoted)
        summary_call = llm.calls[1]
        self.assertEqual(summary_call["system_prompt"], SUMMARY_SYSTEM_PROMPT)
        for n, s in enumerate((S1, S2, S3, S4), 1):
            self.assertIn(f"({n}) {s}", summary_call["user_prompt"])
        self.assertEqual(summary_call["temperature"], 0.0)

    def test_summary_is_off_by_default_in_the_engine(self):
        engine, llm = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1, 2", summary_reply="x")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        self.assertEqual(len(llm.calls), 1)                       # selection only
        self.assertNotIn("Detailed Summary", out["references"])
        self.assertNotIn("Original Sentences", out["references"])

    def test_insufficient_passage_gets_no_summary_section(self):
        out, _ = run_with_summary("INSUFFICIENT")
        self.assertEqual(out["references"], "\n".join([HEAD, "Original Sentences:", f'"{S1}"', f'"{S2}"']))
        self.assertEqual(out["reference_entries"][0]["summary"], [])
        self.assertEqual(out["summary_notes"],
                         [{"citation": 1, "paragraph": "", "reason": "the model said the passage is insufficient"}])
        self.assertEqual(out["reference_entries"][0]["summary_raw"], "INSUFFICIENT")

    def test_empty_summary_reply_is_explained_and_the_raw_reply_is_kept(self):
        out, _ = run_with_summary("")
        self.assertEqual(out["summary_notes"][0]["reason"], "empty reply from the model")
        out, _ = run_with_summary(f"{GOOD_SUMMARY}")                    # model forgot the evidence numbers
        self.assertEqual(out["summary_notes"][0]["reason"], "no sentence numbers at the end")
        self.assertEqual(out["reference_entries"][0]["summary_raw"], GOOD_SUMMARY)

    def test_paragraph_without_valid_evidence_is_dropped(self):
        out, _ = run_with_summary(f"{GOOD_SUMMARY}\n{GOOD_SUMMARY} [99]\n{GOOD_SUMMARY} [0]")
        self.assertNotIn("Detailed Summary", out["references"])
        self.assertEqual([n["reason"] for n in out["summary_notes"]],
                         ["no sentence numbers at the end", "no valid sentence numbers", "no valid sentence numbers"])

    def test_invented_number_or_name_is_dropped(self):
        for bad, reason in (
            ("The passage says pain rises by 40 percent on stairs in these patients [1]", "number not in the passage: 40"),
            ("The passage ties patellofemoral pain on stairs to the work of Virchow in these patients [1]",
             "name or term not in the passage: Virchow"),
            ("The passage ties patellofemoral pain on stairs to the ACL in these patients [1]",
             "name or term not in the passage: ACL"),
        ):
            out, _ = run_with_summary(bad)
            self.assertNotIn("Detailed Summary", out["references"], bad)
            self.assertEqual(out["summary_notes"][0]["reason"], reason)

    def test_copied_sentence_and_unsupported_topic_are_dropped(self):
        copy = f"{S1} This matches what clinicians usually see in practice [1]"
        off_topic = "The author describes how regular swimming improves cardiovascular fitness and endurance [1]"
        out, _ = run_with_summary(f"{copy}\n{off_topic}")
        self.assertNotIn("Detailed Summary", out["references"])
        self.assertEqual([n["reason"] for n in out["summary_notes"]],
                         ["copies an original sentence", "not supported by the sentences it cites"])

    def test_only_supported_paragraphs_are_printed_and_tags_are_removed(self):
        bad = "The passage says pain rises by 40 percent on stairs in these patients [1]"
        out, _ = run_with_summary(f"{bad}\n- {GOOD_SUMMARY} [S1-S2]")
        self.assertEqual(out["reference_entries"][0]["summary"], [GOOD_SUMMARY])
        self.assertNotIn("[", out["references"])
        self.assertNotIn("40", out["references"])

    def test_failed_summary_call_still_prints_the_citation(self):
        out, _ = run_with_summary(None, summary_error=LLMError("server down"))
        self.assertEqual(out["references"], "\n".join([HEAD, "Original Sentences:", f'"{S1}"', f'"{S2}"']))
        self.assertEqual(out["summary_notes"][0]["reason"], "summary call failed: LLMError")

    def test_one_summary_covers_every_passage_of_a_citation(self):
        a = hit("bk_00001", PF_TEXT, section_path="Ch 5 > Knee", page_start="10", page_end="20")
        b = hit("bk_00002", "Rest helps in the acute stage. Ice may ease swelling.", section_path="Ch 5 > Knee",
                page_start="10", page_end="20")
        engine, llm = make_engine(lambda p: [a, b], "Source 1: 1\nSource 2: 1",
                                  summary_reply="The text lists rest in the acute stage as one point [5]")
        out = engine.find_references(RIGHT_KNEE_NOTE, summaries=True)
        self.assertEqual(len(out["reference_entries"]), 1)         # same book, chapter, section, pages
        self.assertEqual(out["reference_entries"][0]["summary"],
                         ["The text lists rest in the acute stage as one point"])   # evidence [5] is in the 2nd passage
        prompt = llm.calls[1]["user_prompt"]
        self.assertIn("(4) Check foot posture.", prompt)
        self.assertIn("(5) Rest helps in the acute stage.", prompt)  # numbering continues across passages
        self.assertEqual(len(llm.calls), 2)                        # one summary call per citation

    def test_parse_summary_evidence_formats(self):
        sents = [S1, S2, S3, S4]
        for tag in ("[1, 2]", "[S1-S2]", "[1-2]", "[s2]"):
            kept, rejected = parse_summary(f"{GOOD_SUMMARY} {tag}", sents)
            self.assertEqual((kept, rejected), ([GOOD_SUMMARY], []), tag)
        self.assertEqual(parse_summary("", sents)[0], [])
        self.assertEqual(parse_summary("insufficient information here", sents)[0], [])


# --------------------------------------------------------------------------
# Noise filters: questions, glued headings, run-ons, and the per-source cap
# --------------------------------------------------------------------------

# Sentences from a real run on the right-knee note (book text only, no patient data).
RUN_ON = ("Pain or poor knee control when descending or ascending stairs Pain with walking, jumping, or running "
          "interfering during ADL, instrumental ADL (IADL), work, and community, recreational, or sport activities "
          "Pain and stiffness with prolonged flexed knee postures, such as sitting or squatting Patellofemoral "
          "Symptoms: Management—Protection Phase When symptoms are acute, treat them as any acute joint problem—with "
          "modalities, rest, gentle motion, and muscle-setting exercises in pain-free positions.")
GLUED_HEADING = ("CHAPTER FIVE The Knee 241 Patellofemoral Disorders REHABILITATION RATIONALE Patellofemoral "
                 "dysfunction, or disorder, may be defined as pain, imbalance, inflammation, and/or instability of "
                 "any component of the extensor mechanism of the knee.")
QUESTIONS = ["Is the pain worse in the morning when the patient gets out of bed and does it abate during the day, "
             "or is it worse after prolonged ambulation or is it weather related?",
             "Does the knee pain affect activities of daily living and the quality of life?",
             "On a scale of a to 10, with the worst pain they have ever experienced being a 10, what does their "
             "daily pain rate?"]
KEPT = ["Pain and joint effusion inhibit the quadriceps, so it is imperative to reduce irritating forces.",
        "These include treatment modalities that have im proved the pain (e.g., physical therapy, cortisone in "
        "jections, medications, and splinting or bracing) as well as what makes the symptoms worse (e.g., walk "
        "ing, running, athletic activity, stair climbing, or job activities).",
        "PFJRFs (Fig. 4-44) increase with flexion of the knee from 0.5 times body weight during level walking to "
        "3 to 4 times body weight during stairclimbing to 7 to 8 times body weight with squatting.",
        "Etiology of Symptoms The cause of anterior knee pain may be direct trauma, overuse, faulty patellar "
        "tracking from malalignment due to anatomical variations or soft tissue length and strength imbalances, "
        "degeneration, or a combination of these factors.",
        "However, critical review of available studies found no evidence that Q-angle measures correlated with "
        "the presence or severity of anterior knee pain."]


def usable(text):
    return all(s.usable for s in sentences_for(text))


class NoiseFilters(unittest.TestCase):

    def test_questions_are_not_usable_or_offered(self):
        for q in QUESTIONS:
            self.assertFalse(usable(q), q)
        text = f"{QUESTIONS[1]} {S2} What is the pain rating?"
        engine, llm = make_engine(lambda p: [hit("a1", text)], "Source 1: 1")
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        self.assertEqual(ref.split("\n")[1:], [f'"{S2}"'])
        self.assertNotIn("quality of life", llm.calls[0]["user_prompt"])
        self.assertNotIn("pain rating", llm.calls[0]["user_prompt"])

    def test_glued_chapter_heading_is_not_usable(self):
        self.assertFalse(usable(GLUED_HEADING))
        self.assertFalse(usable("The ANTERIOR CRUCIATE LIGAMENT is a major stabiliser."))
        self.assertTrue(usable("The ACL and PCL are major stabilisers of the knee joint."))   # short acronyms are fine
        self.assertTrue(usable("Pain was rated on the VAS and the ADL scale during the first week."))

    def test_run_on_sentences_are_not_usable(self):
        self.assertFalse(usable(RUN_ON))
        self.assertFalse(usable("Normal words flow on here. " * 1 + "word " * 90 + "end."))      # over 400 characters
        self.assertTrue(usable("The patient reports pain on stairs and after sitting " + "with the knees bent " * 2 + "."))

    def test_ordinary_long_and_proper_noun_sentences_are_kept(self):
        for s in KEPT:
            self.assertTrue(usable(s), s)

    def test_real_run_filters_out_exactly_the_noise(self):
        text = " ".join([RUN_ON, GLUED_HEADING, *QUESTIONS, *KEPT])
        offered = [s.text for s in sentences_for(text) if s.usable]
        self.assertEqual(offered, KEPT)

    def test_at_most_three_sentences_per_source_in_the_models_own_order(self):
        self.assertEqual(extract_source_sentences("Source 1: 4, 3, 2, 1", [5], max_per_source=3), {1: [2, 3, 4]})
        self.assertEqual(extract_source_sentences("Source 1: 2-5, 1", [9], max_per_source=3), {1: [2, 3, 4]})
        self.assertEqual(extract_source_sentences("Source 1: 4, 3, 2, 1", [5]), {1: [1, 2, 3, 4]})   # no cap by default
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 4, 3, 2, 1")
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        self.assertEqual(ref.split("\n")[1:], [f'"{S2}"', f'"{S3}"', f'"{S4}"'])      # most relevant three, book order
        self.assertNotIn(S1, ref)

    def test_prompt_tells_the_model_what_to_skip(self):
        engine, llm = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1")
        engine.find_references(RIGHT_KNEE_NOTE)
        prompt = llm.calls[0]["user_prompt"]
        for phrase in ("at most 3 sentences per source", "advice on treatment or management",
                       "not tied to a finding", "only makes sense after the sentence before it"):
            self.assertIn(phrase, prompt)


# --------------------------------------------------------------------------
# Supporting behaviour
# --------------------------------------------------------------------------

class SupportingBehaviour(unittest.TestCase):

    def test_model_text_is_never_printed(self):
        reply = "Source 1: 2. Patellofemoral pain is caused by weak quadriceps (outside fact)."
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], reply)
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        self.assertNotIn("caused by weak quadriceps", ref)
        self.assertIn(f'"{S2}"', ref)

    def test_ranges_in_the_model_reply(self):
        self.assertEqual(extract_source_sentences("Source 1: 2-4, 7", [9]), {1: [2, 3, 4, 7]})
        self.assertEqual(extract_source_sentences("Source 1: 5-2", [9]), {1: []})
        self.assertEqual(extract_source_sentences("Source 1: none", [3]), {1: []})

    def test_fusion_prefers_passages_found_by_several_parts_of_the_note(self):
        results = [[hit("one", PF_TEXT, 0.50)], [hit("one", PF_TEXT, 0.45), hit("solo", OA_TEXT, 0.80)],
                   [hit("one", PF_TEXT, 0.40)]]
        ranked = qe.fuse_ranked_hits(results, [1 / 3] * 3)
        self.assertEqual([h["chunk_id"] for h in ranked], ["one", "solo"])
        self.assertEqual(ranked[0]["score"], 0.50)                     # best cosine kept

    def test_each_section_has_equal_weight(self):
        sections = [("S", "Short subjective."), ("O", ". ".join(f"Finding number {i}" for i in range(40)) + ".")]
        pieces, weights = qe.build_queries(sections)
        self.assertAlmostEqual(sum(weights), 2.0)
        self.assertGreater(len(pieces), 2)

    def test_plan_section_is_not_searched_or_sent(self):
        note = "S: Knee pain on stairs.\nO: Flexion 110 degrees.\nA: Patellofemoral pain.\nP: Start quadriceps strengthening."
        engine, llm = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1")
        engine.find_references(note)
        self.assertFalse(any("strengthening" in q for q in engine.queries))
        self.assertNotIn("strengthening", llm.calls[0]["user_prompt"])

    def test_note_pieces_do_not_cut_words_and_overlap(self):
        text = ("Pain on stairs. " * 5) + "Palpation: tenderness along the medial and lateral patellar borders; " * 12
        pieces = qe.split_note(text, 200)
        self.assertTrue(all(len(p) <= 200 for p in pieces))
        words = set(text.replace(";", " ").replace(":", " ").replace(".", " ").split())
        self.assertTrue(all(w.strip(";:.") in words for p in pieces for w in p.split()))
        self.assertTrue(any(p.split(". ")[-1] in pieces[i + 1] for i, p in enumerate(pieces[:-1])))   # overlap

    def test_one_very_long_word_run_is_split_at_spaces(self):
        pieces = qe.split_note(" ".join(["knee"] * 400), 100)
        self.assertTrue(all(len(p) <= 100 and set(p.split()) == {"knee"} for p in pieces))

    def test_sentence_splitter_abbreviations_decimals_and_references(self):
        text = ("Load rises from 0.5 times body weight to 3 times (Fig. 5-29), as shown by Smith et al. in 1999. "
                "Next sentence here.27,47,135 Third sentence is here. Dr. Jones agreed, e.g. at 30 degrees. Last one.")
        got = split_text(text)
        self.assertEqual(len(got), 5)
        self.assertTrue(got[0].endswith("in 1999."))
        self.assertTrue(got[1].endswith("here.27,47,135"))
        self.assertTrue(got[3].startswith("Dr. Jones"))

    def test_reference_numbers_are_removed_but_real_numbers_are_kept(self):
        text = ("The cause may be trauma or overuse.27,47,100,116,135,189,201,215, 242,256 Pain and joint "
                "effusion inhibit the quadriceps,231 so reduce irritating forces. Load rises from 0.5 times "
                "to 3 times body weight (Fig. 4-44), and the angle was 30.5 degrees at Level 3.2 "
                "(Figure 12-5).31 The deep bursa is a source of pain.107")
        got = [s.text for s in sentences_for(text)]
        self.assertEqual(got, [
            "The cause may be trauma or overuse.",
            "Pain and joint effusion inhibit the quadriceps, so reduce irritating forces.",
            "Load rises from 0.5 times to 3 times body weight (Fig. 4-44), and the angle was 30.5 degrees "
            "at Level 3.2 (Figure 12-5).",
            "The deep bursa is a source of pain."])

    def test_printed_sentences_have_no_reference_numbers(self):
        text = "An attempt should be made to find the cause.27,47,135 Splinting may unload the joint.107"
        engine, _ = make_engine(lambda p: [hit("a1", text)], "Source 1: 1, 2")
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        self.assertEqual(ref.split("\n")[1:], ['"An attempt should be made to find the cause."',
                                               '"Splinting may unload the joint."'])

    def test_chapter_and_section_from_stored_outline(self):
        kisner = hit("k", "x", chapter="Therapeutic Exercise: Foundations and Techniques, Fifth Edition",
                     section_path="Therapeutic Exercise: Foundations and Techniques, Fifth Edition > Part IV: "
                                  "Exercise Interventions by Body Region > Chapter 21: The Knee")
        self.assertEqual(chapter_and_section(kisner, "Therapeutic Exercise"), ("Chapter 21: The Knee", None))
        magee = hit("m", "x", section_path="12 Knee > Examination > McConnell Test.")
        self.assertEqual(chapter_and_section(magee, "Magee"), ("12 Knee", "Examination > McConnell Test."))
        scan = hit("h", "x", chapter="06", section="scan0023", section_path=None)
        self.assertEqual(chapter_and_section(scan, "Handbook"), ("06", None))     # scanner junk not shown
        root = hit("r", "x", chapter="Case Studies", section="None", section_path=None)
        self.assertEqual(chapter_and_section(root, "Case Studies"), (None, None))

    def test_json_result_keeps_existing_keys(self):
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        for key in ("references", "source_indices", "retrieved_count", "raw_answer",
                    "model", "retrieval_ms", "total_ms"):
            self.assertIn(key, out)


# --------------------------------------------------------------------------
# Live check: real Qdrant + embedding model + LLM server
# --------------------------------------------------------------------------

@unittest.skipUnless(os.environ.get("PT_RAG_LIVE"), "set PT_RAG_LIVE=1 to run against the real services")
class LiveModelCheck(unittest.TestCase):
    """Run on the server. It prints the result for a person to judge relevance."""

    def test_right_knee_note(self):
        out = qe.PTQueryEngine().find_references(RIGHT_KNEE_NOTE)
        print("\n" + out["references"])
        print("\nraw model reply:\n" + out["raw_answer"])
        print("\ncandidates:", *[str(c) for c in out["candidates"]], sep="\n  ")
        # the model must answer in the 'Source N: numbers' format or say there is no match
        self.assertTrue(out["source_indices"] or NO_REFERENCE_TEXT in out["raw_answer"]
                        or out["retrieved_count"] == 0, "model did not follow the Source N format")
        for banned in ("[...]", "…", "Excerpt"):
            self.assertNotIn(banned, out["references"])
        # soft check for a human: the osteoarthrosis / left-knee case should not be listed
        if "Rheumatic conditions" in out["references"] or "OSTEOARTHROSIS" in out["references"]:
            print("WARNING: the osteoarthrosis passage is still listed")


if __name__ == "__main__":
    unittest.main()
