"""
Tests for SOAP reference mode (ask.py --soap-file).

Run:   python3 -m unittest test_references -v
Live:  PT_RAG_LIVE=1 python3 -m unittest test_references.LiveModelCheck -v

Unit tests use a MOCKED model and MOCKED search results. They prove the
plumbing: query building, ranking, validation of ids, grouping, de-duplication,
text cleanup and printing. They do NOT prove that the real model picks the
right sentences or that the real Qdrant data ranks well; only the live check
(which needs Qdrant, the embedding model and the LLM server) can show that.
"""

import os
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
    build_reference_entries, chapter_and_section, extract_source_sentences,
    format_reference_list, page_text,
)
from passages import build_vocab, repair_split_words, sentences_for, split_text
from prompts import NO_REFERENCE_TEXT

RIGHT_KNEE_NOTE = """S: 54-year-old reports right knee pain for 3 weeks, worse going up and down stairs and after prolonged sitting with knees bent. Pain rated 6/10 at worst and 2/10 at rest. Reports occasional swelling after long walks and a feeling of the knee giving way on stairs. No history of trauma. Difficulty rising from a low chair.
O: Observation: mild effusion over the right knee, no redness. Gait: antalgic with reduced stance time on the right. Palpation: tenderness along the medial and lateral patellar borders. Range of motion: right knee flexion 0-110 degrees, left 0-135 degrees; extension lacks 5 degrees on the right. Strength: quadriceps 4/5 right, 5/5 left; hip abductors 4/5 right. Patellar tracking: lateral deviation during active extension. Special tests: patellar grind test positive, Lachman test negative, McMurray test negative. Single-leg squat shows dynamic knee valgus.
A: Findings are consistent with patellofemoral pain with quadriceps and hip abductor weakness, reduced knee flexion range, and altered patellar tracking. No signs of ligamentous instability. Functional limitation with stair negotiation and sitting tolerance."""

PF_TEXT = ("Patellofemoral pain is often worse on stairs. Weakness of the hip abductors is common. "
           "The patient may report pain after prolonged sitting with the knees flexed. Check foot posture.")
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
    model = "fake-llm"

    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def generate(self, **kw):
        self.calls.append(kw)
        return {"text": self.reply, "model": "fake-llm"}


def make_engine(results_for, reply):
    """Engine whose search returns results_for(piece) and whose model returns reply."""
    llm = FakeLLM(reply)
    engine = qe.PTQueryEngine(client=llm)
    engine.queries = []

    def fake_search(pieces, top_k):
        engine.queries.extend(pieces)
        return [results_for(p) for p in pieces]

    engine._search_pieces = fake_search
    return engine, llm


# --------------------------------------------------------------------------
# The 15 required scenarios
# --------------------------------------------------------------------------

class RequiredScenarios(unittest.TestCase):

    def test_01_complete_note_with_relevant_reference(self):
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1, 2")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        ref = out["references"]
        self.assertIn("1. Book A", ref)
        self.assertIn("Author: A. Author", ref)
        self.assertIn("Chapter: Chapter 21: The Knee", ref)
        self.assertIn("Pages: 687–758 (section range)", ref)
        self.assertIn("Excerpt: Patellofemoral pain is often worse on stairs. "
                      "Weakness of the hip abductors is common.", ref)
        for banned in ("Subjective", "Objective", "Assessment", "diagnos", "treatment"):
            self.assertNotIn(banned, ref)
        self.assertEqual(out["source_indices"], [1])

    def test_02_long_note_information_near_the_end(self):
        filler = " ".join(f"Patient reports general fatigue on day {i}." for i in range(60))
        note = f"S: {filler}\nO: Patellar grind test positive."
        tail_hit = hit("tail", "The patellar grind test is positive in chondromalacia. Compress the patella.")
        engine, _ = make_engine(
            lambda p: [tail_hit] if "grind test" in p else [hit("noise", OTHER_TEXT, 0.4)],
            "Source 1: 1")
        out = engine.find_references(note)
        self.assertGreater(len(engine.queries), 3)
        self.assertTrue(all(len(q) <= qe.NOTE_SEGMENT_CHARS for q in engine.queries))
        self.assertTrue(any("grind test" in q for q in engine.queries))   # the end was searched
        self.assertIn("The patellar grind test is positive in chondromalacia.", out["references"])

    def test_03_only_subjective(self):
        note = "S: Right knee pain on stairs for 3 weeks. Swelling after long walks."
        self.assertEqual([l for l, _ in qe.parse_soap_sections(note)], ["S"])
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1")
        self.assertIn("Excerpt:", engine.find_references(note)["references"])
        self.assertTrue(engine.queries)

    def test_04_only_objective(self):
        note = "Objective: Knee flexion 0-110 degrees. Quadriceps 4/5. Patellar grind test positive."
        self.assertEqual([l for l, _ in qe.parse_soap_sections(note)], ["O"])
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 2")
        self.assertIn("Weakness of the hip abductors is common.", engine.find_references(note)["references"])

    def test_05_only_assessment(self):
        note = "A: Findings consistent with patellofemoral pain with hip abductor weakness."
        self.assertEqual([l for l, _ in qe.parse_soap_sections(note)], ["A"])
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1")
        self.assertIn("Excerpt:", engine.find_references(note)["references"])

    def test_06_duplicate_passages_from_multiple_queries(self):
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], "Source 1: 1, 2")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        self.assertGreater(len(engine.queries), 2)                 # retrieved by every piece
        self.assertEqual(out["retrieved_count"], 1)
        self.assertEqual(out["references"].count("Patellofemoral pain is often worse on stairs."), 1)

    def test_06b_overlapping_chunks_do_not_repeat_a_sentence(self):
        a = hit("bk_00001", "Intro text here. Weakness of the hip abductors is common.")
        b = hit("bk_00002", "Weakness of the hip abductors is common. Another point follows.")
        sents = [sentences_for(a["content"]), sentences_for(b["content"])]
        entries = build_reference_entries([a, b], sents, {1: [1, 2], 2: [1, 2]})
        text = format_reference_list(entries)
        self.assertEqual(text.count("Weakness of the hip abductors is common."), 1)
        self.assertIn("Another point follows.", text)

    def test_07_similar_terms_but_unrelated_passage_is_left_out(self):
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT), hit("z9", OTHER_TEXT, book_id="zz",
                                                                 book_title="Ankle Book")],
                                "Source 1: 1")      # mocked model judges Source 2 unrelated
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        self.assertNotIn("Ankle Book", ref)
        self.assertNotIn("ankle fracture", ref)

    def test_08_invalid_source_ids(self):
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)],
                                "Source 99: 1\nSource 0: 2\nSource 1: 1\nsource 1: 3")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        self.assertEqual(out["source_indices"], [1])
        self.assertEqual(out["references"].count("Excerpt:"), 1)
        self.assertNotIn("Check foot posture", out["references"])   # 'source 1: 3' ignored: first label wins

    def test_09_invalid_sentence_numbers(self):
        engine, _ = make_engine(
            lambda p: [hit("a1", PF_TEXT), hit("b2", OA_TEXT, book_id="oa", book_title="OA Book")],
            "Source 1: 99, 0, 2\nSource 2: 77")
        out = engine.find_references(RIGHT_KNEE_NOTE)
        self.assertIn("Weakness of the hip abductors is common.", out["references"])
        self.assertNotIn("Patellofemoral pain is often worse", out["references"])
        self.assertNotIn("OA Book", out["references"])               # no valid sentence -> nothing shown
        self.assertEqual(out["dropped_sources"], [2])

    def test_10_missing_author_and_page_metadata(self):
        h = hit("a1", PF_TEXT, author=None, page_start=None, page_end=None, chapter=None)
        engine, _ = make_engine(lambda p: [h], "Source 1: 1")
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        self.assertNotIn("Author:", ref)
        self.assertNotIn("Chapter:", ref)
        self.assertNotIn("None", ref)
        self.assertIn("Pages: not recorded", ref)

    def test_11_headers_and_footers_inside_text(self):
        text = ("Do not sit with the knees flexed for long periods. Muscles can control 712 PATELLOFEMORAL "
                "DYSFUNCTION: NONOPERATIVE MANAGEMENT knee function without symptoms. "
                "Next point. NONOPERATIVE MANAGEMENT PATELLOFEMORAL 713 A 12 ACL PCL tears were noted.")
        got = [s.text for s in sentences_for(text)]
        joined = " ".join(got)
        self.assertNotIn("PATELLOFEMORAL", joined)
        self.assertNotIn("713", joined)
        self.assertIn("Muscles can control knee function without symptoms.", got)   # rejoined across the break
        self.assertIn("12 ACL PCL tears were noted.", joined)                       # real text kept

    def test_12_broken_words_and_incomplete_sentences(self):
        vocab = build_vocab(["patellofemoral pain patellofemoral joint patellofemoral dysfunction",
                             "in the knee to the hip", "swell ing here"])
        fixed = repair_split_words("Recurrent swelling may relate to patel lofemoral dysfunction, in to the knee.", vocab)
        self.assertIn("patellofemoral dysfunction", fixed)
        self.assertIn("in to the knee", fixed)                         # two real words: not joined
        self.assertIn("extrava sates", repair_split_words("fluid extrava sates here", vocab))  # never guessed
        sents = sentences_for("symptoms. Complete sentence here. Another that stops mid")
        self.assertTrue(sents[0].cut_start)                           # starts mid-thought
        self.assertFalse(sents[1].cut_start or sents[1].cut_end)
        self.assertTrue(sents[2].cut_end)                             # stops mid-thought
        entry = build_reference_entries([hit("a1", "x")], [sents], {1: [1, 3]})
        self.assertIn("Excerpt: [...] symptoms. [...] Another that stops mid [...]", format_reference_list(entry))

    def test_13_multiple_passages_same_chapter(self):
        a = hit("bk_00002", "Second passage sentence here. Extra.", section_path="Ch 5 > Knee > Gait", page_start="10", page_end="20")
        b = hit("bk_00001", "First passage sentence here. Extra.", section_path="Ch 5 > Knee > Gait", page_start="10", page_end="20")
        c = hit("bk_00009", "Other section sentence here. Extra.", section_path="Ch 5 > Hip > Gait", page_start="30", page_end="40")
        sents = [sentences_for(h["content"]) for h in (a, b, c)]
        entries = build_reference_entries([a, b, c], sents, {1: [1], 2: [1], 3: [1]})
        self.assertEqual(len(entries), 2)                              # same section merged, other section apart
        self.assertEqual(entries[0]["excerpts"], ["First passage sentence here.", "Second passage sentence here."])
        self.assertEqual(entries[1]["excerpts"], ["Other section sentence here."])

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
# Supporting behaviour
# --------------------------------------------------------------------------

class SupportingBehaviour(unittest.TestCase):

    def test_model_text_is_never_printed(self):
        reply = "Source 1: 2. Patellofemoral pain is caused by weak quadriceps (outside fact)."
        engine, _ = make_engine(lambda p: [hit("a1", PF_TEXT)], reply)
        ref = engine.find_references(RIGHT_KNEE_NOTE)["references"]
        self.assertNotIn("caused by weak quadriceps", ref)
        self.assertIn("Weakness of the hip abductors is common.", ref)

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
        text = " ".join(["knee"] * 400)
        pieces = qe.split_note(text, 100)
        self.assertTrue(all(len(p) <= 100 and not p.startswith("kne ") for p in pieces))
        self.assertTrue(all(set(p.split()) == {"knee"} for p in pieces))

    def test_sentence_splitter_abbreviations_decimals_and_references(self):
        text = ("Load rises from 0.5 times body weight to 3 times (Fig. 5-29), as shown by Smith et al. in 1999. "
                "Next sentence here.27,47,135 Third sentence is here. Dr. Jones agreed, e.g. at 30 degrees. Last one.")
        got = split_text(text)
        self.assertEqual(len(got), 5)
        self.assertTrue(got[0].endswith("in 1999."))
        self.assertTrue(got[1].endswith("here.27,47,135"))
        self.assertTrue(got[3].startswith("Dr. Jones"))

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

    def test_page_text(self):
        self.assertEqual(page_text(hit("a", "x", page_start="186", page_end="186")), "Page: 186")
        self.assertEqual(page_text(hit("a", "x", page_start="7", page_end=None)), "Page: 7")
        self.assertEqual(page_text(hit("a", "x", page_start=None, page_end=None)), "Pages: not recorded")

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
        # soft check for a human: the osteoarthrosis / left-knee case should not be listed
        if "Rheumatic conditions" in out["references"] or "OSTEOARTHROSIS" in out["references"]:
            print("WARNING: the osteoarthrosis passage is still listed")


if __name__ == "__main__":
    unittest.main()
