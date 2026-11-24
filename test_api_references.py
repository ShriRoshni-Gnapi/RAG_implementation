"""
Tests for POST /references in api.py.

Run:  python3 -m unittest test_api_references -v

The engine is replaced by a fake, so these tests need neither Qdrant, the
embedding model nor the LLM server. They prove the route's contract: what it
accepts, what it returns, what it refuses, and that a note is never echoed or
logged. They say nothing about the quality of real references.
"""

import contextlib
import io
import json
import unittest

try:
    from fastapi.testclient import TestClient
    import api
    from llm_client import LLMError
    from qdrant_store import VectorStoreError
    import test_references as tr
    HAVE_API = True
except Exception:                    # fastapi/httpx/qdrant-client not installed here
    HAVE_API = False

NOTE_MARKER = "SECRET-NOTE-TEXT-7731"          # appears only in the request; must never come back or be logged


def entry(book="Therapeutic Exercise", summary=("A summary paragraph.",), **kw):
    base = {"book": book, "author": "Carolyn Kisner", "chapter": "Chapter 21: The Knee", "section": None,
            "pages": "pp. 687–758", "summary": list(summary), "sentences": ["An original sentence."],
            "chunk_ids": ["c1"], "sources": [1], "summary_raw": "RAW", "summary_passage": ["P"]}
    base.update(kw)
    return base


def result(entries, notes=(), failures=()):
    return {"reference_entries": entries, "summary_notes": list(notes), "failures": list(failures),
            "generator": "test-gen-1", "raw_answer": "Source 1: 1", "candidates": [{"source": 1}],
            "model": "m", "references": "x"}


@unittest.skipUnless(HAVE_API, "fastapi, httpx and qdrant-client are needed for the route tests")
class ReferencesRoute(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(api.app)             # no `with`: startup (model load) is not run
        self.calls = []
        self.reply = result([entry()])
        self.original = api.engine.find_references
        api.engine.find_references = self._fake

    def tearDown(self):
        api.engine.find_references = self.original

    def _fake(self, note, **kw):
        self.calls.append((note, kw))
        return self.reply


    def post(self, **body):
        body.setdefault("subjective", f"Knee pain on stairs. {NOTE_MARKER}")
        return self.client.post("/references", json=body)

    def test_returns_only_the_agreed_fields(self):
        r = self.post(objective="Flexion 110 degrees.", assessment="Patellofemoral pain.")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {
            "status": "ready", "generator": "test-gen-1",
            "references": [{"book": "Therapeutic Exercise", "author": "Carolyn Kisner",
                            "chapter": "Chapter 21: The Knee", "section": None, "pages": "pp. 687–758",
                            "summary": ["A summary paragraph."], "sentences": ["An original sentence."]}]})
        text = r.text
        for hidden in ("raw_answer", "candidates", "summary_notes", "summary_passage", "summary_raw",
                       "chunk_ids", "Source 1"):
            self.assertNotIn(hidden, text)

    def test_only_the_three_sections_reach_the_engine_as_structured_sections(self):
        self.post(subjective="S text", objective="", assessment="A text")
        note, kw = self.calls[0]
        self.assertEqual(note, "")                                    # no free-text note is passed
        self.assertEqual(kw["sections"], [("S", "S text"), ("A", "A text")])   # empty section left out
        self.assertTrue(kw["summaries"] and kw["summary_only"])
        self.assertEqual(kw["top_k"], 5)

    def test_unknown_fields_are_ignored_and_not_forwarded(self):
        self.post(subjective="S text", plan="Do exercises", patient_name="Jane Doe")
        _, kw = self.calls[0]
        self.assertEqual(kw["sections"], [("S", "S text")])
        self.assertNotIn("Jane Doe", json.dumps(kw["sections"]))

    def test_citations_without_a_supported_summary_are_left_out(self):
        self.reply = result([entry(summary=[]), entry(book="Second Book", summary=["Another."])])
        r = self.post()
        self.assertEqual([x["book"] for x in r.json()["references"]], ["Second Book"])

    def test_nothing_relevant_is_an_empty_set_not_an_error(self):
        self.reply = result([])
        r = self.post()
        self.assertEqual((r.status_code, r.json()["status"], r.json()["references"]), (200, "empty", []))
        self.reply = result([entry(summary=[])])                # found, but no summary could be supported
        self.assertEqual(self.post().json()["status"], "empty")

    def test_a_failed_summary_call_is_unavailable_not_empty(self):
        self.reply = result([entry()], failures=["citation 1: call failed"])
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = self.post()
        self.assertEqual(r.status_code, 503)                          # so the caller does not store an "empty" set
        self.assertEqual(r.json(), {"detail": "The reference service is unavailable (incomplete model reply)."})
        self.assertNotIn(NOTE_MARKER, r.text)
        self.assertIn("citation 1: call failed", out.getvalue())     # the reason is logged, the note is not
        self.assertNotIn(NOTE_MARKER, out.getvalue())

    def test_model_or_vector_store_failures_are_unavailable(self):
        for error in (LLMError(f"cannot reach server {NOTE_MARKER}"),
                      VectorStoreError(f"qdrant down {NOTE_MARKER}")):
            def boom(note, **kw):
                raise error
            api.engine.find_references = boom
            r = self.post()
            self.assertEqual(r.status_code, 503)
            self.assertNotIn(NOTE_MARKER, r.text)                     # the message is not passed on
            self.assertIn(type(error).__name__, r.text)

    def test_an_unexpected_error_is_generic_and_the_note_is_not_logged(self):
        def boom(note, **kw):
            raise RuntimeError(f"unexpected {NOTE_MARKER}")
        api.engine.find_references = boom
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            r = self.post()
        self.assertEqual((r.status_code, r.json()), (500, {"detail": "The reference lookup failed."}))
        self.assertNotIn(NOTE_MARKER, r.text)
        self.assertNotIn(NOTE_MARKER, out.getvalue())
        self.assertIn("RuntimeError", out.getvalue())

    def test_an_empty_note_is_refused(self):
        r = self.client.post("/references", json={"subjective": "  ", "objective": "", "assessment": ""})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(self.calls, [])

    def test_an_over_long_section_is_refused_and_not_echoed(self):
        r = self.client.post("/references", json={"subjective": NOTE_MARKER + "x" * api.MAX_SECTION_CHARS})
        self.assertEqual((r.status_code, r.json()), (422, {"detail": "The request is invalid."}))
        self.assertNotIn(NOTE_MARKER, r.text)
        self.assertEqual(self.calls, [])

    def test_a_bad_top_k_is_refused_without_echoing_the_note(self):
        for top_k in (0, 21, "many"):
            r = self.client.post("/references", json={"subjective": NOTE_MARKER, "top_k": top_k})
            self.assertEqual((r.status_code, r.json()), (422, {"detail": "The request is invalid."}))
            self.assertNotIn(NOTE_MARKER, r.text)

    def test_other_routes_keep_the_default_validation_answer(self):
        r = self.client.post("/ask", json={"question": NOTE_MARKER, "top_k": 0})
        self.assertEqual(r.status_code, 422)
        self.assertIsInstance(r.json()["detail"], list)                # FastAPI's own, detailed answer


@unittest.skipUnless(HAVE_API, "fastapi, httpx and qdrant-client are needed for the route tests")
class UnusableModelReplies(unittest.TestCase):
    """
    The real engine runs behind the route; only the search results and the
    model are fake. An unusable model reply must be a 503 (so a caller does not
    store it), and only a finished lookup may answer "empty".
    """

    INCOMPLETE = {"detail": "The reference service is unavailable (incomplete model reply)."}

    def setUp(self):
        self.client = TestClient(api.app)
        self.original_engine = api.engine

    def tearDown(self):
        api.engine = self.original_engine

    def call(self, selection, summary="x", **llm_kw):
        api.engine, _ = tr.make_engine(lambda p: [tr.hit("a1", tr.PF_TEXT)], selection,
                                       summary_reply=summary, **llm_kw)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            r = self.client.post("/references", json={
                "subjective": f"Knee pain on stairs. {NOTE_MARKER}", "assessment": "Patellofemoral pain."})
        self.assertNotIn(NOTE_MARKER, r.text)                          # never echoed
        self.assertNotIn(NOTE_MARKER, out.getvalue())                  # never logged
        return r, out.getvalue()

    def test_an_unreadable_source_list_is_503_not_empty(self):
        for reply in ("Sure! Here are some thoughts about the note.", ""):
            r, log = self.call(reply)
            self.assertEqual((r.status_code, r.json()), (503, self.INCOMPLETE), repr(reply))
            self.assertIn("the model's source list could not be read", log)

    def test_an_empty_or_unnumbered_summary_reply_is_503(self):
        r, log = self.call("Source 1: 1, 2", "")
        self.assertEqual((r.status_code, r.json()), (503, self.INCOMPLETE))
        self.assertIn("citation 1: empty reply", log)
        r, log = self.call("Source 1: 1, 2", tr.GOOD_SUMMARY)         # no sentence numbers at all
        self.assertEqual(r.status_code, 503)
        self.assertIn("citation 1: reply without sentence numbers", log)

    def test_a_failed_summary_call_is_503(self):
        r, log = self.call("Source 1: 1, 2", summary_error=LLMError(f"down {NOTE_MARKER}"))
        self.assertEqual(r.status_code, 503)
        self.assertIn("citation 1: call failed", log)

    def test_finished_lookups_with_nothing_supported_are_empty_200(self):
        for selection, summary in (
            (tr.NO_REFERENCE_TEXT, "x"),                               # nothing relevant
            ("Source 1: 1, 2", "INSUFFICIENT"),                        # the model says the passage is not enough
            ("Source 1: 1, 2", "The pain rises by 40 percent on stairs in these patients [1]."),   # checks reject it
        ):
            r, log = self.call(selection, summary)
            self.assertEqual((r.status_code, r.json()["status"], r.json()["references"]), (200, "empty", []))
            self.assertEqual(log, "")

    def test_a_good_lookup_is_ready(self):
        r, _ = self.call("Source 1: 1, 2", f"{tr.GOOD_SUMMARY} [1, 2]")
        body = r.json()
        self.assertEqual((r.status_code, body["status"]), (200, "ready"))
        self.assertEqual(body["references"][0]["summary"], [tr.GOOD_SUMMARY])
        self.assertEqual(body["references"][0]["sentences"], [tr.S1, tr.S2])


if __name__ == "__main__":
    unittest.main()
