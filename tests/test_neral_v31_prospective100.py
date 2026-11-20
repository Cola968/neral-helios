"""Contract tests for the prospective V3.1 100-case gate (CPU only)."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_v31_prospective100 import (
    CATEGORIES, build_suite, evaluate_case, seal_suite, summarize,
    canonical_integer, answer_token,
)


class V31ProspectiveSuite(unittest.TestCase):
    def test_reproducible_100_unique_with_20_in_each_category(self):
        a = build_suite("00112233445566778899aabbccddeeff")
        b = build_suite("00112233445566778899aabbccddeeff")
        self.assertEqual(a, b)
        self.assertEqual(len(a), 100)
        self.assertEqual({c: sum(x["category"] == c for x in a)
                          for c in CATEGORIES}, {c: 20 for c in CATEGORIES})
        self.assertEqual(len({x["question_sha256"] for x in a}), 100)
        self.assertEqual(len({x["question"] for x in a}), 100)
        for row in a:
            self.assertEqual(row["question_sha256"],
                             hashlib.sha256(row["question"].encode("utf8")).hexdigest())
            self.assertNotIn("answer", row["question"].lower())

    def test_random_seeds_never_duplicate_questions(self):
        # Programming source template f==2 depends on (a,b) but not c.
        # Random collision must resample, not abort the entire real GPU gate.
        for i in range(256):
            with self.subTest(i=i):
                suite=build_suite(f"{i:032x}")
                self.assertEqual(len(suite),100)
                self.assertEqual(len({c["question_sha256"] for c in suite}),100)

    def test_secret_seed_changes_math_logic_programming(self):
        a = build_suite("00112233445566778899aabbccddeeff")
        b = build_suite("ffeeddccbbaa99887766554433221100")
        for cat in ("mathematics", "logic", "programming"):
            self.assertNotEqual([x["question"] for x in a if x["category"] == cat],
                                [x["question"] for x in b if x["category"] == cat])

    def test_seal_is_create_only_and_detects_tampering(self):
        cases = build_suite("00112233445566778899aabbccddeeff")
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "sealed.json"
            digest = seal_suite(path, cases)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            self.assertEqual(len(json.loads(path.read_text())["cases"]), 100)
            with self.assertRaises(FileExistsError):
                seal_suite(path, cases)

    def test_numeric_parser_fails_closed_on_ambiguous_or_partial(self):
        self.assertEqual(canonical_integer("Ergebnis: 64 Gläser\nu1 (user): 4"), 64)
        self.assertEqual(canonical_integer("Z3: 2"), 2)
        self.assertIsNone(canonical_integer("2 und 3 sind möglich"))
        self.assertIsNone(canonical_integer("Ich vermute 64"))
        self.assertIsNone(canonical_integer(""))

    def test_web_scoring_requires_real_sources_and_checked_citations(self):
        case = next(x for x in build_suite("00112233445566778899aabbccddeeff")
                    if x["category"] == "web")
        good = {"text": "Firefox: Gecko.\nChrome: Blink.\n[web1] „Firefox uses Gecko“\n[web2] „Chrome uses Blink“",
                "tool_used": "direct_reasoning", "agent_checked": True,
                "sources": [{"page_read": True, "url": "https://mozilla.org/"},
                            {"page_read": True, "url": "https://chromium.org/"}]}
        # Constrain test to this specific subject pair to avoid order dependence.
        case = {"id": "web-fixture", "category": "web", "kind": "web",
                "entities": ["Firefox", "Chrome"],
                "facts": ["Gecko", "Blink"]}
        self.assertTrue(evaluate_case(case, good, 200)["correct"])
        self.assertFalse(evaluate_case(case, {**good, "agent_checked": False}, 200)["correct"])
        self.assertFalse(evaluate_case(case, {**good, "sources": []}, 200)["correct"])
        self.assertFalse(evaluate_case(case, {**good, "text": "Firefox Gecko Chrome Blink"}, 200)["correct"])
        self.assertFalse(evaluate_case(case, {**good, "text": "Firefox Gecko Chrome Blink\n[web1] „Chrome Blink“"}, 200)["correct"])
        self.assertFalse(evaluate_case(case, good, 503)["correct"])

    def test_math_and_logic_require_actual_tool_provenance(self):
        math = {"category": "mathematics", "kind": "integer", "gold": 64,
                "tool": "python_sandbox"}
        good = {"text": "Ergebnis: 64", "tool_used": "python_sandbox",
                "agent_checked": True, "sources": []}
        self.assertTrue(evaluate_case(math, good, 200)["correct"])
        self.assertFalse(evaluate_case(math, {**good, "tool_used": None}, 200)["correct"])
        self.assertFalse(evaluate_case(math, {**good, "agent_checked": False}, 200)["correct"])
        logic = {"category": "logic", "kind": "integer", "gold": 2, "tool": "z3_solver"}
        z3 = {**good, "text": "Tag 1 ist Montag.\nZ3: 2", "tool_used": "z3_solver"}
        self.assertTrue(evaluate_case(logic, z3, 200)["correct"])
        self.assertFalse(evaluate_case(logic, {**z3, "text": "Tag 1 ist Montag. 2"}, 200)["correct"])

    def test_summary_refuses_missing_or_duplicate_rows(self):
        cases = build_suite("00112233445566778899aabbccddeeff")
        rows = [{"id": x["id"], "category": x["category"],
                 "correct": True, "latency_seconds": 1.5,
                 "error": None} for x in cases]
        self.assertEqual(summarize(cases, rows)["correct"], 100)
        self.assertTrue(summarize(cases, rows)["gate_90_of_100"])
        with self.assertRaises(ValueError):
            summarize(cases, rows[:99])
        with self.assertRaises(ValueError):
            summarize(cases, rows[:-1] + [rows[0]])

    def test_html_homepage_bootstraps_json_session(self):
        import io
        from unittest.mock import patch
        from scripts.neral_v31_prospective100 import _open_client
        calls = []

        class FakeResponse:
            def __init__(self, status, payload):
                self.status = status
                self.stream = io.BytesIO(payload)
            def __enter__(self):
                return self
            def __exit__(self, *_):
                return False
            def read(self, n=-1):
                return self.stream.read(n)

        class FakeOpener:
            def open(self, req, timeout):
                calls.append((req.full_url, req.get_method(), timeout))
                if req.full_url.endswith(":9690/"):
                    return FakeResponse(200, b"<html>NERAL V3</html>")
                if req.full_url.endswith("/api/neral/preview/session"):
                    return FakeResponse(200, json.dumps({
                        "csrf": "signed-local-csrf",
                        "three_tier_enabled": True,
                        "evidence_agent_enabled": True
                    }).encode())
                raise AssertionError("Unexpected API call")

        with patch("scripts.neral_v31_prospective100.build_opener",
                   return_value=FakeOpener()):
            fetch, csrf = _open_client()
        self.assertEqual(csrf, "signed-local-csrf")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0][1], "GET")
        self.assertEqual(calls[1][1], "GET")

    def test_knowledge_scoring_avoids_answer_in_long_text(self):
        case = {"category": "knowledge", "kind": "word", "gold": "Sauerstoff"}
        self.assertTrue(evaluate_case(case, {"text": "Sauerstoff"}, 200)["correct"])
        self.assertTrue(evaluate_case(case, {"text": "Antwort: Sauerstoff."}, 200)["correct"])
        self.assertFalse(evaluate_case(case, {"text": "Vielleicht Sauerstoff oder Stickstoff"}, 200)["correct"])


if __name__ == "__main__":
    unittest.main()
