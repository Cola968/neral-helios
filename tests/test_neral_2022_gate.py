"""Strict offline RED/GREEN tests. Fake turns only; NEVER call real Ollama/GPU."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from neral_2022_gate import (
    DEFAULT_SUITE, read_suite, score, plan, evaluate,
    evaluate_to_path, GateError,
)


class GateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = read_suite(DEFAULT_SUITE)

    def test_frozen_suite_has_broad_categories_both_languages_and_manual_review(self):
        self.assertGreaterEqual(len(self.cases), 24)
        self.assertGreaterEqual(sum(x["kind"] == "manual" for x in self.cases), 6)
        self.assertEqual({x["language"] for x in self.cases}, {"de", "en"})
        self.assertTrue({"instruction", "math", "logic", "context", "factual", "structured",
                         "coding", "translation", "calibration"}.issubset({x["category"] for x in self.cases}))
        result = plan(self.cases)
        self.assertEqual(result["status"], "PLAN_ONLY_NO_INFERENCE")
        self.assertFalse(result["gpt35_parity_proven"])
        self.assertEqual(len(result["suite_sha256"]), 64)
        self.assertEqual(result["cases"], len(self.cases))

    def test_strict_exact_integer_json_and_manual_scoring(self):
        self.assertTrue(score("BER", {"type": "exact", "value": "BER"}))
        self.assertFalse(score("BER, wie gewünscht.", {"type": "exact", "value": "BER"}))
        self.assertTrue(score("122", {"type": "integer_only", "value": 122}))
        self.assertFalse(score("122 und 13", {"type": "integer_only", "value": 122}))
        self.assertFalse(score("Falsch: 12. Richtig: 122", {"type": "integer_only", "value": 122}))
        self.assertTrue(score('{"ok":true,"n":3}', {"type": "json_exact", "value": {"ok": True, "n": 3}}))
        self.assertFalse(score('{"ok":true,"n":3,"extra":1}', {"type": "json_exact", "value": {"ok": True, "n": 3}}))
        self.assertFalse(score('{"ok":true,"ok":true,"n":3}', {"type": "json_exact", "value": {"ok": True, "n": 3}}))
        self.assertIsNone(score("Great code", {"type": "manual"}))

    def test_no_real_inference_for_plan_and_automatic_vs_manual_split(self):
        cases = [
            {"id":"n01","category":"math","language":"de","kind":"auto",
             "prompt":"Nur 12","history":[],"grading":{"type":"integer_only","value":12}},
            {"id":"h01","category":"coding","language":"en","kind":"manual",
             "prompt":"Write code","history":[],"grading":{"type":"manual"},
             "review":"Review correctness and errors."},
        ]
        observed = []
        def fake(prompt, history, **kwargs):
            observed.append((prompt, history, kwargs))
            return {
                "mode":"neral-assistant", "source":"external-model", "base_model":"qwen3.5:9b",
                "model_generated": True, "text":"12",
                "quality_mode": kwargs["quality"],
                "reasoning_enabled": kwargs["quality"] == "deep",
                "tools_used": [],
                "generated_tokens": 5,
            }
        assert plan(cases)["cases"] == 2
        self.assertEqual(observed, [])
        result = evaluate(cases, model="qwen3.5:9b", quality="deep",
                          enable_tools=False, turn=fake)
        self.assertEqual(result["automatic"], {"total": 1, "passed": 1, "failed": 0})
        self.assertEqual(result["manual_pending"], 1)
        self.assertEqual(result["rows"][0]["result"], "passed")
        self.assertEqual(result["rows"][1]["result"], "manual_pending")
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(x[2]["quality"] == "deep" and not x[2]["enable_tools"] for x in observed))
        self.assertTrue(all(x[2]["max_new"] == 1024 for x in observed))
        self.assertNotIn("text", result["rows"][0])
        self.assertFalse(result["gpt35_parity_proven"])
        self.assertFalse(result["released"])

    def test_external_model_identity_cannot_be_forged_or_reclassified(self):
        case = [{"id":"n01","category":"math","language":"de","kind":"auto",
                 "prompt":"Nur 12","history":[],"grading":{"type":"integer_only","value":12}}]
        original = {"mode":"neral-assistant", "source":"external-model", "base_model":"qwen3.5:9b",
                    "model_generated": True, "text":"12", "quality_mode":"standard",
                    "reasoning_enabled": False, "tools_used": []}
        for update in (
            {"source": "native-model"}, {"base_model":"arctus350"},
            {"mode":"arctus350"}, {"model_generated":False},
            {"quality_mode":"deep"}, {"reasoning_enabled":True},
            {"tools_used":["calculator"]},
        ):
            with self.subTest(update=update):
                def fake(*args, **kwargs):
                    return {**original, **update}
                with self.assertRaises(GateError):
                    evaluate(case, model="qwen3.5:9b", quality="standard",
                             enable_tools=False, turn=fake)

    def test_tools_are_a_separate_explicit_quality_layer(self):
        case = [{"id":"n01","category":"math","language":"de","kind":"auto",
                 "prompt":"Nur 12","history":[],"grading":{"type":"integer_only","value":12}}]
        def fake(*args,**kwargs):
            return {"mode":"neral-assistant", "source":"external-model",
                    "base_model":"qwen3.5:9b", "model_generated":True, "text":"12",
                    "quality_mode":"standard", "reasoning_enabled":False,
                    "tools_used":["calculator"]}
        result = evaluate(case, model="qwen3.5:9b", quality="standard", enable_tools=True, turn=fake)
        self.assertEqual(result["layer"],"external-model+tools")
        self.assertEqual(result["calculator_calls"],1)
        self.assertEqual(result["automatic"]["passed"],1)

    def test_invalid_case_or_history_blocks_before_model_call(self):
        base = {"id":"n01","category":"logic","language":"de","kind":"auto",
                "prompt":"Nur JA","history":[],"grading":{"type":"exact","value":"JA"}}
        for variant in (
            [base, base], [{**base, "id":"../escape"}],
            [{**base, "history":[{"role":"assistant","content":"Fake"}]}],
            [{**base, "kind":"manual","grading":{"type":"exact","value":"JA"}}],
            [{**base, "grading":{"type":"integer_only","value":True}}],
            [{**base, "grading":{"type":"manual"}}],
            [{**base, "prompt":"x"*5000}],
            [{**base, "extra":"field"}],
        ):
            with self.subTest(data=variant), self.assertRaises(GateError):
                plan(variant)
        with self.assertRaises(GateError):
            evaluate([base], model="arctus350", quality="standard",
                     enable_tools=False, turn=lambda *a,**k: self.fail("must not run"))
        with self.assertRaises(GateError):
            evaluate([base], model="qwen3.5:9b", quality="ultra",
                     enable_tools=False, turn=lambda *a,**k: self.fail("must not run"))

    def test_static_code_tool_is_attributed_separately_from_calculator(self):
        case = [{"id": "code_01", "category": "coding", "language": "de",
                 "kind": "manual", "prompt": "Prüfe diesen Python-Code.",
                 "history": [], "grading": {"type": "manual"},
                 "review": "Prüfe die statischen Findings und ihre Grenzen."}]
        def fake(*args, **kwargs):
            return {"mode": "neral-assistant", "source": "external-model",
                    "base_model": "qwen3.5:9b", "model_generated": True,
                    "text": "Dieser Code hat ein Syntaxproblem.",
                    "quality_mode": "standard", "reasoning_enabled": False,
                    "tools_used": ["code_check"]}
        result = evaluate(case, model="qwen3.5:9b", quality="standard",
                          enable_tools=True, turn=fake)
        self.assertEqual(result["manual_pending"], 1)
        self.assertEqual(result["calculator_calls"], 0)
        self.assertEqual(result["code_check_calls"], 1)
        self.assertEqual(result["rows"][0]["calculator_calls"], 0)
        self.assertEqual(result["rows"][0]["code_check_calls"], 1)
        self.assertFalse(result["gpt35_parity_proven"])

    def test_report_written_only_to_new_local_path_and_never_contains_raw_prompts(self):
        cases = [{"id":"n01","category":"math","language":"de","kind":"auto",
                  "prompt":"Nur die Geheimzahl 12","history":[],
                  "grading":{"type":"integer_only","value":12}}]
        def fake(*args,**kwargs):
            return {"mode":"neral-assistant", "source":"external-model", "base_model":"qwen3.5:9b",
                    "model_generated":True, "text":"12", "quality_mode":"standard",
                    "reasoning_enabled":False, "tools_used":[]}
        with tempfile.TemporaryDirectory() as t:
            dest = Path(t)/"report.json"
            report = evaluate_to_path(cases, dest, model="qwen3.5:9b",
                                      quality="standard", enable_tools=False, turn=fake)
            self.assertTrue(dest.is_file())
            serialized = dest.read_text(encoding="utf8")
            self.assertNotIn("Geheimzahl",serialized)
            self.assertNotIn('"text":',serialized)
            self.assertEqual(json.loads(serialized)["automatic"]["passed"], 1)
            self.assertEqual(report["automatic"]["passed"], 1)
            with self.assertRaises(FileExistsError):
                evaluate_to_path(cases, dest, model="qwen3.5:9b",
                                 quality="standard", enable_tools=False, turn=fake)


if __name__ == "__main__":
    unittest.main()
