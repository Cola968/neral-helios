"""CPU-only contract tests for controlled NERAL-EXT three-variant GPU runner."""
from __future__ import annotations
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_ext_run import build_messages, write_variant, combine_scores
from scripts.neral_ext import evaluate


CASES = [
    {"id": "A-01", "domain": "arith", "kind": "integer",
     "prompt": "Wie viel ist zwei plus fünf?", "answer": "7"},
    {"id": "B-01", "domain": "finance", "kind": "eur",
     "prompt": "Was kosten 105 Cent in Euro?", "answer": "1,05 EUR"},
    {"id": "C-01", "domain": "code", "kind": "python",
     "prompt": "Write a function", "answer": "def f(x):\n return x"},
    {"id": "D-01", "domain": "arith", "kind": "integer",
     "prompt": "Wie viel ist zwei plus fünf? Wetter ist sonnig.",
     "answer": "7"},
]
META = [{"id": c["id"], "stratum": c["id"][0]} for c in CASES]


class ExtRunTests(unittest.TestCase):
    def test_base_and_neral_have_identical_prompts(self):
        base = build_messages(CASES[0], "base")
        candidate = build_messages(CASES[0], "neral_v5")
        self.assertEqual(base, candidate)
        self.assertEqual([x["role"] for x in base], ["system", "user"])

    def test_fewshot_has_three_examples_outside_holdout(self):
        rows = build_messages(CASES[0], "base_fewshot")
        self.assertEqual([x["role"] for x in rows],
                         ["system", "user", "assistant", "user", "assistant",
                          "user", "assistant", "user"])
        self.assertEqual(rows[-1]["content"], CASES[0]["prompt"])

    def test_invalid_variant_is_rejected(self):
        with self.assertRaises(ValueError):
            build_messages(CASES[0], "cloud")

    def test_cpu_mocked_generation_is_create_only(self):
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "answers.jsonl"
            calls = []
            def fake(case, messages):
                calls.append(case["id"])
                return {"text": case["answer"], "complete": True, "new_tokens": 5}
            stats = write_variant(CASES, "base", file, fake)
            self.assertEqual(stats["cases"], 4)
            self.assertEqual(stats["generated_tokens"], 20)
            self.assertEqual(calls, [x["id"] for x in CASES])
            self.assertEqual(len(file.read_text().splitlines()), 4)
            with self.assertRaises(ValueError):
                write_variant(CASES, "base", file, fake)

    def test_no_truncated_answer_is_silently_complete(self):
        with tempfile.TemporaryDirectory() as td:
            file = Path(td) / "answers.jsonl"
            def fake(case, messages):
                return {"text": "x" * 4097, "complete": True, "new_tokens": 128}
            result = write_variant(CASES, "base", file, fake)
            row = json.loads(file.read_text().splitlines()[0])
            self.assertFalse(row["complete"])
            self.assertEqual(row["text"], "")
            self.assertEqual(result["incomplete"], 4)

    def test_aggregate_three_variants_per_stratum_without_raw_answers(self):
        base = {c["id"]: {"text": "unknown", "complete": True} for c in CASES}
        few = {c["id"]: {"text": c["answer"], "complete": True} for c in CASES}
        neral = {c["id"]: {"text": c["answer"], "complete": True} for c in CASES}
        score = combine_scores(CASES, META, base, few, neral,
                               case_sha="a" * 64, base_revision="b" * 40,
                               adapter_sha="c" * 64)
        self.assertEqual(set(score["strata"]), {"A", "B", "C", "D"})
        self.assertEqual(score["strata"]["A"]["base"]["semantic_correct"], 0)
        self.assertEqual(score["strata"]["A"]["neral_v5"]["semantic_correct"], 1)
        self.assertEqual(score["strata"]["C"]["neral_v5"]["ungraded"], 1)
        self.assertEqual(score["format_adjusted_gap_A_B"], 0)
        self.assertFalse(score["release_validated"])
        for token in ("Wie viel", "Wetter", "Write a function", "text", "prompt"):
            self.assertNotIn(token, json.dumps(score))

    def test_unmatched_manifest_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            combine_scores(CASES, META[:-1], {}, {}, {},
                           case_sha="a"*64, base_revision="b"*40, adapter_sha="c"*64)


if __name__ == "__main__":
    unittest.main()
