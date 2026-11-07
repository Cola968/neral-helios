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


    def test_quantitative_transfer_gap_excludes_ungraded_python_cases(self):
        cases = [
            {"id": "A-num", "domain": "math", "kind": "integer",
             "prompt": "Find the integer.", "answer": "7"},
            {"id": "A-code", "domain": "code", "kind": "python",
             "prompt": "Implement the function.", "answer": "def f(): return 1"},
            {"id": "B-eur", "domain": "money", "kind": "eur",
             "prompt": "Convert cents to euros.", "answer": "1,05 EUR"},
            {"id": "C-code", "domain": "code", "kind": "python",
             "prompt": "Add the return.", "answer": "def g(): return 1"},
            {"id": "D-num", "domain": "math", "kind": "integer",
             "prompt": "Find the integer again.", "answer": "4"},
        ]
        meta = [{"id": c["id"], "stratum": c["id"][0]} for c in cases]
        base = {c["id"]: {"text": "?", "complete": True} for c in cases}
        few = {c["id"]: {"text": c["answer"], "complete": True} for c in cases}
        candidate = {c["id"]: {"text": c["answer"], "complete": True} for c in cases}
        candidate["A-num"]["text"] = "8"

        report = combine_scores(
            cases, meta, base, few, candidate,
            case_sha="a" * 64, base_revision="b" * 40, adapter_sha="c" * 64)
        numeric = report["quantitative_A_B"]
        self.assertEqual(numeric["total"], 2)
        self.assertEqual(numeric["base_fewshot_correct"], 2)
        self.assertEqual(numeric["neral_v5_correct"], 1)
        self.assertEqual(numeric["gap_percentage_points"], -50.0)
        self.assertEqual(numeric["paired"]["regressions"], 1)
        self.assertEqual(report["python_A_B"]["total"], 1)
        self.assertIs(report["python_A_B"]["functional_correctness_evaluated"], False)
        self.assertEqual(report["python_A_B"]["verified_correct"], None)
        self.assertEqual(report["format_adjusted_gap_A_B"], -33.3333)
        self.assertNotIn("Implement the function.", json.dumps(report))

    def test_no_numeric_transfer_cases_is_unscored_not_zero_gap(self):
        cases = [
            {"id": "A-code", "domain": "code", "kind": "python",
             "prompt": "Implement.", "answer": "def f(): return 1"},
            {"id": "B-code", "domain": "code", "kind": "python",
             "prompt": "Implement too.", "answer": "def g(): return 1"},
        ]
        meta = [{"id": c["id"], "stratum": c["id"][0]} for c in cases]
        responses = {c["id"]: {"text": c["answer"], "complete": True}
                     for c in cases}
        report = combine_scores(
            cases, meta, responses, responses, responses,
            case_sha="a" * 64, base_revision="b" * 40, adapter_sha="c" * 64)
        self.assertEqual(report["quantitative_A_B"]["total"], 0)
        self.assertIsNone(report["quantitative_A_B"]["gap_percentage_points"])
        self.assertEqual(report["python_A_B"]["total"], 2)

    def test_duplicate_tasks_are_counted_without_exporting_case_data(self):
        cases = [
            {"id": "A-private-1", "domain": "math", "kind": "integer",
             "prompt": "Calculate seven.", "answer": "7"},
            {"id": "B-private-1", "domain": "math", "kind": "integer",
             "prompt": " CALCULATE   seven.\n", "answer": "7"},
            {"id": "B-private-2", "domain": "math", "kind": "integer",
             "prompt": "Unique task.", "answer": "8"},
            {"id": "B-private-3", "domain": "math", "kind": "integer",
             "prompt": "\tUnique task.  ", "answer": "8"},
        ]
        meta = [{"id": c["id"], "stratum": c["id"][0]} for c in cases]
        wrong = {c["id"]: {"text": "0", "complete": True} for c in cases}
        right = {c["id"]: {"text": c["answer"], "complete": True} for c in cases}
        report = combine_scores(
            cases, meta, wrong, wrong, right,
            case_sha="a" * 64, base_revision="b" * 40, adapter_sha="c" * 64)

        clusters = report["task_clusters"]
        self.assertEqual(clusters["normalization"], "casefold_whitespace")
        self.assertEqual(clusters["overall"], {
            "total_cases": 4, "distinct_tasks": 2, "duplicate_extra_cases": 2,
            "repeated_clusters": 2, "max_multiplicity": 2,
        })
        self.assertEqual(clusters["per_stratum"]["A"]["distinct_tasks"], 1)
        self.assertEqual(clusters["per_stratum"]["B"]["distinct_tasks"], 2)
        self.assertEqual(clusters["per_stratum"]["B"]["duplicate_extra_cases"], 1)
        self.assertEqual(clusters["per_stratum"]["C"]["max_multiplicity"], 0)
        paired = report["overall"]["paired_fewshot_vs_neral"]
        self.assertEqual(paired["improvements"], 4)  # Existing row scores stay intact.
        self.assertEqual(paired["exact_mcnemar_p"], 0.125)
        self.assertEqual(paired["independence"], "duplicate_tasks_not_independent")
        self.assertTrue(paired["exploratory"])
        self.assertFalse(paired["significance_claim_allowed"])
        self.assertEqual(report["quantitative_A_B"]["paired"]["independence"],
                         "duplicate_tasks_not_independent")
        self.assertFalse(report["significance_claim_allowed"])
        exported = json.dumps(report)
        for case in cases:
            self.assertNotIn(case["id"], exported)
            self.assertNotIn(case["prompt"], exported)

    def test_distinct_tasks_do_not_prove_independence(self):
        responses = {c["id"]: {"text": c["answer"], "complete": True} for c in CASES}
        report = combine_scores(
            CASES, META, responses, responses, responses,
            case_sha="a" * 64, base_revision="b" * 40, adapter_sha="c" * 64)
        self.assertEqual(report["task_clusters"]["overall"]["distinct_tasks"], 4)
        paired = report["overall"]["paired_fewshot_vs_neral"]
        self.assertEqual(paired["independence"], "unverified")
        self.assertTrue(paired["exploratory"])
        self.assertFalse(paired["significance_claim_allowed"])

    def test_unmatched_manifest_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            combine_scores(CASES, META[:-1], {}, {}, {},
                           case_sha="a"*64, base_revision="b"*40, adapter_sha="c"*64)


    def test_pinned_runtime_identity_works_with_shadowed_model_catalog(self):
        """Regression for real Windows run 42394712562 (wrong control catalog)."""
        import sys
        import types
        from unittest.mock import patch
        from neral_hf_identity import (
            CHAT_TEMPLATE, validate_hf_snapshot, validate_adapter,
        )
        from scripts.neral_ext_run import pinned_runtime_identity

        fake = types.ModuleType("model_catalog")
        fake.catalog = lambda: None  # intentionally lacks HF_BASE
        with patch.dict(sys.modules, {"model_catalog": fake}):
            template, base_validator, adapter_validator = pinned_runtime_identity()
        self.assertEqual(template, CHAT_TEMPLATE)
        self.assertIs(base_validator, validate_hf_snapshot)
        self.assertIs(adapter_validator, validate_adapter)

    def test_gpu_inference_never_imports_ambiguous_trainer_modules(self):
        """No hidden PYTHONPATH shadowing, and no sys.path mutation."""
        import ast
        source = Path("scripts/neral_ext_run.py").read_text(encoding="utf8")
        module = ast.parse(source)
        fn = next(node for node in module.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_gpu_run")
        for node in ast.walk(fn):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn(node.module,
                    {"qwen_local_runtime", "model_catalog", "neral_adapter_candidate"})
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertFalse(
                    isinstance(node.func.value, ast.Attribute)
                    and isinstance(node.func.value.value, ast.Name)
                    and node.func.value.value.id == "sys"
                    and node.func.value.attr == "path"
                    and node.func.attr in {"insert", "append"})

    def test_pinned_source_checkout_still_required(self):
        """Local GPU evaluation keeps the exact immutable trainer checkout gate."""
        import ast
        source = Path("scripts/neral_ext_run.py").read_text(encoding="utf8")
        tree = ast.parse(source)
        fn = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == "_gpu_run")
        self.assertIn("args.trainer_root.resolve(strict=True)",
                      ast.unparse(fn))


if __name__ == "__main__":
    unittest.main()
