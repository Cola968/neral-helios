"""CPU-only NERAL-EXT scorer safety and paired correctness tests."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_ext import (
    read_cases, evaluate, paired_report, semantic_grade, ngram_overlap,
    write_report,
)


def cases():
    return [
        {"id": "public-01", "domain": "word_math", "kind": "integer", "prompt": "Wie viele Kugeln bleiben nach der Abgabe von acht aus 37?", "answer": "29"},
        {"id": "public-02", "domain": "fractions", "kind": "rational", "prompt": "Berechne ein Drittel von sechs Siebteln.", "answer": "2/7"},
        {"id": "public-03", "domain": "finance", "kind": "eur", "prompt": "Wie viel Euro sind 125 Cent?", "answer": "1,25 EUR"},
        {"id": "public-04", "domain": "code_trace", "kind": "integer", "prompt": "Welchen Wert ergibt sum([2, 3, 5])?", "answer": "10"},
    ]


class NeralExtTests(unittest.TestCase):
    def test_semantic_equals_with_explicit_final_only(self):
        task = cases()[0]
        self.assertEqual(semantic_grade(task, "29"), "correct")
        self.assertEqual(semantic_grade(task, "Rechnung 37-8=29\nFINAL_ANSWER: 29"), "correct")
        self.assertEqual(semantic_grade(task, "Rechnung 37-8=29\nFINAL_ANSWER: 28"), "incorrect")
        self.assertEqual(semantic_grade(task, "Es sind 37 Kugeln, dann 8, daher 29"), "ungraded")
        self.assertEqual(semantic_grade(task, "FINAL_ANSWER: 29\nFINAL_ANSWER: 28"), "ungraded")
        self.assertEqual(semantic_grade(task, "FINAL_ANSWER: 29\nweiterer Text"), "ungraded")
        self.assertEqual(semantic_grade(task, "29", complete=False), "ungraded")

    def test_rational_math_exact_and_units_must_match(self):
        self.assertEqual(semantic_grade(cases()[1], "4/14"), "correct")
        self.assertEqual(semantic_grade(cases()[1], "0.285714"), "incorrect")
        self.assertEqual(semantic_grade(cases()[2], "1.25 EUR"), "correct")
        self.assertEqual(semantic_grade(cases()[2], "1,25"), "ungraded")
        self.assertEqual(semantic_grade(cases()[2], "1,25 USD"), "ungraded")
        self.assertEqual(semantic_grade(cases()[2], "1,26 EUR"), "incorrect")
        self.assertEqual(semantic_grade(cases()[3], "import os; os.system('whoami')"), "ungraded")

    def test_reject_dataset_modification_and_invalid_cases(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ext.jsonl"
            path.write_text("\n".join(json.dumps(z) for z in cases()) + "\n", encoding="utf8")
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(len(read_cases(path, sha)), 4)
            with self.assertRaises(ValueError):
                read_cases(path, "0" * 64)
            path.write_text(path.read_text() + "tampering", encoding="utf8")
            with self.assertRaises(ValueError):
                read_cases(path, sha)

    def test_aggregate_never_exposes_prompts_or_model_text(self):
        original = cases()
        base = {"public-01": {"text": "29", "complete": True},
                "public-02": {"text": "3/7", "complete": True},
                "public-03": {"text": "1,25 USD", "complete": True},
                "public-04": {"text": "10", "complete": False}}
        candidate = {"public-01": {"text": "29", "complete": True},
                     "public-02": {"text": "4/14", "complete": True},
                     "public-03": {"text": "1,26 EUR", "complete": True},
                     "public-04": {"text": "10", "complete": True}}
        b = evaluate(original, base)
        n = evaluate(original, candidate)
        score = paired_report(original, b, n, case_hash="e"*64,
                              base_revision="f"*40, adapter_sha="a"*64)
        self.assertEqual(score["base"]["semantic_correct"], 1)
        self.assertEqual(score["candidate"]["semantic_correct"], 3)
        self.assertEqual(score["paired"]["improvements"], 2)
        self.assertEqual(score["paired"]["regressions"], 0)
        self.assertFalse(score["general_improvement_proven"])
        self.assertFalse(score["external_independence_verified"])
        self.assertFalse(score["trained_adapter_promoted"])
        for value in ("37", "Kugeln", "1,25 USD", "whoami", "prompt", "text"):
            self.assertNotIn(value, json.dumps(score))

    def test_rejects_incomplete_or_misaligned_responses(self):
        base = {z["id"]: {"text": z["answer"], "complete": True} for z in cases()}
        self.assertEqual(evaluate(cases(), base)["public-01"]["semantic"], "correct")
        base["extra"] = {"text": "0", "complete": True}
        with self.assertRaises(ValueError):
            evaluate(cases(), base)

    def test_10gram_overlap_requires_true_shared_text(self):
        query = "Heute werden in einem Werk zwölf neue Einheiten von drei Gruppen hergestellt"
        self.assertTrue(ngram_overlap([query], [query + " morgen"]))
        self.assertFalse(ngram_overlap([query], ["Das ist eine andere Formulierung"]))

    def test_fail_closed_report_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "report.json"
            write_report(file, {"schema": 1, "general_improvement_proven": False})
            self.assertTrue(file.exists())
            with self.assertRaises(ValueError):
                write_report(file, {"schema": 1})


    def test_mbpp_python_is_never_counted_correct_without_sandbox(self):
        """Identical source is not evidence that a generated program passes tests."""
        case = {"id": "mbpp-001", "domain": "mbpp", "kind": "python",
                "prompt": "Implement a function.", "answer": "def f(x):\\n    return x + 1"}
        from scripts.neral_ext import evaluate
        scored = evaluate([case], {"mbpp-001": {"text": case["answer"], "complete": True}})
        self.assertEqual(scored["mbpp-001"]["semantic"], "ungraded")
        self.assertFalse(scored["mbpp-001"]["strict"])

    def test_400_case_contract_accepts_pinned_mbpp_as_ungraded(self):
        from scripts.neral_ext import read_cases
        code = "def f(x):\\n    return x + 1\\n" + "# preserved benchmark reference\\n" * 5
        case = {"id": "mbpp-001", "domain": "mbpp", "kind": "python",
                "prompt": "Implement a function.", "answer": code}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "cases.jsonl"
            path.write_text(json.dumps(case) + "\\n", encoding="utf8")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(read_cases(path, digest)[0]["kind"], "python")


if __name__ == "__main__":
    unittest.main()
