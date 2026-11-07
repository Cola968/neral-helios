"""NERAL-EXT data contracts: strictly offline synthetic fixtures, no GPU."""
from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_ext_dataset import (
    SEED, build, export, _new_family_cases, _control_cases,
    _perturbed_cases, _canonical_number,
)
from scripts.neral_decontamination import audit, normalize, shingles, load_jsonl


REVISIONS = {k: ("a" * 40) for k in ("gsm8k", "math500", "mbpp")}


def synthetic_sources():
    return {
        "gsm8k": [
            {"question": f"A story with {i} apples: calculate {i} + 3.", "answer": f"Use math.\n#### {i+3}"}
            for i in range(100)
        ],
        "math500": [
            {"problem": f"Compute the value {i*7} minus {i}.", "answer": str(i*6)}
            for i in range(60)
        ],
        "mbpp": [
            {"text": f"Write a function which adds {i} to a number.",
             "code": f"def f(x):\n    return x + {i}",
             "test_list": [f"assert f(1) == {i + 1}"],
             "test_setup_code": ""}
            for i in range(50)
        ],
    }


class ExtDatasetTests(unittest.TestCase):
    def setUp(self):
        self.sources = synthetic_sources()

    def test_400_cases_are_built_without_remote_io(self):
        cases, manifest = build(self.sources, REVISIONS)
        self.assertEqual(len(cases), 400)
        self.assertEqual(manifest["strata"], {"A": 200, "B": 100, "C": 50, "D": 50})

    def test_200_public_cases_have_three_documented_sources(self):
        cases, _ = build(self.sources, REVISIONS)
        self.assertEqual(sum(c["domain"] == "gsm8k" for c in cases), 100)
        self.assertEqual(sum(c["domain"] == "math500" for c in cases), 50)
        self.assertEqual(sum(c["domain"] == "mbpp" for c in cases), 50)

    def test_100_new_cases_from_three_independent_generators(self):
        rows = _new_family_cases()
        self.assertEqual(len(rows), 100)
        self.assertEqual(sum(x[0]["domain"] == "interest" for x in rows), 34)
        self.assertEqual(sum(x[0]["domain"] == "conversion" for x in rows), 33)
        self.assertEqual(sum(x[0]["domain"] == "logic" for x in rows), 33)

    def test_new_cases_are_deterministic_for_seed(self):
        self.assertEqual(_new_family_cases(SEED), _new_family_cases(SEED))
        self.assertNotEqual(_new_family_cases(SEED),
                            _new_family_cases(SEED + 1))

    def test_interest_answers_have_exact_euro_units(self):
        for row, _ in _new_family_cases()[:34]:
            self.assertEqual(row["kind"], "eur")
            self.assertRegex(row["answer"], r"^-?\d+,\d{2} EUR$")

    def test_bilingual_generation_is_not_entangled_with_ground_truth(self):
        rows = [x[0] for x in _new_family_cases() if x[0]["domain"] == "logic"]
        yes = [r for r in rows if r["answer"] == "1"]
        no = [r for r in rows if r["answer"] == "0"]
        self.assertTrue(any("Given:" in x["prompt"] for x in yes))
        self.assertTrue(any("Gegeben:" in x["prompt"] for x in no))

    def test_exact_conversion_values_not_rounded(self):
        rows = [x[0] for x in _new_family_cases() if x[0]["domain"] == "conversion"]
        self.assertTrue(all(x["kind"] == "integer" for x in rows))
        self.assertTrue(all(x["answer"].lstrip("-").isdigit() for x in rows))

    def test_control_cases_are_50_explicit_v5_style7(self):
        rows = _control_cases()
        self.assertEqual(len(rows), 50)
        self.assertTrue(all(x[1]["stratum"] == "C" for x in rows))
        self.assertTrue(all(x[1]["source"].startswith("v5-style7:")
                            for x in rows))

    def test_perturbations_preserve_oracle(self):
        a, _ = build(self.sources, REVISIONS)
        orig = {x["id"]: x for x in a}
        for case, info in _perturbed_cases([(x, {}) for x in a if x["id"].startswith("A-")]):
            source = info["source"].split(":", 1)[1]
            self.assertEqual(case["answer"], orig[source]["answer"])
            self.assertNotEqual(case["prompt"], orig[source]["prompt"])

    def test_all_cases_have_unique_ids(self):
        cases, _ = build(self.sources, REVISIONS)
        self.assertEqual(len(set(x["id"] for x in cases)), len(cases))

    def test_source_revisions_must_be_full_commit_shas(self):
        with self.assertRaises(ValueError):
            build(self.sources, {**REVISIONS, "mbpp": "main"})

    def test_missing_gsm8k_fails_closed(self):
        self.sources["gsm8k"].pop()
        with self.assertRaises(ValueError):
            build(self.sources, REVISIONS)

    def test_missing_mbpp_tests_fails_closed(self):
        self.sources["mbpp"][0]["test_list"] = []
        with self.assertRaises(ValueError):
            build(self.sources, REVISIONS)

    def test_unscorable_math500_fails_closed(self):
        self.sources["math500"] = [{"problem": "unknown", "answer": r"\sqrt{2}"} for _ in range(60)]
        with self.assertRaises(ValueError):
            build(self.sources, REVISIONS)

    def test_manifest_is_pinned_and_write_once(self):
        cases, meta = build(self.sources, REVISIONS)
        with tempfile.TemporaryDirectory() as path:
            root = Path(path)
            manifest = export(cases, meta, root)
            raw = (root / "neral-ext-400.jsonl").read_bytes()
            self.assertEqual(manifest["dataset_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(len(raw.splitlines()), 400)
            with self.assertRaises(ValueError):
                export(cases, meta, root)

    def test_reject_unparseable_public_reference(self):
        self.sources["gsm8k"][0]["answer"] = "The answer is infinity"
        with self.assertRaises(ValueError):
            build(self.sources, REVISIONS)

    def test_canonical_exact_arithmetic(self):
        self.assertEqual(_canonical_number("3/6"), ("rational", "1/2"))
        self.assertEqual(_canonical_number("3.0"), ("integer", "3"))
        self.assertIsNone(_canonical_number(r"\frac{1}{2}"))
        self.assertIsNone(_canonical_number("1/0"))


class DecontaminationTests(unittest.TestCase):
    def test_normalize_punctuation_casefold(self):
        self.assertEqual(normalize("Äpfel, ÄPFEL!"), ("äpfel", "äpfel"))

    def test_10gram_overlap(self):
        self.assertTrue(shingles("one two three four five six seven eight nine ten") &
                        shingles("one two three four five six seven eight nine ten eleven"))

    def test_no_overlap_is_clean(self):
        rows = [{"id": "new1", "prompt": "independent problem",
                 "kind": "integer", "answer": "2"}]
        meta = [{"id": "new1", "stratum": "A"}]
        report = audit(rows, meta, [{"prompt": "train this model carefully"}])
        self.assertTrue(report["A_B_D_clean"])

    def test_exact_overlap_A_is_flagged(self):
        prompt = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        rows = [{"id": "case-01", "prompt": prompt}]
        meta = [{"id": "case-01", "stratum": "A"}]
        report = audit(rows, meta, [{"prompt": "x " + prompt + " y"}])
        self.assertFalse(report["A_B_D_clean"])
        self.assertEqual(report["overlap_ids"]["A"], ["case-01"])

    def test_C_overlap_is_expected_negative_control(self):
        prompt = "alpha beta gamma delta epsilon zeta eta theta iota kappa"
        rows = [{"id": "case-01", "prompt": prompt}]
        meta = [{"id": "case-01", "stratum": "C"}]
        report = audit(rows, meta, [{"prompt": prompt}])
        self.assertTrue(report["A_B_D_clean"])
        self.assertEqual(report["overlap_counts"]["C"], 1)

    def test_misaligned_metadata_is_rejected(self):
        with self.assertRaises(ValueError):
            audit([{"id": "one", "prompt": "x"}],
                  [{"id": "two", "stratum": "B"}], [{"prompt": "y"}])

    def test_jsonl_loader_rejects_non_dictionary(self):
        with tempfile.TemporaryDirectory() as dir:
            file = Path(dir) / "dataset.jsonl"
            file.write_text("[1,2,3]\n")
            with self.assertRaises(ValueError):
                load_jsonl(file)


if __name__ == "__main__":
    unittest.main()
