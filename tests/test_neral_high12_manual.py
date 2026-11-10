"""CPU-only gates for paired HIGH12 candidate; no Ollama/GPU calls."""
from __future__ import annotations
import copy
import hashlib
import unittest

from scripts.neral_high12_manual import select_cases, score_pairs, High12GateError

DIGEST = "sha256:" + "a" * 64
MODEL = "qwen3.5:9b"
PROFILE = "measured_nonthinking_2048"


def fixture(*, raw_correct=7, high_correct=7):
    from scripts.neral_low_high_quality import make_prompt
    cases = select_cases(123456789)
    rows = []
    for i, case in enumerate(cases):
        prompt = make_prompt(case)
        ph = hashlib.sha256(prompt.encode("utf8")).hexdigest()
        row = {"id": case["id"], "prompt_sha256": ph, "arms": {}}
        for arm, correct in (("raw_qwen35", i < raw_correct),
                             ("neral_high", i < high_correct)):
            row["arms"][arm] = {
                "text": ("FINAL_ANSWER: " + str(case["answer"] if correct else
                           case["answer"] + 1111)),
                "complete": True, "failed": False, "seconds": 1.25,
                "model": MODEL, "model_digest": DIGEST,
                "profile": PROFILE,
                "tool_used": None,
            }
        rows.append(row)
    return cases, rows


class PairedSelectionTests(unittest.TestCase):
    def test_12_balanced_six_families_same_seed(self):
        a = select_cases(31415926)
        b = select_cases(31415926)
        self.assertEqual(a, b)
        self.assertEqual(len(a), 12)
        self.assertEqual(len({x["id"] for x in a}), 12)
        for domain in ("math", "reasoning", "code_trace"):
            chosen = [c for c in a if c["domain"] == domain]
            self.assertEqual(len(chosen), 4)
            self.assertEqual(len(set(c["family"] for c in chosen)), 2)
            self.assertEqual(sorted([sum(c["family"] == f for c in chosen)
                                     for f in set(c["family"] for c in chosen)]),
                             [2, 2])

    def test_different_seeds_change_inputs(self):
        a, b = select_cases(31415926), select_cases(31415927)
        self.assertNotEqual([x["prompt"] for x in a], [x["prompt"] for x in b])

    def test_invalid_seed_rejected(self):
        for seed in (True, -1, 2**64, "31415926"):
            with self.subTest(seed=seed), self.assertRaises(High12GateError):
                select_cases(seed)


class PairedRatchetTests(unittest.TestCase):
    def test_equal_7_7_is_development_only_pass(self):
        cases, rows = fixture()
        result = score_pairs(cases, rows, model_digest=DIGEST)
        self.assertEqual(result["raw_correct"], 7)
        self.assertEqual(result["high_correct"], 7)
        self.assertTrue(result["development_gate_passed"])
        self.assertFalse(result["release_approval"])
        self.assertFalse(result["independent_holdout_verified"])
        self.assertEqual(result["paired_losses"], 0)

    def test_lower_high_fails_gate_with_losses(self):
        cases, rows = fixture(raw_correct=8, high_correct=5)
        result = score_pairs(cases, rows, model_digest=DIGEST)
        self.assertFalse(result["development_gate_passed"])
        self.assertEqual(result["paired_wins"], 0)
        self.assertEqual(result["paired_losses"], 3)

    def test_high_improvement_reports_paired_wins_not_release(self):
        cases, rows = fixture(raw_correct=6, high_correct=8)
        result = score_pairs(cases, rows, model_digest=DIGEST)
        self.assertEqual(result["paired_wins"], 2)
        self.assertEqual(result["paired_losses"], 0)
        self.assertTrue(result["development_gate_passed"])
        self.assertFalse(result["release_approval"])

    def test_incomplete_or_unpaired_samples_fail_closed(self):
        cases, rows = fixture()
        for broken in (rows[:-1], rows + rows[:1],
                       [dict(rows[0], id="other")] + rows[1:]):
            with self.assertRaises(High12GateError):
                score_pairs(cases, broken, model_digest=DIGEST)

    def test_untrusted_profile_digest_prompt_hash_and_model_fail(self):
        cases, rows = fixture()
        for field, value in (("prompt_sha256", "0"*64),
                             ("model", "qwen3:8b"),
                             ("model_digest", "sha256:" + "b"*64),
                             ("profile", "legacy")):
            with self.subTest(field=field):
                changed = copy.deepcopy(rows)
                obs = changed[0] if field == "prompt_sha256" else changed[0]["arms"]["neral_high"]
                obs[field] = value
                with self.assertRaises(High12GateError):
                    score_pairs(cases, changed, model_digest=DIGEST)

    def test_high_errors_are_not_a_pass(self):
        cases, rows = fixture()
        rows[0]["arms"]["neral_high"].update(text="",complete=False,failed=True)
        result = score_pairs(cases, rows, model_digest=DIGEST)
        self.assertFalse(result["development_gate_passed"])

    def test_no_prompts_or_answers_in_published_report(self):
        cases, rows = fixture()
        report = score_pairs(cases, rows, model_digest=DIGEST)
        self.assertNotIn(cases[0]["prompt"], str(report))
        self.assertNotIn("FINAL_ANSWER", str(report))
        self.assertNotIn("text", str(report))
        self.assertEqual(report["by_domain"]["math"]["count"], 4)


if __name__ == "__main__":
    unittest.main()
