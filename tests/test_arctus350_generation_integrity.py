"""Arctus350 generation metrics may not hide missing exact instruction evidence."""
from __future__ import annotations
import unittest
from frontier350.chat350_generation_metrics import (
    compare_health, summarize_generations, repeated_ngram_fraction
)


def health(**overrides):
    result={
        "eos_fraction":0.2,
        "mean_repeated_trigram_fraction":0.1,
        "utf8_fraction":1.0,
        "exact_instruction_pct":50.0,
    }
    result.update(overrides)
    return result


class GenerationIntegrityTests(unittest.TestCase):
    def test_missing_candidate_exact_instruction_metric_never_counts_as_nonregression(self):
        outcome=compare_health(health(),health(exact_instruction_pct=None))
        self.assertFalse(outcome["exact_instruction_no_regression"])
        self.assertFalse(outcome["health_no_regression"])

    def test_absent_baseline_metric_is_not_proof_of_nonregression(self):
        outcome=compare_health(health(exact_instruction_pct=None),health(exact_instruction_pct=None))
        self.assertFalse(outcome["health_no_regression"])

    def test_nonfinite_and_out_of_range_metric_rejected(self):
        for bad in (float("nan"),float("inf"),-1,101,"100",True):
            with self.subTest(metric=bad),self.assertRaises(ValueError):
                compare_health(health(),health(exact_instruction_pct=bad))
        for bad in (float("nan"),float("inf"),-0.1,1.1,"0.5",True):
            with self.subTest(metric=bad),self.assertRaises(ValueError):
                compare_health(health(),health(eos_fraction=bad))

    def test_valid_health_metrics_are_still_comparable(self):
        outcome=compare_health(health(),health(eos_fraction=0.4,exact_instruction_pct=55.0))
        self.assertTrue(outcome["health_no_regression"])
        self.assertTrue(outcome["exact_instruction_no_regression"])

    def test_generation_input_requires_nonnegative_actual_token_ids(self):
        for generated in ([True,123],[3,-1],["4",5],[1.2,3],None,"123"):
            with self.subTest(generated=generated),self.assertRaises(ValueError):
                summarize_generations([{"generated_ids":generated,"text":"hi"}],eos_id=1)

    def test_bad_eos_identity_cannot_be_exploited(self):
        for eos in (True,-1,"1",None):
            with self.subTest(eos=eos),self.assertRaises(ValueError):
                summarize_generations([{"generated_ids":[1,2],"text":"hi"}],eos_id=eos)

    def test_generation_summary_valid_example(self):
        summary=summarize_generations([
            {"generated_ids":[3,4,0],"text":"Antwort","expected_exact":"Antwort"},
            {"generated_ids":[5,6],"text":"Nein","expected_exact":"Antwort"},
        ],eos_id=0)
        self.assertEqual(summary["cases"],2)
        self.assertAlmostEqual(summary["eos_fraction"],0.5)
        self.assertEqual(summary["exact_instruction_pct"],50.0)


    def test_trailing_tokens_after_eos_are_invalid(self):
        # Generation must terminate immediately at EOS; malformed persisted
        # sequences cannot count as clean early stopping.
        for ids in ([6, 0, 7], [6, 0, 0]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                summarize_generations([{"generated_ids": ids, "text": "Antwort"}], eos_id=0)

    def test_immediate_eos_is_a_valid_empty_generation(self):
        summary = summarize_generations(
            [{"generated_ids": [0], "text": ""}], eos_id=0
        )
        self.assertEqual(summary["eos_fraction"], 1.0)
        self.assertEqual(summary["mean_generated_tokens_before_eos"], 0.0)

if __name__=="__main__":
    unittest.main()
