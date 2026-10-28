"""CPU-only contracts for read-only Arctus350 checkpoint-delta evidence."""
from __future__ import annotations

import unittest
from frontier350.arctus350_weight_delta import classify_weight, summarize_group


class WeightDeltaContractTests(unittest.TestCase):
    def test_parameter_family_is_not_guesswork(self):
        expected = {
            "tok_emb.weight": "embedding",
            "lm_head.weight": "tied_embedding_alias",
            "norm.weight": "rmsnorm",
            "blocks.2.n1.weight": "rmsnorm",
            "blocks.2.attn.qn.weight": "rmsnorm",
            "blocks.3.attn.q.weight": "attention_qkv",
            "blocks.3.attn.k.weight": "attention_qkv",
            "blocks.3.attn.v.weight": "attention_qkv",
            "blocks.3.attn.o.weight": "attention_output",
            "blocks.2.ffn.gate.weight": "ffn_gate_up",
            "blocks.2.ffn.up.weight": "ffn_gate_up",
            "blocks.2.ffn.down.weight": "ffn_down",
        }
        for key, family in expected.items():
            with self.subTest(key=key):
                self.assertEqual(classify_weight(key), family)

    def test_counts_denominator_and_zero_weight_are_valid(self):
        report = summarize_group(
            {"total": 8, "changed": 2, "low_total": 3, "low_changed": 2,
             "high_total": 5, "high_changed": 0, "baseline_sq": 0.0, "delta_sq": 2.0}
        )
        self.assertEqual(report["fraction_changed"], .25)
        self.assertEqual(report["fraction_changed_low_magnitude"], 2/3)
        self.assertEqual(report["fraction_changed_high_magnitude"], 0)
        self.assertIsNone(report["relative_l2"])
        self.assertEqual(report["changed_count"], 2)

    def test_missing_and_contradictory_data_rejected(self):
        valid={"total":3, "changed":1, "low_total":2,"low_changed":1,"high_total":1,
               "high_changed":0,"baseline_sq":9.,"delta_sq":1.}
        for patch in ({"changed":4},{"low_total":4},{"high_changed":2},
                      {"baseline_sq":-1.}, {"delta_sq":float("nan")},
                      {"changed":True}):
            with self.subTest(patch=patch), self.assertRaises(ValueError):
                summarize_group({**valid, **patch})

    def test_no_changes_not_false_success(self):
        x=summarize_group({"total":10,"changed":0, "low_total":4,"low_changed":0,
                           "high_total":6,"high_changed":0,"baseline_sq":16.,"delta_sq":0.})
        self.assertEqual(x["relative_l2"], 0.)
        self.assertEqual(x["fraction_changed"], 0.)


if __name__ == "__main__":
    unittest.main()
