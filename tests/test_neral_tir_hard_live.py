"""Fresh hard36 TIR development-case and scalar accounting contracts."""
import unittest
from collections import Counter

from scripts.neral_tir_hard_live import generate_hard_cases, validated_tool_counts


class HardPilotContracts(unittest.TestCase):
    def test_six_harder_families_and_balanced_domains(self):
        cases = generate_hard_cases(20261108)
        self.assertEqual(len(cases), 36)
        self.assertEqual(len({row["id"] for row in cases}), 36)
        self.assertEqual(dict(Counter(c["domain"] for c in cases)),
                         {"math": 12, "reasoning": 12, "code_trace": 12})
        self.assertEqual(len({(c["domain"], c["family"]) for c in cases}), 6)
        self.assertEqual(len({c["prompt"] for c in cases}), 36)
        self.assertTrue(all(type(c["answer"]) is int and c["kind"] == "integer"
                            for c in cases))
        self.assertTrue(any("UTC" in c["prompt"] for c in cases))
        self.assertTrue(any("off-by-one" in c["prompt"] for c in cases))

    def test_fresh_instances_not_same_prompt_for_other_seed(self):
        a = generate_hard_cases(20261108)
        b = generate_hard_cases(20261109)
        self.assertNotEqual([r["prompt"] for r in a],
                            [r["prompt"] for r in b])

    def test_counters_reject_faked_or_inconsistent_success(self):
        self.assertEqual(validated_tool_counts({"tool_calls_triggered": 2,
                                                "tool_calls_succeeded": 1}),
                         (2, 1))
        for item in ({"tool_calls_triggered": 1, "tool_calls_succeeded": 2},
                     {"tool_calls_triggered": True, "tool_calls_succeeded": 1},
                     {"tool_calls_triggered": -1, "tool_calls_succeeded": 0},
                     {"tool_calls_triggered": 3, "tool_calls_succeeded": 0}):
            with self.subTest(item=item), self.assertRaises(ValueError):
                validated_tool_counts(item)


if __name__ == "__main__":
    unittest.main()
