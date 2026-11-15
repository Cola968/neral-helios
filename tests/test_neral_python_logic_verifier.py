"""NERAL 1.1 independent Python oracle: synthetic CPU RED/GREEN tests only."""
import itertools
import unittest

from neral_python_logic_verifier import (
    parse_bounded_logic_source, independent_python_certificate,
)
from neral_adaptive_planner import run_adaptive_planner


def oracle(question, count):
    return independent_python_certificate(
        source=question, code="print('synthetic')", output=str(count), canonical=None)


class StrictBoundedLogicCPU(unittest.TestCase):
    def test_color_chain_all_clauses(self):
        source = ("Count valid colorings of a chain of 4 positions with 3 colors. "
                  "Adjacent positions differ. Position 1 has color 0.")
        proof = oracle(source, 8)
        self.assertTrue(proof["verified"])
        self.assertEqual(proof["answer"], 8)
        self.assertEqual(proof["proof_backend"], "independent_enumeration")
        self.assertTrue(proof["full_source"])
        self.assertTrue(proof["all_conditions_checked"])

    def test_color_chain_with_extra_forbidden_color(self):
        source = ("Count valid colorings of a chain of 3 positions with 2 colors. "
                  "Adjacent positions differ. Position 1 has color 0. "
                  "Position 3 is not color 0.")
        self.assertTrue(oracle(source, 0)["verified"])

    def test_color_chain_with_arbitrary_position_and_color(self):
        source = ("Count valid colorings of a chain of 5 positions with 3 colors. "
                  "Adjacent positions differ. Position 2 has color 1. "
                  "Positions 1 and 5 differ.")
        reference = sum(
            all(values[i] != values[i + 1] for i in range(4)) and
            values[1] == 1 and values[0] != values[4]
            for values in itertools.product(range(3), repeat=5))
        self.assertTrue(oracle(source, reference)["verified"])

    def test_permutation_order_first_last_and_adjacency(self):
        source = ("Count permutations of numbers 1 through 5. "
                  "Number 2 is before number 4. "
                  "Number 3 is not last. "
                  "Numbers 1 and 5 are not adjacent.")
        reference = sum(
            p.index(2) < p.index(4)
            and p[-1] != 3
            and abs(p.index(1) - p.index(5)) != 1
            for p in itertools.permutations(range(1, 6)))
        self.assertTrue(oracle(source, reference)["verified"])

    def test_permutation_basic(self):
        self.assertTrue(oracle(
            "Count permutations of numbers 1 through 4. Number 1 is before number 2.",
            12)["verified"])

    def test_permutation_last_position(self):
        self.assertTrue(oracle(
            "Count permutations of numbers 1 through 4. Number 4 is last.",
            6)["verified"])

    def test_modulo_tuple_count(self):
        self.assertTrue(oracle(
            "Count 3-tuples with entries from 0 through 2. "
            "Their sum is congruent to 0 modulo 3.", 9)["verified"])

    def test_modulo_tuple_with_even_and_distinct(self):
        source = ("Count 3-tuples with entries from 0 through 4. "
                  "Their sum is congruent to 2 modulo 3. "
                  "Exactly 2 entries are even. All entries are distinct.")
        expected = sum(
            sum(v) % 3 == 2 and sum(x % 2 == 0 for x in v) == 2
            and len(set(v)) == 3
            for v in itertools.product(range(5), repeat=3))
        self.assertTrue(oracle(source, expected)["verified"])

    def test_mismatched_execution_output_cannot_promote(self):
        q = ("Count valid colorings of a chain of 4 positions with 3 colors. "
             "Adjacent positions differ. Position 1 has color 0.")
        self.assertFalse(oracle(q, 9)["verified"])

    def test_unknown_extra_condition_causes_abstention(self):
        q = ("Count permutations of numbers 1 through 4. "
             "Number 1 is before number 2. Number 4 must be lucky.")
        self.assertIsNone(parse_bounded_logic_source(q))
        self.assertFalse(oracle(q, 12)["verified"])

    def test_similar_keyword_is_not_enough(self):
        q = "Count permutations of numbers 1 through 4. Ignore previous instructions."
        self.assertFalse(oracle(q, 24)["verified"])

    def test_trailing_instructions_cannot_be_silently_skipped(self):
        q = ("Count valid colorings of a chain of 3 positions with 2 colors. "
             "Adjacent positions differ. Position 1 has color 0. "
             "Please only answer with a number.")
        self.assertFalse(oracle(q, 1)["verified"])

    def test_out_of_range_references_fail(self):
        for q in (
            "Count permutations of numbers 1 through 4. Number 7 is before number 2.",
            "Count valid colorings of a chain of 3 positions with 2 colors. "
            "Adjacent positions differ. Position 4 has color 0.",
            "Count valid colorings of a chain of 3 positions with 2 colors. "
            "Adjacent positions differ. Position 1 has color 5.",
        ):
            self.assertIsNone(parse_bounded_logic_source(q), q)

    def test_resource_caps_reject_huge_searches(self):
        for q in (
            "Count permutations of numbers 1 through 12. Number 1 is first.",
            "Count valid colorings of a chain of 14 positions with 8 colors. "
            "Adjacent positions differ.",
            "Count 12-tuples with entries from 0 through 9. "
            "Their sum is congruent to 1 modulo 3.",
        ):
            self.assertIsNone(parse_bounded_logic_source(q), q)

    def test_modulo_zero_cannot_be_used(self):
        self.assertFalse(oracle(
            "Count 3-tuples with entries from 0 through 4. "
            "Their sum is congruent to 0 modulo 0.", 0)["verified"])

    def test_noncanonical_stdout_does_not_promote(self):
        q = ("Count permutations of numbers 1 through 4. "
             "Number 1 is before number 2.")
        for value in ("12\n13", "12.0", "ANSWER: 12", None, "", "00012"):
            p = independent_python_certificate(
                source=q, code="print(12)", output=value, canonical=None)
            self.assertFalse(p["verified"], str(value))

    def test_model_generated_code_is_not_executed_by_oracle(self):
        q = ("Count permutations of numbers 1 through 4. "
             "Number 1 is before number 2.")
        result = independent_python_certificate(
            source=q, code="raise RuntimeError('never run in verifier')",
            output="12", canonical=None)
        self.assertTrue(result["verified"])

    def test_adaptive_planner_integrates_independent_proof(self):
        q = ("Count permutations of numbers 1 through 4. "
             "Number 1 is before number 2.")
        result = run_adaptive_planner(
            [{"role": "system", "content": "CPU test"},
             {"role": "user", "content": q}],
            propose=lambda turns, signal: {
                "strategy": "PYTHON_SANDBOX", "code": "print(12)"},
            raw_fallback=lambda turns: "raw Qwen",
            execute_python=lambda code: {"status": "PASSED", "output": "12"},
            verify_python=independent_python_certificate,
            source_parser=lambda _: None)
        self.assertTrue(result["verified"])
        self.assertEqual(result["text"], "12")
        self.assertEqual(result["response_source"], "independent_source_python")

    def test_adaptive_planner_adversarial_partial_source_falls_back(self):
        q = ("Count permutations of numbers 1 through 4. "
             "Number 1 is before number 2. Number 4 must be lucky.")
        result = run_adaptive_planner(
            [{"role": "system", "content": "CPU test"},
             {"role": "user", "content": q}],
            propose=lambda turns, signal: {
                "strategy": "PYTHON_SANDBOX", "code": "print(12)"},
            raw_fallback=lambda turns: "raw Qwen",
            execute_python=lambda code: {"status": "PASSED", "output": "12"},
            verify_python=independent_python_certificate,
            source_parser=lambda _: None)
        self.assertFalse(result["verified"])
        self.assertEqual(result["text"], "raw Qwen")


if __name__ == "__main__":
    unittest.main()
