"""CPU regression tests for read-only equation strategy diagnostic (no training)."""
from fractions import Fraction
import unittest

from scripts.neral_v3_equation_probe import (
    derive_equations, build_variants, grade_reply, summarize, MODES, MODELS
)


class EquationProbeTests(unittest.TestCase):
    def test_frozen_eight_cases_and_six_known_failures(self):
        equations = derive_equations()
        self.assertEqual(len(equations), 8)
        self.assertEqual([x['id'] for x in equations], [f'novel-{i:03d}' for i in range(41, 49)])
        expected = {
            'novel-041': '53', 'novel-043': '8', 'novel-044': '31',
            'novel-045': '19', 'novel-046': '41', 'novel-048': '42',
        }
        for row in equations:
            if row['id'] in expected:
                self.assertEqual(row['expected'], expected[row['id']])
        self.assertEqual(sum(x['id'] in expected for x in equations), 6)

    def test_modes_preserve_original_and_never_leak_solutions(self):
        for case in derive_equations():
            variants = build_variants(case)
            self.assertEqual(set(variants), set(MODES))
            self.assertEqual(variants['direct'], case['prompt'])
            self.assertIn('Zwischenschritt', variants['two_step'])
            self.assertIn('Nur', variants['silent_steps'])
            self.assertNotIn('x = '+case['expected'], variants['two_step'])

    def test_numeric_accuracy_and_format_are_separate(self):
        equation = derive_equations()[0]
        self.assertEqual(grade_reply(equation, 'direct', '53'), (True, True, False))
        self.assertEqual(grade_reply(equation, 'direct', 'x = 53'), (True, False, False))
        self.assertEqual(grade_reply(equation, 'silent_steps', '54'), (False, True, False))
        self.assertEqual(grade_reply(equation, 'direct', '6x = 318; x = 53'), (True, False, False))

    def test_two_step_requires_correct_algebra_and_result(self):
        equation = derive_equations()[0]  # 6x + (-32) = 286
        self.assertEqual(grade_reply(equation, 'two_step', '6x = 318\nx = 53'), (True, True, True))
        self.assertEqual(grade_reply(equation, 'two_step', '6x = 320\nx = 53'), (True, False, False))
        self.assertEqual(grade_reply(equation, 'two_step', '6x = 318\nx = 54'), (False, False, True))
        self.assertEqual(grade_reply(equation, 'two_step', 'x = 53'), (True, False, False))

    def test_summary_counts_are_exact_and_never_authorize_promotion(self):
        cases = derive_equations()
        data = {}
        for model in MODELS:
            data[model] = {}
            for mode in MODES:
                data[model][mode] = [
                    {'id': x['id'], 'correct': True, 'format_valid': True,
                     'algebra_valid': mode == 'two_step', 'answer_sha256': '0'*64}
                    for x in cases
                ]
        result = summarize(data, cases)
        self.assertEqual(result['models']['new_rank16_v2']['direct']['correct'], 8)
        self.assertEqual(result['cases'], 8)
        self.assertFalse(result['general_improvement_proven'])
        self.assertFalse(result['weights_promoted'])
        self.assertFalse(result['training_performed'])
        with self.assertRaises(ValueError):
            summarize({'base': {}}, cases)


if __name__ == '__main__':
    unittest.main()
