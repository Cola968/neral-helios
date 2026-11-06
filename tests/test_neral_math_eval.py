import unittest
from scripts.neral_math_eval import CASES, normalized_number, summary

class MathEvalTests(unittest.TestCase):
    def test_frozen_nonempty_unique_cases(self):
        self.assertGreaterEqual(len(CASES),12)
        self.assertEqual(len({x['id'] for x in CASES}),len(CASES))
    def test_numeric_format_no_coercion_of_explanation(self):
        self.assertEqual(normalized_number(' 42 '),'42')
        self.assertEqual(normalized_number('2/3'),'2/3')
        self.assertEqual(normalized_number('3,5'),'3.5')
        self.assertIsNone(normalized_number('Ergebnis: 42'))
        self.assertIsNone(normalized_number('4 + 2 = 6'))
        self.assertIsNone(normalized_number('6\nNoch ein Satz'))
    def test_report_labels_correct_not_overclaimed(self):
        results={x:[{'id':case['id'],'correct':i == 1,'format_valid':i == 1} for i,case in enumerate(CASES)]
                 for x in ('base','old_rank32','new_math_rank16')}
        s=summary(results)
        self.assertEqual(s['scores']['base']['correct'],1)
        self.assertEqual(s['scores']['new_math_rank16']['total'],len(CASES))
        self.assertFalse(s['general_improvement_proven'])
        self.assertEqual(s['status'],'SMALL_MATH_DEV_DIAGNOSTIC')
if __name__=='__main__':unittest.main()

