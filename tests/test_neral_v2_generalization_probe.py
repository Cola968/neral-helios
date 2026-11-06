"""Focused CPU-only contracts for additional generalization diagnostic."""
import unittest
from scripts.neral_v2_generalization_probe import CASES, checked_cases, grade, summary

class GeneralizationProbeTests(unittest.TestCase):
    def test_cases_are_diverse_and_unique(self):
        self.assertEqual(len(checked_cases()),36)
        self.assertEqual(len({x[0] for x in CASES}),36)
        self.assertEqual(sum(x[1]=='math' for x in CASES),24)
        self.assertEqual(sum(x[1]=='instruction' for x in CASES),12)

    def test_grading_does_not_count_explanations_as_correct_numbers(self):
        self.assertFalse(grade('Das Ergebnis ist 42.','math','42')['correct'])
        self.assertTrue(grade('42','math','42')['correct'])
        self.assertTrue(grade('0,5','math','1/2')['correct'])
        self.assertTrue(grade('-4','math','-4')['correct'])
        self.assertFalse(grade('42\n42','math','42')['format_valid'])

    def test_instruction_requires_exact_content(self):
        self.assertTrue(grade(' BEREIT ','instruction','BEREIT')['correct'])
        self.assertFalse(grade('BEREIT.','instruction','BEREIT')['correct'])
        self.assertFalse(grade('bereit','instruction','BEREIT')['correct'])

    def test_summary_requires_three_variants_no_release(self):
        rows=[{'id':cid,'correct':True,'format_valid':True} for cid,*_ in CASES]
        result=summary({x:rows for x in ('base','old_rank32','new_rank16_v2')})
        self.assertEqual(result['cases'],36)
        self.assertEqual(result['scores']['base']['by_type']['math']['correct'],24)
        self.assertFalse(result['independent_holdout'])
        self.assertFalse(result['general_improvement_proven'])
        self.assertFalse(result['weights_promoted'])

    def test_summary_rejects_missing_cases(self):
        with self.assertRaises(ValueError):
            summary({'base':[],'old_rank32':[],'new_rank16_v2':[]})

if __name__=='__main__':unittest.main()