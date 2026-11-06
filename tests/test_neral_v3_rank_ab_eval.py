"""Regression tests for comparison scoring; no model/GPU needed."""
import unittest

from scripts.neral_v3_rank_ab_eval import (
    HOLDOUT, suite_score, summarize, MODEL_KEYS, validate_suite
)


class RankABEvalTests(unittest.TestCase):
    def test_holdout_is_fresh_and_deterministic(self):
        self.assertEqual(len(HOLDOUT),20)
        self.assertEqual(len({x['id'] for x in HOLDOUT}),20)
        self.assertEqual({x['kind'] for x in HOLDOUT},{'numeric','algebra','instruction'})
        validate_suite(HOLDOUT)

    def test_numeric_format_strict(self):
        self.assertEqual(suite_score({'kind':'numeric','expected':'49'},'49'),(True,True))
        self.assertEqual(suite_score({'kind':'numeric','expected':'49'},'49.'),(False,False))
        self.assertEqual(suite_score({'kind':'numeric','expected':'49'},'50'),(False,True))

    def test_algebra_requires_exact_two_lines(self):
        row={'kind':'algebra','expected':'4x = 32\nx = 8'}
        self.assertEqual(suite_score(row,'4x = 32\nx = 8'),(True,True))
        self.assertEqual(suite_score(row,'x = 8'),(False,False))
        self.assertEqual(suite_score(row,'4x = 32; x = 8'),(False,False))

    def test_development_report_does_not_claim_independence(self):
        cases=[{'id':str(i),'correct':True,'format_valid':True} for i in range(20)]
        rows={name:{'holdout20':cases} for name in MODEL_KEYS}
        result=summarize(rows,{'holdout20':20})
        self.assertEqual(result['scores']['rank32_v3']['holdout20']['correct'],20)
        self.assertFalse(result['general_improvement_proven'])
        self.assertFalse(result['weights_promoted'])
        self.assertFalse(result['independent_external_benchmark'])

    def test_incomplete_model_or_suite_rejected(self):
        rows={name:{'holdout20':[]} for name in MODEL_KEYS}
        with self.assertRaises(ValueError):
            summarize(rows,{'holdout20':20})


if __name__=='__main__':
    unittest.main()
