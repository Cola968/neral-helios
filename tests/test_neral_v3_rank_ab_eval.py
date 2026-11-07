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


    def test_paired_scores_track_true_wins_and_regressions(self):
        base = [True, True, True, False, False, False, False, False]
        r16  = [True, False, True, True, True, False, False, False]
        r32  = [True, True, True, True, True, True, True, False]

        def records(flags):
            return [
                {'id': f'case-{i}', 'correct': ok, 'format_valid': ok,
                 'local_raw': 'PRIVATE_MODEL_OUTPUT_MUST_NOT_LEAK'}
                for i, ok in enumerate(flags)
            ]

        rows = {
            'base': {'holdout20': records(base)},
            'rank16_v3': {'holdout20': list(reversed(records(r16)))},
            'rank32_v3': {'holdout20': records(r32)},
        }
        result = summarize(rows, {'holdout20': 8})
        p16 = result['paired_vs_base']['rank16_v3']['holdout20']
        p32 = result['paired_vs_base']['rank32_v3']['holdout20']
        self.assertEqual(p16['improvements'], 2)
        self.assertEqual(p16['regressions'], 1)
        self.assertEqual(p16['net_gain'], 1)
        self.assertEqual(p16['discordant'], 3)
        self.assertEqual(p16['exact_mcnemar_p'], 1.0)
        self.assertEqual(p32['improvements'], 4)
        self.assertEqual(p32['regressions'], 0)
        self.assertEqual(p32['net_gain'], 4)
        self.assertAlmostEqual(p32['exact_mcnemar_p'], 0.125)
        self.assertNotIn('PRIVATE_MODEL_OUTPUT_MUST_NOT_LEAK', repr(result))

    def test_small_development_set_cannot_prove_general_improvement(self):
        from scripts.neral_v3_rank_ab_eval import exact_mcnemar_p
        self.assertAlmostEqual(exact_mcnemar_p(8, 0), 0.0078125)
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(2, 1), 1.0)

    def test_paired_scoring_rejects_missing_case_ids(self):
        base = [{'id': 'a', 'correct': True, 'format_valid': True},
                {'id': 'b', 'correct': False, 'format_valid': True}]
        rows = {key: {'holdout20': [dict(z) for z in base]} for key in MODEL_KEYS}
        rows['rank16_v3']['holdout20'][1]['id'] = 'unpaired'
        with self.assertRaisesRegex(ValueError, 'matched case ids'):
            summarize(rows, {'holdout20': 2})

    def test_paired_scoring_rejects_non_boolean_scores(self):
        cases = [{'id': 'a', 'correct': True, 'format_valid': True}]
        rows = {key: {'holdout20': [dict(z) for z in cases]} for key in MODEL_KEYS}
        rows['rank32_v3']['holdout20'][0]['correct'] = 1
        with self.assertRaisesRegex(ValueError, 'boolean'):
            summarize(rows, {'holdout20': 1})



if __name__=='__main__':
    unittest.main()
