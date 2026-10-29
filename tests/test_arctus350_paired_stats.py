"""RED -> GREEN: exact paired binary confidence for claimed native Arctus observations.

No inference, no pretrained weights, no GPU; all fixtures are synthetic.
"""
import unittest

from frontier350.arctus350_paired_stats import exact_paired_gain
from frontier350.arctus350_paired_evidence import compare_native_runs
from tests.test_arctus350_paired_evidence import fixture


class ExactPairedStatisticsTests(unittest.TestCase):
    def test_one_extra_correct_on_eight_is_not_a_research_breakthrough(self):
        prior=fixture("a"*64)
        candidate=fixture("b"*64)
        candidate["cases"][-1]["correct"]=True
        report=compare_native_runs(prior,candidate)
        self.assertEqual(report["overall"]["gained"],1)
        self.assertEqual(report["overall"]["lost"],0)
        self.assertEqual(report["decision"],"INSUFFICIENT_STATISTICAL_EVIDENCE")
        self.assertEqual(report["statistics"]["p_value_two_sided"],1.0)
        self.assertEqual(report["statistics"]["status"],"INSUFFICIENT_EVIDENCE")
        self.assertFalse(report["statistics"]["supports_gain"])
        self.assertFalse(report["provenance_verified"])
        self.assertEqual(report["release_authorization"],"none")

    def test_exact_p_values_do_not_use_asymptotic_approximation(self):
        self.assertEqual(exact_paired_gain(0,0,32)["p_value_two_sided"],1.0)
        self.assertEqual(exact_paired_gain(5,0,32)["p_value_two_sided"],0.0625)
        self.assertEqual(exact_paired_gain(6,0,32)["p_value_two_sided"],0.03125)
        self.assertEqual(exact_paired_gain(8,2,40)["p_value_two_sided"],0.109375)
        self.assertFalse(exact_paired_gain(8,2,40)["supports_gain"])
        self.assertTrue(exact_paired_gain(6,0,32)["supports_gain"])

    def test_significant_synthetic_gain_remains_review_not_release(self):
        prior=fixture("a"*64)
        candidate=fixture("b"*64)
        # Grow both identical case inventories to exactly 32, maintaining
        # >=2 cases per category and recomputing frozen ID/category digest.
        import hashlib,json
        additions=[{"id":f"heldout-extra-{i}","category":"math",
                    "correct":False,"eos":True} for i in range(24)]
        prior["cases"].extend(additions)
        candidate["cases"].extend([dict(x) for x in additions])
        # 6 gains on previously failed math cases.
        for a in candidate["cases"][8:14]:
            a["correct"]=True
        fingerprint=hashlib.sha256(json.dumps(
            [{"id":v["id"],"category":v["category"]}
             for v in sorted(prior["cases"],key=lambda x:x["id"])],
            sort_keys=True,separators=(",",":")
        ).encode("utf-8")).hexdigest()
        prior["benchmark_sha256"]=candidate["benchmark_sha256"]=fingerprint
        result=compare_native_runs(prior,candidate)
        self.assertEqual(result["decision"],"REVIEW_CANDIDATE")
        self.assertEqual(result["statistics"]["p_value_two_sided"],0.03125)
        self.assertFalse(result["provenance_verified"])
        self.assertEqual(result["release_authorization"],"none")

    def test_stochastic_decoding_cannot_turn_one_draw_into_a_review_candidate(self):
        # Synthetic six-of-32 gain: without repetitions/decoding provenance,
        # a single positive-temperature draw can differ because of sampling.
        import hashlib,json
        prior=fixture("a"*64)
        candidate=fixture("b"*64)
        additions=[{"id":f"heldout-extra-{i}","category":"math",
                    "correct":False,"eos":True} for i in range(24)]
        prior["cases"].extend(additions)
        candidate["cases"].extend([dict(x) for x in additions])
        for a in candidate["cases"][8:14]:
            a["correct"]=True
        fingerprint=hashlib.sha256(json.dumps(
            [{"id":v["id"],"category":v["category"]}
             for v in sorted(prior["cases"],key=lambda x:x["id"])],
            sort_keys=True,separators=(",",":")
        ).encode("utf-8")).hexdigest()
        prior["benchmark_sha256"]=candidate["benchmark_sha256"]=fingerprint
        for temperature in (0.7,0.0001,2):
            with self.subTest(temperature=temperature):
                prior["sampling"]["temperature"]=temperature
                candidate["sampling"]["temperature"]=temperature
                result=compare_native_runs(prior,candidate)
                self.assertEqual(result["decision"],"STOCHASTIC_EXPLORATORY_ONLY")
                self.assertTrue(result["statistics"]["supports_gain"])
                self.assertEqual(result["statistics"]["p_value_two_sided"],0.03125)
                self.assertFalse(result["provenance_verified"])
                self.assertEqual(result["release_authorization"],"none")

    def test_stochastic_sampling_does_not_hide_regression_or_small_sample(self):
        prior=fixture("a"*64)
        candidate=fixture("b"*64)
        prior["sampling"]["temperature"]=candidate["sampling"]["temperature"]=0.7
        candidate["cases"][0]["correct"]=False
        self.assertEqual(compare_native_runs(prior,candidate)["decision"],"REJECT_CATEGORY_REGRESSION")
        candidate=fixture("b"*64)
        candidate["sampling"]["temperature"]=0.7
        candidate["cases"][-1]["correct"]=True
        self.assertEqual(compare_native_runs(prior,candidate)["decision"],"INSUFFICIENT_STATISTICAL_EVIDENCE")

    def test_small_but_perfect_synthetic_gain_still_needs_sufficient_cases(self):
        result=exact_paired_gain(8,0,8)
        self.assertEqual(result["p_value_two_sided"],0.0078125)
        self.assertFalse(result["supports_gain"])
        self.assertEqual(result["status"],"INSUFFICIENT_EVIDENCE")
        self.assertEqual(result["minimum_cases_policy"],32)

    def test_negative_effect_and_ties_cannot_support_improvement(self):
        for gain,loss,total in [(0,8,32),(5,5,32),(0,0,32),(2,8,40)]:
            with self.subTest(gain=gain,loss=loss):
                self.assertFalse(exact_paired_gain(gain,loss,total)["supports_gain"])

    def test_invalid_counts_never_pass_or_silently_coerce(self):
        for args in [(True,0,32),(1,False,32),(-1,0,32),(0,0,0),
                     (2,1,2),(1,1,513),("6",0,32),(0,1,None),
                     (1.0,0,32),(1,0,3.0)]:
            with self.subTest(args=args),self.assertRaises(ValueError):
                exact_paired_gain(*args)

    def test_stats_do_not_override_category_regression(self):
        base=fixture("a"*64)
        cand=fixture("b"*64)
        base["cases"][0]["correct"]=True
        cand["cases"][0]["correct"]=False
        self.assertEqual(compare_native_runs(base,cand)["decision"],"REJECT_CATEGORY_REGRESSION")


if __name__=="__main__":
    unittest.main()
