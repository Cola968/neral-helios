"""Frozen CPU contracts for a REAL local-app, cross-model fresh 36-case pilot.

Test-first: two installed models, identical generated private questions,
identical measured inference profiles, exact independent oracles, and scalar
report privacy. Not a blinded holdout and never a release-promotion gate.
"""
from __future__ import annotations
import json
import unittest
from scripts.neral_low_high_fresh import generate_fresh_cases
from scripts.neral_crossmodel_fresh_app import (
    MODELS, STATUS, PilotError, summarize_pairs, client_config,
    validate_releases_ready, validate_local_models
)

class CrossModelFreshAppContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=generate_fresh_cases(20261108)

    def observations(self):
        result={}
        for model in MODELS:
            result[model]={}
            for idx,case in enumerate(self.cases):
                correct=idx < 24 if model=="qwen3:8b" else idx >= 6
                value=case["answer"] if correct else case["answer"]+101
                result[model][case["id"]]={
                    "text":f"Worked out.\nFINAL_ANSWER: {value}",
                    "finished":True,"seconds":5.0 if model=="qwen3:8b" else 12.0,
                    "http_error":False
                }
        return result

    def vrams(self):
        return {
            "qwen3:8b":{"peak_used_mib":6890,"total_mib":12282,"sample_count":200},
            "qwen3.5:9b":{"peak_used_mib":7790,"total_mib":12282,"sample_count":400},
        }

    def digests(self):
        return {
            "qwen3:8b":"sha256:"+"a"*64,
            "qwen3.5:9b":"sha256:"+"b"*64,
        }

    def test_suite_has_new_six_families_and_36_cases(self):
        self.assertEqual(len(self.cases),36)
        self.assertEqual(len({c["id"] for c in self.cases}),36)
        self.assertEqual(len({(c["domain"],c["family"]) for c in self.cases}),6)
        self.assertEqual({d:sum(c["domain"]==d for c in self.cases)
                          for d in ("math","reasoning","code_trace")},
                         {"math":12,"reasoning":12,"code_trace":12})

    def test_paired_oracle_score_and_no_promotion(self):
        z=summarize_pairs(self.cases,self.observations(),self.vrams(),self.digests())
        self.assertEqual(z["status"],STATUS)
        self.assertEqual(z["cases_per_model"],36)
        self.assertEqual(z["models"]["qwen3:8b"]["correct"],24)
        self.assertEqual(z["models"]["qwen3.5:9b"]["correct"],30)
        self.assertEqual(z["paired"]["qwen35_only_correct"],12)
        self.assertEqual(z["paired"]["qwen3_only_correct"],6)
        self.assertEqual(z["models"]["qwen3.5:9b"]["sampled_peak_vram_gib"],round(7790/1024,3))
        self.assertFalse(z["independent_holdout_verified"])
        self.assertFalse(z["release_validated"])
        self.assertFalse(z["model_promotion_authorized"])
        self.assertFalse(z["neural_improvement_proven"])
        self.assertEqual(z["models"]["qwen3:8b"]["by_domain"]["math"]["count"],12)

    def test_private_answers_and_seeds_never_enter_scalar_report(self):
        rows=self.observations()
        rows["qwen3.5:9b"][self.cases[0]["id"]]["text"]="SECRET_PRIVATE_ANSWER\nFINAL_ANSWER: 1"
        summary=summarize_pairs(self.cases,rows,self.vrams(),self.digests())
        self.assertNotIn("SECRET_PRIVATE_ANSWER",json.dumps(summary))
        self.assertNotIn("Worked out",json.dumps(summary))
        for case in self.cases:
            self.assertNotIn(case["prompt"],json.dumps(summary))
        self.assertNotIn("20261108",json.dumps(summary))

    def test_missing_pairs_or_vram_refuse_success(self):
        rows=self.observations()
        del rows["qwen3:8b"][self.cases[0]["id"]]
        with self.assertRaises(PilotError):
            summarize_pairs(self.cases,rows,self.vrams(),self.digests())
        with self.assertRaises(PilotError):
            summarize_pairs(self.cases,self.observations(),{},self.digests())

    def test_http_error_and_partial_not_counted(self):
        rows=self.observations()
        cid=self.cases[0]["id"]
        rows["qwen3:8b"][cid]={"text":"","finished":False,"seconds":14.0,"http_error":True}
        z=summarize_pairs(self.cases,rows,self.vrams(),self.digests())
        self.assertEqual(z["models"]["qwen3:8b"]["correct"],23)
        self.assertEqual(z["models"]["qwen3:8b"]["api_failures"],1)
        self.assertEqual(z["models"]["qwen3:8b"]["finished"],35)
        self.assertEqual(z["models"]["qwen3:8b"]["by_domain"]["math"]["correct"],7)

    def test_safe_http_reason_counts_without_private_payloads(self):
        observations=self.observations()
        first,second=self.cases[0]["id"],self.cases[1]["id"]
        observations["qwen3.5:9b"][first]={
            "text":"","finished":False,"seconds":27.0,
            "http_error":True,"error_code":"token_limit"}
        observations["qwen3.5:9b"][second]={
            "text":"","finished":False,"seconds":34.0,
            "http_error":True,"error_code":"timeout"}
        report=summarize_pairs(self.cases,observations,self.vrams(),self.digests())
        self.assertEqual(report["models"]["qwen3.5:9b"]["api_failures"],2)
        self.assertEqual(report["models"]["qwen3.5:9b"]["api_failure_codes"],
                         {"timeout":1,"token_limit":1})
        self.assertEqual(report["models"]["qwen3:8b"]["api_failure_codes"],{})
        self.assertNotIn("PRIVATE",json.dumps(report))
        observations["qwen3.5:9b"][first]["error_code"]="C:/PRIVATE/secret.txt"
        with self.assertRaises(PilotError):
            summarize_pairs(self.cases,observations,self.vrams(),self.digests())

    def test_malformed_response_and_model_digests_fail_closed(self):
        rows=self.observations()
        rows["qwen3:8b"][self.cases[0]["id"]]["seconds"]=-1
        with self.assertRaises(PilotError):
            summarize_pairs(self.cases,rows,self.vrams(),self.digests())
        with self.assertRaises(PilotError):
            summarize_pairs(self.cases,self.observations(),self.vrams(),{"qwen3:8b":"bad"})

    def test_profile_is_identical_and_explicit_for_both_models(self):
        a=client_config("qwen3:8b")
        b=client_config("qwen3.5:9b")
        self.assertEqual(a["profile"],"measured_nonthinking_2048")
        self.assertEqual(b["profile"],a["profile"])
        self.assertEqual(a["token_budget"],2048)
        self.assertEqual(b["think"],False)
        self.assertNotEqual(a["model"],b["model"])

    def test_missing_models_and_release_pending_refuse_start(self):
        with self.assertRaises(PilotError):
            validate_local_models({"qwen3:8b":"sha256:"+"a"*64})
        with self.assertRaises(PilotError):
            validate_releases_ready({"conclusion":"pending"})
        with self.assertRaises(PilotError):
            validate_releases_ready({"conclusion":"failure"})
        self.assertTrue(validate_releases_ready({"conclusion":"success"}))

if __name__=="__main__":
    unittest.main()
