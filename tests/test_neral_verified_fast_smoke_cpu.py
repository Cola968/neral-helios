"""Known-task fast-path GPU smoke: failure phases and latency, not a blind test."""
import unittest
from scripts.neral_verified_fast_smoke import (
    SELECTED_DEV_IDS, select_unchanged_cases, compare_known_cases, _api_one,
)

class FastSmokeContracts(unittest.TestCase):
    def test_only_known_frozen_cases_include_previous_logic_cutoffs(self):
        rows=select_unchanged_cases()
        self.assertEqual(len(rows),6)
        self.assertEqual([x["id"] for x in rows],list(SELECTED_DEV_IDS))
        self.assertIn("blind100-logic-02",SELECTED_DEV_IDS)
        self.assertIn("blind100-logic-06",SELECTED_DEV_IDS)
        self.assertEqual(len(set(SELECTED_DEV_IDS)),6)
        self.assertTrue(all(x["id"].startswith("blind100-") for x in rows))

    def sample(self):
        return [{"id":n,"category":n.split("-")[1],
            "full_latency_ms":10000+i*500,"fast_latency_ms":3000+i*100,
            "full_correct":i!=0,"fast_correct":i!=0,
            "full_verified":i in (3,4),"fast_verified":i in (3,4),
            "full_model_loads":2,"fast_model_loads":1 if i not in (3,4) else 0,
            "full_status":"completed","fast_status":"completed",
            "full_failure_phase":None,"fast_failure_phase":None}
            for i,n in enumerate(SELECTED_DEV_IDS)]

    def test_summary_distinguishes_latency_from_model_intelligence(self):
        observations=self.sample()
        result=compare_known_cases(observations)
        self.assertEqual(result["count"],6)
        self.assertFalse(result["blind_test"])
        self.assertEqual(result["paired_answer_disagreements"],0)
        self.assertLess(result["fast_p50_ms"],result["full_p50_ms"])
        self.assertEqual(result["baseline_model_only_responses"],4)
        self.assertEqual(result["full_model_load_events"],12)
        with self.assertRaises(ValueError):
            compare_known_cases(observations[:5])

    def test_token_limit_503_keeps_stage_and_does_not_fake_load_count(self):
        item=select_unchanged_cases()[1]
        class Response:
            status_code=503
            json={"error_code":"ollama_transport","diagnostic":{
                "reason":"token_limit","http_status":200,"done_reason":"length",
                "inference_phase":"base_controller_fallback",
                "schema_attempts":1,"base_fallback_attempts":1}}
        class Client:
            def post(self,*args,**kwargs):
                return Response()
        result=_api_one(Client(),{},item,False)
        self.assertEqual(result["status"],"token_limit")
        self.assertEqual(result["done_reason"],"length")
        self.assertEqual(result["failure_phase"],"base_controller_fallback")
        self.assertEqual(result["schema_attempts"],1)
        self.assertEqual(result["base_fallback_attempts"],1)
        self.assertIsNone(result["model_loads"])
        self.assertFalse(result["correct"])
        self.assertFalse(result["verified"])

    def test_cpu_summary_counts_failed_phase_without_inventing_loads(self):
        observations=self.sample()
        observations[1].update({
            "full_status":"token_limit","fast_status":"token_limit",
            "full_correct":False,"fast_correct":False,
            "full_verified":False,"fast_verified":False,
            "full_failure_phase":"base_controller_fallback",
            "fast_failure_phase":"base_fast_fallback",
            "full_model_loads":None,"fast_model_loads":None})
        report=compare_known_cases(observations)
        self.assertEqual(report["full_failure_phases"],{"base_controller_fallback":1})
        self.assertEqual(report["fast_failure_phases"],{"base_fast_fallback":1})
        self.assertEqual(report["full_token_limit_cases"],1)
        self.assertEqual(report["fast_token_limit_cases"],1)
        self.assertEqual(report["full_model_load_missing_cases"],1)
        self.assertEqual(report["full_model_load_events"],10)
        self.assertEqual(report["fast_model_load_missing_cases"],1)

    def test_other_http_errors_fail_closed(self):
        item=select_unchanged_cases()[1]
        class Response:
            status_code=500
            json={"error":"unknown"}
        class Client:
            def post(self,*args,**kwargs):
                return Response()
        with self.assertRaisesRegex(RuntimeError,"unexpected.*HTTP"):
            _api_one(Client(),{},item,False)

if __name__=="__main__":
    unittest.main()
