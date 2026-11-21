"""Contracts for the private, non-training NERAL 200-case benchmark."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.neral_v41_blind200 import CATEGORIES, validate_suite


class PrivateBenchmarkContracts(unittest.TestCase):
    def test_generated_suite_is_sealed_and_counts_are_exact(self):
        from scripts.neral_v41_blind200_build import build_suite, norm
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            old=root/"history.jsonl"
            old.write_text(
                json.dumps({"instruction":"Frühere V3.9 Frage?",
                            "meta":{"source":"v39_benchmark"}})+"\n"+
                json.dumps({"instruction":"Andere V3.10 Frage?",
                            "meta":{"source":"live_verified"}})+"\n",
                encoding="utf-8")
            web=[{"title":f"Headline {i} fresh test only",
                  "url":f"https://example.org/news/{i}",
                  "source":"example","published":"2026-11-21","category":"news"}
                 for i in range(20)]
            web += [{"title":f"project-{i}", "url":f"https://example.org/api/{i}",
                     "source":"example", "published":"2026-11-21",
                     "category":"number", "value":str(i*127+11),
                     "repo":f"project-{i}", "metric":"stargazers_count"}
                    for i in range(15)]
            path=root/"sealed200.jsonl"
            with patch('scripts.neral_v41_blind200_build._prior_rows', return_value=(
                {norm('Frühere V3.9 Frage?'), norm('Andere V3.10 Frage?')},
                {'v39_benchmark', 'v310_private_holdout'})):
                info=build_suite(path, archives=[old], web_facts=web)
            self.assertEqual(info["cases"],200)
            self.assertEqual(info["categories"],CATEGORIES)
            self.assertEqual(info["sha256"],validate_suite(path)["sha256"])
            records=[json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(records),200)
            prompts={x["prompt"] for x in records}
            self.assertNotIn("Frühere V3.9 Frage?",prompts)
            self.assertNotIn("Andere V3.10 Frage?",prompts)
            with self.assertRaises(FileExistsError):
                build_suite(path, archives=[old], web_facts=web)

    def test_benchmark_http_opt_out_does_not_read_or_capture_training(self):
        from flask import Flask
        from unittest.mock import patch
        import threading
        from test_neral_three_tier_api import make_runtime
        from neral_three_tier_runtime import register_three_tier_api
        runtime,low,high,_=make_runtime()
        app=Flask("v41-benchmark-no-training")
        register_three_tier_api(app,runtime,busy=threading.Lock())
        client=app.test_client()
        with patch("neral_verified_notes.exact_note_lookup",
                   side_effect=AssertionError("recalled blind task")), \
             patch("neral_verified_training.append_verified_example",
                   side_effect=AssertionError("captured blind task")):
            r=client.post("/api/neral/chat/three-tier",json={
                "tier":"Amber","prompt":"Hallo!",
                "history":[],"benchmark_isolated":True})
        self.assertEqual(r.status_code,200,r.json)
        self.assertIs(r.json.get("benchmark_isolated"),True)
        self.assertEqual(r.json["tier"],"Amber")

    def test_exact_grader_does_not_reward_numbers_buried_in_hallucination(self):
        from scripts.neral_v41_blind200_live import grade_response
        case={"category":"math_logic","grader":"independent_exact",
              "sealed_reference":{"kind":"integer","value":"42"}}
        self.assertEqual(grade_response(case,{"text":"42","verifier_ok":False})[0],"pass")
        self.assertEqual(grade_response(case,{"text":"42 oder 43","verifier_ok":False})[0],"fail")
        self.assertEqual(grade_response(case,{"text":"43","verifier_ok":True})[0],"fail")

    def test_web_provenance_is_not_equal_to_verified_fact(self):
        from scripts.neral_v41_blind200_live import grade_response
        case={"category":"live_web","grader":"source_freshness",
              "sealed_reference":{"kind":"headline","value":"Solar mission arrives",
                                  "url":"https://example.org/solar","source":"example"}}
        self.assertEqual(grade_response(case,{"text":"Solar mission arrives",
                                             "sources":[]})[0],"fail")
        good={"text":"Solar mission arrives","sources":[{"url":"https://example.org/solar",
                                                         "page_read":True}]}
        self.assertEqual(grade_response(case,good)[0],"pass")

    def test_soft_tasks_never_get_automatic_success(self):
        from scripts.neral_v41_blind200_live import grade_response
        case={"category":"writing","grader":"blind_pairwise"}
        self.assertEqual(grade_response(case,{"text":"Ein schöner Text"})[0],"ungraded")
        self.assertEqual(grade_response(case,{"text":""})[0],"fail")


if __name__=="__main__":
    unittest.main()
