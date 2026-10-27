"""Recorded NERAL model evaluations must preserve real-vs-simulated provenance."""
import unittest
from neral_persona_eval import load_holdout, grade_recorded_run
from neral_persona_ollama_runner import collect_ollama


def fixture(payload):
    return {"model":payload["model"],"done":True,
            "message":{"role":"assistant","content":"fixture"}}


def tags():
    return {"models":[{"name":"qwen3.5:9b","digest":"b"*64}]}


class RunScoringTests(unittest.TestCase):
    def setUp(self):
        self.cases=load_holdout()
        self.run=collect_ollama(self.cases,model="qwen3.5:9b",
                                model_call=fixture,tags_call=tags)

    def test_simulated_run_never_receives_benchmark_points(self):
        result=grade_recorded_run(self.cases,self.run)
        self.assertEqual(result["status"],"not_measured")
        self.assertIsNone(result["accuracy_pct"])
        self.assertEqual(result["run_provenance"],"simulated_test_transport")

    def test_suite_fingerprint_and_digest_must_match(self):
        for field,value in (("suite_sha256","0"*64),("model_digest","invalid"),
                            ("model_id","another:fake")):
            altered={**self.run,"provenance":{**self.run["provenance"],field:value}}
            with self.subTest(field=field),self.assertRaises(ValueError):
                grade_recorded_run(self.cases,altered)

    def test_single_model_identity_required_for_every_response(self):
        corrupted={**self.run,"predictions":dict(self.run["predictions"])}
        key=self.cases[0]["id"]
        corrupted["predictions"][key]={**corrupted["predictions"][key],
                                        "base_model":"qwen3.5:4b"}
        with self.assertRaises(ValueError):
            grade_recorded_run(self.cases,corrupted)

    def test_no_automatic_score_without_genuine_model_generated_marker(self):
        pretending={**self.run,"provenance":{
            **self.run["provenance"],"mode":"local_ollama_http"}}
        result=grade_recorded_run(self.cases,pretending)
        self.assertEqual(result["status"],"not_measured")
        self.assertIsNone(result["accuracy_pct"])


if __name__=="__main__":
    unittest.main()
