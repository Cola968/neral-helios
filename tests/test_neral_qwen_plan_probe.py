"""Synthetic checks for one-shot local Ollama plan availability; no real provider."""
import unittest
from neral_qwen_plan_probe import inspect_probe


class OllamaProbeTests(unittest.TestCase):
    def test_no_model_does_not_call_planner(self):
        calls=[]
        result=inspect_probe(lambda:[],lambda **kwargs:calls.append(kwargs))
        self.assertEqual(calls,[])
        self.assertFalse(result["model_available"])
        self.assertFalse(result["plan_validated"])
        self.assertFalse(result["execution_enabled"])

    def test_local_qwen_model_proposes_only_nonexecuting_steps(self):
        def fake(**kwargs):
            self.assertIn("Flughafen Berlin Brandenburg",kwargs["goal"])
            return {"status":"PROPOSED_UNAPPROVED","source":"external-model",
                    "base_model":"qwen3.5:9b","steps":[{"type":"research",
                    "instruction":"PRIVATE_TEST_INSTRUCTION"}],
                    "execution_enabled":False,"tools_executed":[]}
        result=inspect_probe(lambda:["qwen3.5:9b"],fake)
        self.assertTrue(result["model_available"])
        self.assertTrue(result["plan_validated"])
        self.assertEqual(result["step_count"],1)
        self.assertNotIn("PRIVATE_TEST_INSTRUCTION",str(result))
        self.assertFalse(result["execution_enabled"])

    def test_rejected_model_response_reports_failure_without_outputs(self):
        def bad(**kwargs):
            raise ValueError("Private model error with SECRET_RESPONSE")
        result=inspect_probe(lambda:["qwen3.5:9b"],bad)
        self.assertTrue(result["model_available"])
        self.assertFalse(result["plan_validated"])
        self.assertEqual(result["status"],"STRUCTURED_PLAN_REJECTED")
        self.assertNotIn("SECRET_RESPONSE",str(result))


if __name__=="__main__":
    unittest.main()
