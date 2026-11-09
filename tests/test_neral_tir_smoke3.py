"""Three-task C-only opt-in GPU smoke is unit-tested without GPU."""
import unittest
from unittest.mock import patch

from scripts.neral_tir_smoke3 import TASKS, run_smoke_three


class ThreeCaseSmokeContracts(unittest.TestCase):
    def test_three_cases_only_and_counts_real_c_tool_call_packets(self):
        observed = []
        def fake(arm, question, backend, client, headers):
            observed.append((arm, question))
            return "FINAL_ANSWER: 3",True,False,0.1,"neral_python_sandbox",1,1,None
        with patch("scripts.neral_tir_smoke3.run_one_with_backoff",side_effect=fake):
            result = run_smoke_three(None,None,{})
        self.assertEqual(len(TASKS),3)
        self.assertEqual(len(observed),3)
        self.assertEqual([arm for arm,_ in observed],["neral_high_router"]*3)
        self.assertEqual(result["http_503"],0)
        self.assertEqual(result["tool_calls_triggered"],3)
        self.assertEqual(result["tool_calls_succeeded"],3)
        self.assertTrue(result["ready_for_larger_development_test"])

    def test_zero_tools_or_terminal_error_blocks_any_promotion(self):
        with patch("scripts.neral_tir_smoke3.run_one_with_backoff",
                   return_value=("",False,True,1.0,None,0,0,"token_limit")):
            result = run_smoke_three(None,None,{})
        self.assertEqual(result["http_503"],3)
        self.assertEqual(result["tool_calls_triggered"],0)
        self.assertFalse(result["ready_for_larger_development_test"])


    def test_safe_per_attempt_ollama_http_and_thinking_diagnostics(self):
        class Backend:
            inference_diagnostic_events = [{
                "endpoint": "/api/chat",
                "model": "qwen3.5:9b",
                "request_keys": ["keep_alive", "messages", "model", "options", "stream", "think"],
                "option_keys": ["num_ctx", "num_predict", "top_k"],
                "think": False, "keep_alive": "5m",
                "ollama_http_status": 400,
                "ollama_error_type": "invalid_option",
                "content_length": 0, "thinking_length": 31,
                "done_reason": None,
                "private_prompt": "DO_NOT_UPLOAD_SECRET",
            }]
        with patch("scripts.neral_tir_smoke3.run_one_with_backoff",
                   return_value=("",False,True,1.0,None,0,0,"model_error")):
            result = run_smoke_three(Backend(),None,{})
        self.assertEqual(result["ollama_http_status_counts"], {"400":1})
        self.assertEqual(result["ollama_error_type_counts"], {"invalid_option":1})
        self.assertEqual(result["ollama_attempts"][0]["content_length"],0)
        self.assertEqual(result["ollama_attempts"][0]["thinking_length"],31)
        self.assertNotIn("DO_NOT_UPLOAD_SECRET",repr(result))
        self.assertEqual(result["http_503"],3)


if __name__=="__main__":
    unittest.main()
