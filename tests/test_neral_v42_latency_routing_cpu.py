"""Regression: normal Amber/Quartz chat must not invoke a 3-pass 9B evidence agent.

CPU-only; no owner prompts, model weights, GPU or network calls.
"""
import unittest
from unittest.mock import patch
from test_neral_three_tier_api import make_runtime


def conversation(prompt):
    return [{"role": "system", "content": "Answer clearly."},
            {"role": "user", "content": prompt}]


class V42ChatLatencyRoutingCPU(unittest.TestCase):
    def test_everyday_question_uses_real_selected_model_not_9b_planner(self):
        runtime, amber, nine, _ = make_runtime()
        runtime.evidence_agent = True
        with patch("neral_evidence_agent.run_evidence_agent",
                   side_effect=AssertionError("Evidence agent must not run for normal conversation")):
            a = runtime.answer(conversation("Warum ist der Himmel blau?"), "Amber")
            q = runtime.answer(conversation("Warum ist der Himmel blau?"), "Quartz")
        self.assertEqual(a["tier"], "Amber")
        self.assertEqual(q["tier"], "Quartz")
        self.assertEqual(a["model"], "qwen3.5:4b")
        self.assertEqual(q["model"], "qwen3.5:9b+neral-v2-full")
        self.assertEqual(len(amber.calls), 1)
        self.assertEqual(len(nine.calls), 1)

    def test_factual_question_with_model_numbers_is_not_a_sandbox_job(self):
        runtime, amber, nine, _ = make_runtime()
        runtime.evidence_agent = True
        with patch("neral_evidence_agent.run_evidence_agent",
                   side_effect=AssertionError("Evidence agent must not run")), \
             patch.object(runtime, "_free_code_loop",
                          side_effect=AssertionError("Aircraft models are not math")):
            result = runtime.answer(
                conversation("Was unterscheidet Airbus A320 und A321 im Passagierbetrieb?"), "Amber")
        self.assertEqual(result["model"], "qwen3.5:4b")
        self.assertEqual(len(amber.calls), 1)
        self.assertEqual(len(nine.calls), 0)

    def test_explicit_sandbox_keeps_independent_agent_path(self):
        runtime, amber, nine, _ = make_runtime()
        runtime.evidence_agent = True
        evidence = {"text": "Ungeprüfte Rechnung", "tool_used": "python_sandbox",
                    "verified": False, "safe_to_promote": False}
        with patch("neral_evidence_agent.run_evidence_agent", return_value=evidence) as agent:
            result = runtime.answer(conversation("Berechne die Anzahl der Schichtpläne."),
                                    "Amber", force_sandbox=True)
        self.assertEqual(result["tier"], "Amber")
        self.assertTrue(agent.called)
        self.assertEqual(agent.call_args.kwargs["force_sandbox"], True)

    def test_plain_chat_preserves_original_amber_768_token_ceiling(self):
        runtime, amber, nine, _ = make_runtime()
        runtime.evidence_agent = True
        with patch("neral_evidence_agent.run_evidence_agent",
                   side_effect=AssertionError("No 9B planner")):
            result = runtime.answer(conversation("Erkläre mir ausführlich die Stadtklimatologie."), "Amber")
        self.assertEqual(result["model"], "qwen3.5:4b")
        self.assertEqual(amber.calls[0]["max_tokens"], 768)
        self.assertFalse(nine.calls)


if __name__ == "__main__":
    unittest.main()
