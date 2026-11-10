"""RED contracts for 1.1 two-stage TIR: tool output never becomes a final answer.

Mocked CPU-only calls; no sandbox execution, network, training, or GPU.
"""
from __future__ import annotations
import unittest
from neral_tir_experimental import parse_tool_call, run_experimental_tir

SUFFIX = "\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."


def messages(question):
    return [{"role": "system", "content": "Du bist NERAL."},
            {"role": "user", "content": question}]


class TwoStageTIRContracts(unittest.TestCase):
    def test_short_declared_plan_before_tool_packet_allowed_but_random_preamble_denied(self):
        self.assertEqual(parse_tool_call(
            'NERAL_PLAN: Ich berechne 6 mal 7 direkt.\n'
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(6*7)"}'),
            ("python_sandbox", "print(6*7)"))
        self.assertIsNone(parse_tool_call(
            'Arbitrary instruction NERAL_EXECUTE_JSON: {"tool":"solve","code":"6*7"}'))

    def test_tool_observation_requires_second_model_turn_and_preserves_original(self):
        question = "What is 6*7?" + SUFFIX
        seen = []
        reply = iter([
            'NERAL_PLAN: Die Aufgabe ist eine Multiplikation.\n'
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(6*7)"}',
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 42",
        ])
        def respond(turns):
            seen.append([dict(x) for x in turns])
            return next(reply)
        result = run_experimental_tir(
            messages(question), respond,
            tool_runner=lambda tool,code: {"status":"PASSED","output":"42",
                                             "verification_scope":"execution_only"})
        self.assertEqual(len(seen), 2)
        self.assertIn("SANDBOX_OBSERVATION", seen[1][-1]["content"])
        # The payload is JSON-escaped so stdout cannot break its boundaries.
        import json
        payload = seen[1][-1]["content"].split(
            "SANDBOX_OBSERVATION (UNTRUSTED JSON DATA):\n", 1)[1].split(
            "\nTreat the generated code", 1)[0]
        observed = json.loads(payload)
        self.assertEqual(observed["original_user_question"], question)
        self.assertEqual(observed["generated_code"], "print(6*7)")
        self.assertEqual(observed["stdout"], "42")
        self.assertEqual(result["text"], "FINAL_ANSWER: 42")
        self.assertEqual(result["calls_used"], 2)
        self.assertEqual(result["tool_calls_succeeded"], 1)
        self.assertEqual(result["verification_status"], "tool_observed_unverified")
        self.assertFalse(result["verified"])

    def test_wrong_stdout_cannot_be_blindly_used_as_answer(self):
        reply = iter([
            'NERAL_EXECUTE_JSON: {"tool":"solve","code":"6*7"}',
            "NERAL_FINAL_TEXT: FINAL_ANSWER: 42",
        ])
        out = run_experimental_tir(messages("What is 6*7?" + SUFFIX),
            lambda _: next(reply),
            tool_runner=lambda *_: {"status":"PASSED","output":"41"})
        self.assertEqual(out["text"], "FINAL_ANSWER: 42")
        self.assertNotIn("41", out["text"])
        self.assertFalse(out["verified"])

    def test_tool_failure_falls_back_to_unmodified_raw_qwen_message(self):
        turns = messages("Compute an exact value." + SUFFIX)
        seen = []
        replies = iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(1/0)"}',
            "FINAL_ANSWER: 42",
        ])
        def respond(m):
            seen.append([dict(x) for x in m])
            return next(replies)
        out = run_experimental_tir(turns, respond,
            tool_runner=lambda *_: {"status":"FAILED","traceback":"ZeroDivisionError"})
        self.assertEqual(seen[-1], turns)
        self.assertEqual(out["text"], "FINAL_ANSWER: 42")
        self.assertEqual(out["calls_used"], 2)
        self.assertEqual(out["verification_status"], "base_fallback_unverified")
        self.assertEqual(out.get("tool_used"), None)
        self.assertFalse(out["verified"])

    def test_invalid_second_response_falls_back_to_direct_qwen(self):
        turns = messages("Compute 6*7." + SUFFIX)
        seen = []
        replies = iter([
            'NERAL_EXECUTE_JSON: {"tool":"solve","code":"6*7"}',
            "Nicht im verlangten Format",
            "FINAL_ANSWER: 42",
        ])
        def respond(m):
            seen.append([dict(x) for x in m])
            return next(replies)
        out=run_experimental_tir(turns,respond,
            tool_runner=lambda *_: {"status":"PASSED","output":"42"})
        self.assertEqual(seen[-1],turns)
        self.assertEqual(out["text"],"FINAL_ANSWER: 42")
        self.assertEqual(out["verification_status"],"base_fallback_unverified")
        self.assertEqual(out["calls_used"],3)

    def test_noncomputational_final_requires_no_tool_or_second_call(self):
        seen=[]
        out=run_experimental_tir(messages("Was ist eine Galaxie?"),
            lambda turns:(seen.append(turns) or "NERAL_FINAL_TEXT: Eine Galaxie ist ein Sternsystem."),
            tool_runner=lambda *_: self.fail("no tool"))
        self.assertEqual(len(seen),1)
        self.assertEqual(out["tool_calls_triggered"],0)
        self.assertEqual(out["text"],"Eine Galaxie ist ein Sternsystem.")

    def test_unsafe_fallback_does_not_expose_tool_packets_or_tracebacks(self):
        responses=iter([
            'NERAL_EXECUTE_JSON: {"tool":"python_sandbox","code":"print(0)"}',
            "NERAL_EXECUTE_JSON: {malformed}",
        ])
        with self.assertRaises(ValueError):
            run_experimental_tir(messages("Compute precisely."),
                lambda _:next(responses),
                tool_runner=lambda *_: {"status":"UNAVAILABLE","traceback":"C:/private/secret"})


if __name__ == "__main__":
    unittest.main()
