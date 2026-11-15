"""CPU-only contracts for NERAL 1.1; all examples are synthetic."""
import unittest
from neral_adaptive_planner import (
    deterministic_prefilter, run_adaptive_planner, run_python_in_docker,
    MAX_REPAIRS,
)


class AdaptivePlannerCPU(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "Synthetic test."},
            {"role": "user", "content": "Count the combinations of five objects."},
        ]
        self.original = [dict(x) for x in self.messages]
        self.feedback = []
        self.fallback_calls = []

    def raw(self, turns):
        self.fallback_calls.append([dict(x) for x in turns])
        return "Original Qwen answer"

    def proposer(self, turns, signal):
        self.assertEqual(turns, self.original)
        self.feedback.append(signal)
        return {"strategy": "PYTHON_SANDBOX", "code": "print(5)"}

    def exercise(self, **kw):
        return run_adaptive_planner(
            self.messages,
            kw.pop("propose", self.proposer),
            kw.pop("raw_fallback", self.raw),
            kw.pop("execute_python", lambda code: {
                "status": "PASSED", "output": "5", "verification_scope": "execution_only"}),
            kw.pop("verify_python", lambda **args: {"verified": False}),
            kw.pop("verify_z3", lambda **args: {"verified": False}),
            source_parser=kw.pop("source_parser", lambda q: None),
            **kw,
        )

    def test_direct_does_not_make_model_call(self):
        self.messages[-1]["content"] = "What is a galaxy?"
        answer = self.exercise(propose=lambda *args: self.fail("No planning turn for DIRECT"))
        self.assertEqual(answer["text"], "Original Qwen answer")
        self.assertEqual(answer["planner_route"], "DIRECT")

    def test_prefilter_not_tuned_to_case_ids(self):
        self.assertEqual(deterministic_prefilter("Count permutations", False), "PYTHON_SANDBOX")
        self.assertEqual(deterministic_prefilter("Wieviele Kombinationen?", False), "PYTHON_SANDBOX")
        self.assertEqual(deterministic_prefilter("Unrelated chat", False), "DIRECT")
        self.assertEqual(deterministic_prefilter("Unrelated chat", True), "Z3_SCHEMA")

    def test_sandbox_execution_only_never_promotes(self):
        answer = self.exercise()
        self.assertFalse(answer["verified"])
        self.assertEqual(answer["text"], "Original Qwen answer")
        self.assertEqual(self.fallback_calls, [self.original])

    def test_model_claimed_verified_flag_is_ignored(self):
        answer = self.exercise(propose=lambda *args: {
            "strategy": "PYTHON_SANDBOX", "code": "print(5)",
            "verified": True, "safe_to_promote": True,
            "all_conditions_checked": True,
        })
        self.assertFalse(answer["verified"])

    def test_trusted_independent_python_certificate_can_promote(self):
        checks = []
        def proof(**args):
            checks.append(args)
            return {"verified": True, "full_source": True,
                    "all_conditions_checked": True, "answer": 5,
                    "proof_backend": "independent_enumeration"}
        answer = self.exercise(verify_python=proof)
        self.assertTrue(answer["verified"])
        self.assertEqual(answer["text"], "5")
        self.assertEqual(self.fallback_calls, [])
        self.assertEqual(checks[0]["source"], self.original[-1]["content"])

    def test_python_proof_missing_components_is_rejected(self):
        for proof in (
            {"verified": True, "all_conditions_checked": True, "answer": 5},
            {"verified": True, "full_source": True, "answer": 5},
            {"verified": True, "full_source": True, "all_conditions_checked": True,
             "answer": 5, "proof_backend": "model_claim"},
        ):
            self.fallback_calls.clear()
            answer = self.exercise(verify_python=lambda **args: proof)
            self.assertFalse(answer["verified"])
            self.assertEqual(len(self.fallback_calls), 1)

    def test_repair_feedback_is_fixed_error_enum_not_traceback(self):
        answer = self.exercise(execute_python=lambda code: {
            "status": "FAILED", "traceback": "PRIVATE_USER_TEXT: crash"})
        self.assertFalse(answer["verified"])
        self.assertEqual(len(self.feedback), MAX_REPAIRS + 1)
        self.assertEqual([f["feedback_code"] for f in self.feedback],
                         [None, "python_runtime_failed", "python_runtime_failed"])
        self.assertNotIn("PRIVATE_USER_TEXT", str(self.feedback))
        self.assertEqual(self.fallback_calls, [self.original])

    def test_one_successful_repair(self):
        feedbacks = []
        def propose(turns, signal):
            feedbacks.append(signal["feedback_code"])
            return {"strategy": "PYTHON_SANDBOX",
                    "code": "print(5)" if len(feedbacks) > 1 else "broken code"}
        def execute(code):
            return ({"status": "PASSED", "output": "5"} if code == "print(5)"
                    else {"status": "FAILED", "traceback": "private"})
        answer = self.exercise(propose=propose, execute_python=execute,
            verify_python=lambda **args: {"verified": True, "full_source": True,
                "all_conditions_checked": True, "answer": 5,
                "proof_backend": "independent_enumeration"})
        self.assertTrue(answer["verified"])
        self.assertEqual(feedbacks, [None, "python_runtime_failed"])
        self.assertEqual(self.fallback_calls, [])

    def test_sandbox_unavailable_skips_repairs(self):
        answer = self.exercise(execute_python=lambda code: {"status": "UNAVAILABLE"})
        self.assertFalse(answer["verified"])
        self.assertEqual(len(self.feedback), 1)
        self.assertEqual(self.fallback_calls, [self.original])

    def test_invalid_strategy_bounded(self):
        answer = self.exercise(propose=lambda *args: {
            "strategy": "ROOT_SHELL", "code": "print(5)"})
        self.assertEqual(answer["planner_attempts"], 3)
        self.assertEqual(answer["text"], "Original Qwen answer")
        self.assertEqual(len(self.fallback_calls), 1)

    def test_z3_requires_full_source_certificate(self):
        self.messages[-1]["content"] = "Synthetic completely parsed finite problem."
        proof = {"verified": True, "proof_backend": "z3_unsat",
                 "verification_scope": "full_source_grammar_exact",
                 "verification_status": "source_canonical_z3_verified",
                 "answer": 7}
        answer = self.exercise(
            source_parser=lambda text: {"canonical": "independent"},
            propose=lambda turns, signal: {"strategy": "Z3_SCHEMA", "schema": {"version": 2}},
            verify_z3=lambda **kwargs: proof)
        self.assertTrue(answer["verified"])
        self.assertEqual(answer["text"], "7")

    def test_unverified_z3_no_proof(self):
        answer = self.exercise(
            propose=lambda turns, signal: {
                "strategy": "Z3_SCHEMA", "schema": {"version": 2}, "verified": True},
            verify_z3=lambda **args: {"verified": True, "answer": 7})
        self.assertFalse(answer["verified"])
        self.assertEqual(self.fallback_calls, [self.original])

    def test_invalid_raw_fallback_is_clear_refusal(self):
        def broken(turns):
            raise RuntimeError("PRIVATE_RESPONSE_CONTENT")
        answer = self.exercise(raw_fallback=broken)
        self.assertFalse(answer["verified"])
        self.assertEqual(answer["selection_policy"], "explicit_safe_refusal")
        self.assertNotIn("PRIVATE_RESPONSE_CONTENT", str(answer))

    def test_repaired_model_does_not_modify_original_baseline(self):
        answer = self.exercise()
        self.assertEqual(self.messages, self.original)
        self.assertEqual(self.fallback_calls, [self.original])
        self.assertIsNone(self.feedback[0]["feedback_code"])
        self.assertEqual(answer["selection_policy"], "raw_qwen_due_to_unverified_adaptive")

    def test_python_adapter_uses_existing_sandbox(self):
        import sys, types
        fake = types.ModuleType("neral_tir_experimental")
        calls = []
        fake.execute_tir_tool = lambda tool, code: (
            calls.append((tool, code)) or {"status": "UNAVAILABLE"})
        old = sys.modules.get("neral_tir_experimental")
        sys.modules["neral_tir_experimental"] = fake
        try:
            self.assertEqual(run_python_in_docker("print(1)"), {"status": "UNAVAILABLE"})
            self.assertEqual(calls, [("python_sandbox", "print(1)")])
        finally:
            if old is None:
                sys.modules.pop("neral_tir_experimental", None)
            else:
                sys.modules["neral_tir_experimental"] = old

    def test_raw_path_delegates_to_unmodified_pr537_guard(self):
        from unittest.mock import patch
        from neral_tir_regression_guard import choose_guarded_tir_reply
        with patch("neral_tir_regression_guard.choose_guarded_tir_reply",
                   wraps=choose_guarded_tir_reply) as guard:
            answer = self.exercise()
        self.assertTrue(guard.called)
        self.assertFalse(answer["verified"])
        self.assertEqual(answer["text"], "Original Qwen answer")
        self.assertEqual(self.fallback_calls, [self.original])

    def test_python_stdout_must_match_independent_answer(self):
        answer = self.exercise(verify_python=lambda **args: {
            "verified": True, "full_source": True, "all_conditions_checked": True,
            "answer": 6, "proof_backend": "independent_enumeration"})
        self.assertFalse(answer["verified"])
        self.assertEqual(answer["text"], "Original Qwen answer")

    def test_multiturn_cannot_claim_full_source_python(self):
        self.messages.insert(1, {"role": "user", "content": "An extra condition."})
        self.original = [dict(x) for x in self.messages]
        original = self.original
        answer = self.exercise(verify_python=lambda **args: {
            "verified": True, "full_source": True, "all_conditions_checked": True,
            "answer": 5, "proof_backend": "independent_enumeration"})
        self.assertFalse(answer["verified"])
        self.assertEqual(self.fallback_calls, [original])

    def test_z3_coverage_repair_feedback_never_leaks_source(self):
        self.messages[-1]["content"] = "Synthetic certificate input."
        feedbacks = []
        def propose(turns, signal):
            feedbacks.append(signal["feedback_code"])
            return {"strategy": "Z3_SCHEMA", "schema": {"version": 2}}
        answer = self.exercise(
            source_parser=lambda text: {"canonical": "trusted"},
            propose=propose,
            verify_z3=lambda **args: {
                "status": "coverage_rejected", "coverage_reason": "PRIVATE_PROMPT_STRING"})
        self.assertFalse(answer["verified"])
        self.assertEqual(feedbacks, [None, "coverage_rejected", "coverage_rejected"])
        self.assertNotIn("PRIVATE_PROMPT_STRING", str(feedbacks))


if __name__ == "__main__":
    unittest.main()
