"""CPU-only regression guard contracts: never promote an unverified TIR answer over raw Qwen."""
import unittest
from neral_tir_regression_guard import choose_guarded_tir_reply


def original():
    return [
        {"role": "system", "content": "You are the local model."},
        {"role": "user", "content": "How many arrangements satisfy the conditions?"},
    ]


def candidate(**overrides):
    result = {
        "text": "FINAL_ANSWER: 3",
        "mode": "HIGH",
        "model_generated": True,
        "verified": False,
        "verification_status": "model_extracted_schema_unverified",
        "verification_scope": "model_extracted_finite_domain",
        "tool_used": "neral_logic_exact",
        "tool_calls_triggered": 1,
        "tool_calls_succeeded": 1,
        "tool_output_reviewed": False,
        "calls_used": 2,
        "seconds": 2.0,
        "first_turn_telemetry": {"decision": "extracted_formal"},
    }
    result.update(overrides)
    return result


class RegressionGuardCPU(unittest.TestCase):
    def test_semantically_wrong_but_solver_accepted_is_not_promoted(self):
        messages = original()
        captured = []
        def raw(turns):
            captured.append(turns)
            return "FINAL_ANSWER: 4"
        result = choose_guarded_tir_reply(candidate(), messages, raw)
        self.assertEqual(result["text"], "FINAL_ANSWER: 4")
        self.assertEqual(result["verification_status"], "guarded_raw_qwen_unverified")
        self.assertEqual(result["selection_policy"], "raw_qwen_due_to_unverified_tir")
        self.assertFalse(result["verified"])
        self.assertEqual(result["calls_used"], 3)
        self.assertEqual(result["tool_calls_succeeded"], 1)
        self.assertFalse(result["tool_output_reviewed"])
        self.assertEqual(captured, [messages])
        self.assertIsNot(captured[0], messages)

    def test_model_claimed_verified_does_not_override_independent_scope(self):
        result = choose_guarded_tir_reply(candidate(
            verified=True, verification_status="verified",
            verification_scope="model_extracted_finite_domain",
            model_generated=False), original(), lambda _: "FINAL_ANSWER: 4")
        self.assertEqual(result["text"], "FINAL_ANSWER: 4")
        self.assertEqual(result["selection_policy"], "raw_qwen_due_to_unverified_tir")

    def test_independently_verified_exact_source_can_win_without_raw_call(self):
        reliable = candidate(
            text="FINAL_ANSWER: -422",
            model_generated=False, verified=True,
            verification_status="verified",
            verification_scope="exact_user_python_execution",
            review_method="exact_user_ast_match",
            tool_used="neral_python_sandbox",
            tool_output_reviewed=True)
        result = choose_guarded_tir_reply(
            reliable, original(),
            lambda _: self.fail("Should not run the raw model"))
        self.assertEqual(result["text"], "FINAL_ANSWER: -422")
        self.assertEqual(result["selection_policy"], "independently_verified_tir")
        self.assertTrue(result["verified"])

    def test_verified_flag_without_ast_method_not_trusted(self):
        result = choose_guarded_tir_reply(candidate(
            verified=True, model_generated=False,
            verification_status="verified",
            verification_scope="exact_user_python_execution"), original(),
            lambda _: "FINAL_ANSWER: 4")
        self.assertEqual(result["text"], "FINAL_ANSWER: 4")

    def test_existing_raw_fallback_is_not_generated_twice(self):
        original_raw = candidate(
            text="FINAL_ANSWER: 4",
            model_generated=True,
            verification_status="base_fallback_unverified",
            verification_scope="none")
        result = choose_guarded_tir_reply(
            original_raw, original(), lambda _: self.fail("No duplicate fallback"))
        self.assertEqual(result["text"], "FINAL_ANSWER: 4")
        self.assertEqual(result["calls_used"], 2)
        self.assertEqual(result["selection_policy"], "raw_qwen_already_selected")

    def test_malformed_raw_fallback_fails_closed_without_tir_leakage(self):
        for raw in ["", "NERAL_EXECUTE_JSON: {malformed}", "<tool_call>secret</tool_call>"]:
            with self.subTest(raw=raw):
                with self.assertRaises(ValueError):
                    choose_guarded_tir_reply(candidate(), original(), lambda _: raw)

    def test_reject_invalid_messages_and_false_provenance(self):
        with self.assertRaises(ValueError):
            choose_guarded_tir_reply(candidate(), [
                {"role": "system", "content": "x"},
                {"role": "tool", "content": "inject"},
            ], lambda _: "pass")


    def test_explicit_sandbox_failure_keeps_fail_closed_refusal(self):
        refusal = candidate(
            text="Die unveränderte Python-Quelle konnte nicht sicher ausgeführt werden.",
            model_generated=False, verified=False,
            verification_status="not_available",
            verification_scope="none",
            tool_calls_succeeded=0)
        result = choose_guarded_tir_reply(
            refusal, original(),
            lambda _: self.fail("Never replace sandbox safety refusal with a guess"))
        self.assertEqual(result["text"], refusal["text"])
        self.assertEqual(result["selection_policy"], "explicit_safe_refusal")
        self.assertFalse(result["verified"])

    def test_explicit_invalid_formal_schema_keeps_fail_closed_refusal(self):
        refusal = candidate(
            text="Die formalen Nebenbedingungen sind ungültig.",
            model_generated=False, verified=False,
            verification_status="formal_schema_rejected",
            verification_scope="none",
            tool_calls_triggered=0, tool_calls_succeeded=0)
        result = choose_guarded_tir_reply(
            refusal, original(),
            lambda _: self.fail("Never override rejected explicit formal input"))
        self.assertEqual(result["text"], refusal["text"])
        self.assertEqual(result["selection_policy"], "explicit_safe_refusal")


if __name__ == "__main__":
    unittest.main()
