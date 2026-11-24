"""CPU-only V4.3 draft-first coordination; no model or hardware."""
import unittest
from unittest.mock import Mock
from neral_v43_bounded import bounded_answer, Verdict
from neral_v42_deadline import DeadlineExceeded


class BoundedAnswerCPU(unittest.TestCase):
    def test_deadline_after_draft_preserves_text_and_clears_error(self):
        draft={"text":"42","model_generated":True,"verified":False}
        generate=Mock(return_value=draft)
        verify=Mock(side_effect=DeadlineExceeded("test"))
        result=bounded_answer(generate,verify,remaining=lambda:30)
        self.assertEqual(result["text"],"42")
        self.assertIsNone(result["error_code"])
        self.assertEqual(result["verification_status"],"verification_incomplete")
        self.assertFalse(result["verified"])
        self.assertEqual(result["generation_rounds"],1)

    def test_only_confirmed_counterexample_may_trigger_one_repair(self):
        draft={"text":"41","model_generated":True}
        generate=Mock(return_value=draft)
        verify=Mock(side_effect=[Verdict("counterexample","trusted mismatch"),
                                 Verdict("counterexample","still wrong")])
        repair=Mock(return_value={"text":"40","model_generated":True})
        result=bounded_answer(generate,verify,repair=repair,remaining=lambda:30)
        self.assertEqual(result["text"],"41")
        self.assertFalse(result["verified"])
        self.assertEqual(result["verification_status"],"counterexample_unresolved")
        repair.assert_called_once()
        self.assertEqual(result["generation_rounds"],2)

    def test_repaired_answer_only_accepted_on_independent_pass(self):
        result=bounded_answer(lambda:{"text":"41"},Mock(side_effect=[
            Verdict("counterexample","trusted mismatch"),Verdict("verified")]),
            repair=lambda *_:{"text":"42"},remaining=lambda:30)
        self.assertEqual(result["text"],"42")
        self.assertTrue(result["verified"])
        self.assertEqual(result["generation_rounds"],2)

    def test_timeout_unknown_and_tool_error_do_not_trigger_repair(self):
        for status in ("unknown","unsupported","timeout","sandbox_error"):
            repair=Mock()
            result=bounded_answer(lambda:{"text":"42"},lambda _:Verdict(status),
                repair=repair,remaining=lambda:30)
            self.assertEqual(result["text"],"42")
            self.assertFalse(result["verified"])
            repair.assert_not_called()

    def test_insufficient_budget_for_repair(self):
        repair=Mock()
        result=bounded_answer(lambda:{"text":"42"},lambda _:Verdict("counterexample","wrong"),
            repair=repair,remaining=lambda:7)
        self.assertEqual(result["text"],"42")
        repair.assert_not_called()

    def test_no_fallback_without_real_draft(self):
        with self.assertRaises(DeadlineExceeded):
            bounded_answer(Mock(side_effect=DeadlineExceeded("test")),remaining=lambda:30)

    def test_runtime_failure_after_draft_is_not_an_empty_response(self):
        result=bounded_answer(lambda:{"text":"42"},Mock(side_effect=RuntimeError("tool")),
            remaining=lambda:30)
        self.assertEqual(result["text"],"42")
        self.assertEqual(result["verification_status"],"verification_incomplete")

    def test_repair_deadline_preserves_draft_and_counterexample_warning(self):
        repair=Mock(side_effect=DeadlineExceeded("repair"))
        result=bounded_answer(lambda:{"text":"41"},lambda _:Verdict("counterexample","wrong"),
            repair=repair,remaining=lambda:30)
        self.assertEqual(result["text"],"41")
        self.assertEqual(result["verification_status"],"counterexample_unresolved")
        self.assertEqual(result["generation_rounds"],2)

if __name__=="__main__":
    unittest.main()
