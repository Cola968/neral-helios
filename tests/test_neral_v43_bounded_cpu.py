"""CPU-only V4.3 draft-first coordination; no model or hardware."""
import unittest
from unittest.mock import Mock
from neral_v43_bounded import bounded_answer, math_final_instruction, normalize_math_final, Verdict
from neral_v42_deadline import DeadlineExceeded


class BoundedAnswerCPU(unittest.TestCase):
    def test_numeric_tail_preserves_work_and_is_accepted_by_exact_grader(self):
        from scripts.neral_v41_blind200_live import _exact_text

        # Synthetic quantities exercise the three observed output shapes
        # without retaining any sealed prompts, values or unit wording.
        for subtotal, removed, ending in ((24, 5, 19), (36, 9, 27), (54, 7, 47)):
            with self.subTest(ending=ending):
                work = (f"Vorrat: 6 Schachteln × {subtotal // 6} Knöpfe = {subtotal} Knöpfe.\n"
                        f"Nach der Entnahme: {subtotal} Knöpfe - {removed} Knöpfe = {ending} Knöpfe.\n")
                original = f"{work}{ending}"
                self.assertIsNone(_exact_text(original, "integer"))
                normalized = normalize_math_final(original)
                self.assertEqual(normalized, f"{work}ENDERGEBNIS: {ending}")
                self.assertEqual(_exact_text(normalized, "integer"), str(ending))
                self.assertEqual(normalize_math_final(normalized), normalized)

    def test_latex_and_markdown_cleanup_preserves_exact_integer(self):
        for original in (r"\boxed{-0042}", r"\[\boxed{-0042}\]",
                         r"Rechnung dokumentiert." "\n" r"ENDERGEBNIS: $\boxed{-0042}$",
                         "ENDERGEBNIS: **-0042**", "**-0042**"):
            with self.subTest(original=original):
                result = normalize_math_final(original, requested=True)
                line = result.splitlines()[-1]
                self.assertIn(line, ("ENDERGEBNIS: -0042", "ENDERGEBNIS: **-0042**"))
                self.assertEqual(result.splitlines()[:-1], original.splitlines()[:-1])

    def test_ambiguous_and_already_parseable_answers_are_untouched(self):
        items = ("+0042", "Antwort: 42", "Rechnung\nENDERGEBNIS: 42",
                 "Text\n42", "19\n42", "50 - 9 = 41 Stück.\n42",
                 "ENDERGEBNIS: 41\n50 - 8 = 42.\n42",
                 r"\boxed{41}" "\n" r"\boxed{42}",
                 r"\boxed{\frac{1}{2}}", r"\boxed{3,5}", "50 - 9 = 41.\n41 und 43",
                 "\x60\x60\x60python\nprint(42)\n\x60\x60\x60")
        for text in items:
            with self.subTest(text=text):
                self.assertEqual(normalize_math_final(text, requested=True), text)

    def test_format_instruction_respects_other_contracts(self):
        self.assertIn("ENDERGEBNIS: <wert>", math_final_instruction("Wie viele Stück sind übrig?"))
        for question in ("Nenne die Hauptstadt von Italien.",
                         "Berechne 31 + 11. Nur die Zahl.",
                         "Calculate the answer as JSON.",
                         "Berechne die Werte in Python."):
            self.assertEqual(math_final_instruction(question), "")

    def test_runtime_normalizes_before_verification_and_keeps_source_question(self):
        from unittest.mock import patch
        from neral_three_tier_runtime import ThreeTierRuntime, RAW_9B

        runtime = Mock(spec=ThreeTierRuntime)
        runtime.helios = Mock(model=RAW_9B, loaded=True, adapter_sha256=None)
        runtime.helios.respond_chat.return_value = "Rechnung: 29 - 6 = 23 Stück.\n23"
        question = "Wie viele Stück sind übrig?"
        turns = [{"role": "system", "content": "Answer directly."},
                 {"role": "user", "content": question}]
        with patch("neral_v43_routes.verify_answer", return_value=Verdict("unsupported")) as verifier:
            result = ThreeTierRuntime.answer_v43_bounded(runtime, turns, "Obsidian")
        self.assertEqual(verifier.call_args.args[0], question)
        self.assertEqual(verifier.call_args.args[1]["text"], result["text"])
        self.assertTrue(result["text"].endswith("\nENDERGEBNIS: 23"))
        self.assertFalse(result["verified"])
        self.assertEqual(result["generation_rounds"], 1)
        sent = runtime.helios.respond_chat.call_args.args[0]
        self.assertIn("ENDERGEBNIS: <wert>", sent[0]["content"])
        self.assertEqual(turns[0]["content"], "Answer directly.")
        self.assertEqual(sent[-1]["content"], question)

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

    def test_independent_exact_integer_cannot_use_numbers_in_prose(self):
        from neral_v43_bounded import exact_integer_verdict
        self.assertEqual(exact_integer_verdict({"text":"41"},{"text":"42"}).status,"counterexample")
        self.assertEqual(exact_integer_verdict({"text":"42"},{"text":"42"}).status,"verified")
        self.assertEqual(exact_integer_verdict({"text":"I saw 42"},{"text":"42"}).status,"unsupported")

    def test_actual_runtime_does_not_call_old_planner(self):
        from neral_three_tier_runtime import ThreeTierRuntime, HELIOS
        runtime=Mock(spec=ThreeTierRuntime)
        runtime.helios=Mock(model=HELIOS,loaded=True,adapter_sha256="0"*64)
        runtime.helios.respond_chat.return_value="42"
        runtime._verified_source.side_effect=DeadlineExceeded("tool")
        turns=[{"role":"system","content":"Answer directly."},{"role":"user","content":"What is 6 times 7?"}]
        result=ThreeTierRuntime.answer_v43_bounded(runtime,turns,"Obsidian")
        self.assertEqual(result["text"],"42")
        self.assertTrue(result["v43_bounded"])
        self.assertEqual(runtime.helios.respond_chat.call_count,1)
        self.assertEqual(result["generation_rounds"],1)
        self.assertIsNone(result["error_code"])

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
