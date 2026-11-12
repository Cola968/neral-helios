"""CPU contracts for direct source execution and one bounded solver repair turn."""
import json
import unittest
from neral_tir_experimental import run_experimental_tir

SUFFIX="\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."
Q=("How many Boolean assignments to A,B satisfy ALL: "
   "A OR B is True; B is False? Return one integer.")
GOOD={"kind":"formal","complete":True,"schema":{
    "variables":{"A":[False,True],"B":[False,True]},
    "rules":[],"bool_rules":[{"op":"or","args":["A","B"]},
                              {"op":"eq","args":["B",False]}],"query":"count"},
    "evidence":["How many Boolean assignments to A,B","A OR B is True",
                "B is False","Return one integer"]}
BAD=json.loads(json.dumps(GOOD))
BAD["schema"]["bool_rules"].append(False)
SOURCE=("data = [2, -32, -70, 50, 80, 25, 0, -9, -81]\n"
        "for i in range(len(data)):\n"
        "    if i % 2:\n"
        "        data[i] = data[i] * 2 + i\n"
        "    else:\n"
        "        data[i] -= i * 5\n"
        "print(sum(data[::2])*2 - sum(data[1::2]))")

def messages(q):
    return [{"role":"system","content":"NERAL"},{"role":"user","content":q}]

class DirectUserSourceCPU(unittest.TestCase):
    def test_direct_original_code_no_model_rewrite(self):
        seen=[]
        def tool(t,c):
            seen.append((t,c))
            return {"status":"PASSED","output":"-422","stderr":"",
                    "verification_scope":"execution_only"}
        r=run_experimental_tir(
            messages("What exact integer does this Python program print? "
                     "You may execute the program in an offline sandbox:\n"+SOURCE+SUFFIX),
            lambda _:self.fail("No generated code"),
            extract_constraints=lambda _:self.fail("No extraction before direct source"),
            tool_runner=tool)
        self.assertEqual(seen,[("python_sandbox",SOURCE)])
        self.assertTrue(r["verified"])
        self.assertEqual(r["text"],"FINAL_ANSWER: -422")
        self.assertEqual(r["review_method"],"exact_user_ast_match")
        self.assertEqual(r["tool_calls_succeeded"],1)
        self.assertEqual(r["calls_used"],0)

    def test_terminal_fenced_python_runs_unchanged(self):
        src="a=2\nprint(a+3)"
        r=run_experimental_tir(
            messages("What integer does this Python code print?\n"+
                     "\x60\x60\x60python\n"+src+"\n\x60\x60\x60"),
            lambda _:self.fail("No model"),
            tool_runner=lambda t,c:({"status":"PASSED","output":"5","stderr":""}
                if (t,c)==("python_sandbox",src) else self.fail("Wrong source")))
        self.assertTrue(r["verified"])
        self.assertEqual(r["text"],"5")

    def test_never_auto_execute_unsafe_or_ambiguous_source(self):
        for source in (
            "print(__import__('os').system('id'))",
            "open('/tmp/x').write('evil')",
            "print(1)\nprint(2)",
            "import os\nprint(1)",
            "print(1)\nnot Python prose after code",
        ):
            with self.subTest(source=source):
                r=run_experimental_tir(
                    messages("What does this Python program print:\n"+source),
                    lambda _:"NERAL_FINAL_TEXT: unverified",
                    tool_runner=lambda *_:self.fail("Unsafe code execution"))
                self.assertFalse(r["verified"])
                self.assertEqual(r["tool_calls_succeeded"],0)

    def test_fix_request_must_not_run_original_code_as_solution(self):
        r=run_experimental_tir(
            messages("Fix this Python program and print result:\nprint(1)"),
            lambda _:"NERAL_FINAL_TEXT: no",tool_runner=lambda *_:self.fail("No direct run"))
        self.assertFalse(r["verified"])

    def test_sandbox_unavailable_fails_closed(self):
        r=run_experimental_tir(
            messages("What integer does this Python program print:\nprint(1)"),
            lambda _:"NERAL_FINAL_TEXT: unavailable",
            tool_runner=lambda *_:{"status":"UNAVAILABLE"})
        self.assertFalse(r["verified"])

class OneTurnConstraintRepairCPU(unittest.TestCase):
    def test_unsat_feedback_one_retry_and_solver_result(self):
        asked=[]
        proposals=iter((json.dumps(BAD),json.dumps(GOOD)))
        def extract(turns):
            asked.append(turns)
            return next(proposals)
        r=run_experimental_tir(
            messages(Q+SUFFIX),lambda _:self.fail("No raw fallback on repaired schema"),
            extract_constraints=extract)
        self.assertEqual(len(asked),2)
        self.assertIn("UNSAT",asked[1][0]["content"])
        self.assertIn("unsat_core_index",asked[1][1]["content"])
        self.assertIn("original",asked[1][1]["content"].lower())
        self.assertEqual(r["text"],"FINAL_ANSWER: 1")
        self.assertFalse(r["verified"])
        self.assertEqual(r["verification_status"],"model_extracted_schema_unverified")
        self.assertEqual(r["tool_used"],"neral_logic_exact")
        self.assertEqual(r["tool_calls_succeeded"],1)
        self.assertEqual(r["calls_used"],2)
        self.assertEqual(r["extraction_status"],"repaired")
        self.assertEqual(r["first_turn_telemetry"]["solver_repair_status"],"repaired")

    def test_multiple_assignments_count_not_blocked_by_one_solution_gate(self):
        good=json.loads(json.dumps(GOOD))
        good["schema"]["bool_rules"]=[{"op":"or","args":["A","B"]}]
        proposals=iter((json.dumps(BAD),json.dumps(good)))
        r=run_experimental_tir(messages(Q),lambda _:self.fail("No raw fallback"),
                               extract_constraints=lambda _:next(proposals))
        self.assertEqual(r["text"],"3")
        self.assertFalse(r["verified"])

    def test_only_one_retry_even_if_retry_is_unsat(self):
        observed=[]
        proposals=iter((json.dumps(BAD),json.dumps(BAD)))
        def extract(turns):
            observed.append(turns)
            return next(proposals)
        r=run_experimental_tir(messages(Q),lambda _:"Raw answer",
                               extract_constraints=extract)
        self.assertEqual(len(observed),2)
        self.assertFalse(r["verified"])
        self.assertEqual(r["verification_status"],"base_fallback_unverified")
        self.assertEqual(r["first_turn_telemetry"]["solver_repair_status"],"retry_unsat")

    def test_type_mismatch_gets_one_strict_feedback_turn(self):
        bad = json.loads(json.dumps(GOOD))
        # Boolean A compared with an integer: a grounded schema candidate
        # that the finite-domain solver must reject as comparison_types.
        bad["schema"]["rules"] = [{"left":"A","op":"==","value":0}]
        proposals = iter((json.dumps(bad), json.dumps(GOOD)))
        received = []
        def extract(turns):
            received.append(turns)
            return next(proposals)
        r = run_experimental_tir(
            messages(Q), lambda _: self.fail("A repaired schema must not fall back"),
            extract_constraints=extract)
        self.assertEqual(len(received), 2)
        self.assertIn("comparison_types", received[1][1]["content"])
        self.assertIn("original_question", received[1][1]["content"])
        self.assertEqual(r["text"], "1")
        self.assertEqual(r["extraction_status"], "repaired")
        self.assertFalse(r["verified"])
        tele = r["first_turn_telemetry"]
        self.assertEqual(tele["extraction_rejection_reason"], "solver_type_error")
        self.assertEqual(tele["solver_rejection_detail"], "comparison_types")
        self.assertEqual(tele["solver_repair_status"], "repaired")

    def test_repair_accepts_markdown_fence_and_leading_explanation(self):
        for lead, label in (
            ("", "fenced_json"),
            ("Hier ist das korrigierte Schema:\n", "prefixed_fenced_json"),
        ):
            with self.subTest(label=label):
                wrapped = lead + "\x60\x60\x60json\n" + json.dumps(GOOD) + "\n\x60\x60\x60"
                proposals = iter((json.dumps(BAD), wrapped))
                r = run_experimental_tir(
                    messages(Q), lambda _: self.fail("Valid repaired schema must solve"),
                    extract_constraints=lambda _: next(proposals))
                self.assertEqual(r["text"], "1")
                self.assertEqual(r["extraction_status"], "repaired")
                self.assertEqual(r["first_turn_telemetry"]["solver_repair_status"], "repaired")
                self.assertEqual(r["first_turn_telemetry"]["solver_repair_input_format"], label)
                self.assertFalse(r["verified"])

    def test_repair_accepts_leading_plain_text_and_whitespace(self):
        revised = "Here is the corrected schema:\n" + json.dumps(GOOD) + " \n"
        proposals = iter((json.dumps(BAD), revised))
        r = run_experimental_tir(
            messages(Q), lambda _: self.fail("Valid repaired schema must solve"),
            extract_constraints=lambda _: next(proposals))
        self.assertEqual(r["text"], "1")
        self.assertEqual(r["first_turn_telemetry"]["solver_repair_input_format"],
                         "prefixed_json")

    def test_repair_six_clauses_solves_twelve_without_dropping_clues(self):
        question = (
            "How many Boolean assignments to A,B,C,D,E,F,G satisfy ALL: "
            "(A XOR C XOR F) is False; (B OR D) is True; E = NOT F; "
            "(A AND G) is False; C implies G; "
            "at least 2 of the seven variables are True? Return one integer."
        )
        clauses = [
            {"op": "eq", "args": [{"op": "xor", "args": ["A", "C", "F"]}, False]},
            {"op": "or", "args": ["B", "D"]},
            {"op": "eq", "args": ["E", {"op": "not", "args": ["F"]}]},
            {"op": "eq", "args": [{"op": "and", "args": ["A", "G"]}, False]},
            {"op": "implies", "args": ["C", "G"]},
            {"op": "at_least", "min": 2,
             "args": ["A", "B", "C", "D", "E", "F", "G"]},
        ]
        evidence = [
            "How many Boolean assignments to A,B,C,D,E,F,G",
            "(A XOR C XOR F) is False", "(B OR D) is True", "E = NOT F",
            "(A AND G) is False", "C implies G",
            "at least 2 of the seven variables are True", "Return one integer",
        ]
        good = {"kind": "formal", "complete": True,
                "schema": {"variables": {n: [False, True] for n in "ABCDEFG"},
                           "rules": [], "bool_rules": clauses, "query": "count"},
                "evidence": evidence}
        bad = json.loads(json.dumps(good))
        bad["schema"]["bool_rules"].append(False)
        proposals = iter((json.dumps(bad),
                          "Repaired:\n\x60\x60\x60json\n"
                          + json.dumps(good) + "\n\x60\x60\x60"))
        r = run_experimental_tir(
            messages(question), lambda _: self.fail("Six correct clauses should solve"),
            extract_constraints=lambda _: next(proposals))
        self.assertEqual(r["text"], "12")
        self.assertEqual(r["extraction_status"], "repaired")
        self.assertEqual(r["tool_calls_succeeded"], 1)
        self.assertFalse(r["verified"])  # Model translation is never certified.

    def test_repair_rejection_reports_safe_scalar_reason(self):
        proposals = iter((json.dumps(BAD), "Repaired: {bad JSON}"))
        r = run_experimental_tir(
            messages(Q), lambda _: "Raw answer",
            extract_constraints=lambda _: next(proposals))
        tele = r["first_turn_telemetry"]
        self.assertEqual(tele["solver_repair_status"], "retry_rejected")
        self.assertEqual(tele["solver_repair_rejection_reason"], "invalid_json")
        self.assertNotIn("bad JSON", str(tele))

    def test_repair_does_not_strip_extra_payload_or_relax_validation(self):
        malicious = (
            "Corrected schema:\n\x60\x60\x60json\n" + json.dumps(GOOD)
            + "\n\x60\x60\x60\nAdditional answer: 1"
        )
        bad_evidence = json.loads(json.dumps(GOOD))
        bad_evidence["evidence"][1] = "invented evidence not in question"
        for revised in (malicious, json.dumps(GOOD) + "\n" + json.dumps(GOOD),
                        json.dumps(bad_evidence),
                        "Only prose, no JSON"):
            with self.subTest(revised=revised[:20]):
                proposals = iter((json.dumps(BAD), revised))
                r = run_experimental_tir(
                    messages(Q), lambda _: "Raw answer",
                    extract_constraints=lambda _: next(proposals))
                self.assertEqual(r["text"], "Raw answer")
                self.assertFalse(r["verified"])
                self.assertEqual(r["tool_calls_succeeded"], 0)
                self.assertEqual(r["first_turn_telemetry"]["solver_repair_status"],
                                 "retry_rejected")

    def test_malformed_retry_falls_back_unverified(self):
        proposals=iter((json.dumps(BAD),"INVALID"))
        seen=[]
        r=run_experimental_tir(messages(Q),lambda m:(seen.append(m) or "Raw answer"),
                               extract_constraints=lambda _:next(proposals))
        self.assertEqual(seen,[messages(Q)])
        self.assertFalse(r["verified"])
        self.assertEqual(r["first_turn_telemetry"]["solver_repair_status"],"retry_rejected")

    def test_no_retry_on_unrelated_parser_failure(self):
        seen=[]
        r=run_experimental_tir(messages(Q),lambda _:"Raw",
                               extract_constraints=lambda m:(seen.append(m) or "INVALID"))
        self.assertEqual(len(seen),1)
        self.assertFalse(r["verified"])

class SolverRepairTelemetryExportCPU(unittest.TestCase):
    def test_second_turn_rejection_is_preserved_in_safe_gpu_scalars(self):
        from scripts.neral_tir_hard_live import safe_first_turn_event
        event = {
            "eval_count": 374, "prompt_eval_count": 839,
            "done_reason": "stop", "decision": "extraction_fallback",
            "solver_repair_status": "retry_rejected",
            "solver_repair_input_format": "prefixed_fenced_json",
            "solver_repair_rejection_reason": "solver_schema_error",
            "solver_repair_solver_detail": "boolean_cardinality_threshold",
            "solver_repair_ast_check_code": "boolean_literal_token",
            "model_text": "PRIVATE_MODEL_REPLY",
        }
        safe = safe_first_turn_event(event)
        self.assertIsNotNone(safe)
        for key in ("solver_repair_status", "solver_repair_input_format",
                    "solver_repair_rejection_reason", "solver_repair_solver_detail",
                    "solver_repair_ast_check_code"):
            self.assertEqual(safe[key], event[key])
        self.assertNotIn("PRIVATE_MODEL_REPLY", str(safe))

    def test_invalid_repair_diagnostics_cannot_be_exported(self):
        from scripts.neral_tir_hard_live import safe_first_turn_event
        base = {"eval_count": 200, "prompt_eval_count": 700,
                "done_reason": "stop", "decision": "extraction_fallback"}
        for key in ("solver_repair_input_format", "solver_repair_rejection_reason",
                    "solver_repair_solver_detail", "solver_repair_ast_check_code"):
            with self.subTest(key=key):
                self.assertIsNone(safe_first_turn_event({
                    **base, key: "PRIVATE_MODEL_REPLY"}))


if __name__=="__main__":
    unittest.main()
