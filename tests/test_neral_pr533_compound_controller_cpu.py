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

if __name__=="__main__":
    unittest.main()
