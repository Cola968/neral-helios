"""NEW disjoint synthetic CPU cases. No historical smoke prompts or blind100 data."""
import ast
import json
import time
import unittest
from unittest.mock import patch

from neral_adaptive_planner import run_adaptive_planner
from neral_python_assert_contract import inspect_python_contract
from neral_python_unique_logic import (
    parse_finite_unique_source, independent_unique_certificate)
from neral_python_source_router import verify_python_full_source
from neral_adaptive_bounded_fallback import make_bounded_fallback

TRUTH_UNIQUE=(
    "Find the unique truth assignment for speakers A, B, C. "
    "A says B lies. B says C tells the truth. C says A lies. "
    "Exactly 1 speakers tell the truth. Return the truth bitmask."
)
TRUTH_AMBIGUOUS=(
    "Find the unique truth assignment for speakers A, B, C. "
    "A says B lies. B says C tells the truth. C says A lies. "
    "Return the truth bitmask."
)
TRUTH_IMPOSSIBLE=(
    "Find the unique truth assignment for speakers A, B, C. "
    "A says B lies. B says C lies. C says A lies. "
    "Return the truth bitmask."
)
ROOM_UNIQUE=(
    "Find the unique room assignment for Mara, Niko, Oona, Petra to rooms 1 through 4. "
    "Mara is before Niko. Niko is immediately before Oona. "
    "Petra is last. Return Mara's room number."
)
ROOM_AMBIGUOUS=(
    "Find the unique room assignment for Mara, Niko, Oona, Petra to rooms 1 through 4. "
    "Niko is immediately before Oona. Petra is last. "
    "Return Mara's room number."
)
ROOM_OTHER=(
    "Find the unique room assignment for Iris, Jaro, Kira to rooms 1 through 3. "
    "Iris is first. Kira is last. Return Jaro's room number."
)

POSSIBLE="""
from itertools import product
solutions = []
for a, b, c in product([False, True], repeat=3):
    if (a == (not b) and b == c and c == (not a) and sum((a,b,c)) == 1):
        solutions.append((a,b,c))
assert len(solutions) == 1
answer = sum((1 << i) for i, bit in enumerate(solutions[0]) if bit)
assert answer == 1
print(answer)
""".strip()

ROOM_SCRIPT="""
from itertools import permutations
solutions = []
for rooms in permutations((1,2,3,4)):
    if rooms[0]<rooms[1] and rooms[1]+1==rooms[2] and rooms[3]==4:
        solutions.append(rooms)
assert len(solutions) == 1
answer = solutions[0][0]
assert answer == 1
print(answer)
""".strip()

def turns(q):
    return [{"role":"system","content":"New synthetic CPU contract."},
            {"role":"user","content":q}]


class ASTGateCPU(unittest.TestCase):
    def test_assert_missing_has_exact_code(self):
        result=inspect_python_contract("answer=42\nprint(answer)")
        self.assertEqual(result,"python_missing_assert")

    def test_print_missing_has_exact_code(self):
        result=inspect_python_contract("n=9\nassert n == 3*3")
        self.assertEqual(result,"python_missing_print")

    def test_both_missing_prioritizes_assert(self):
        self.assertEqual(inspect_python_contract("x=1"),"python_missing_assert")

    def test_no_fake_assert_or_print_in_comments(self):
        self.assertEqual(inspect_python_contract(
            "x=10\n# assert x == 2*5\nprint(x)"),"python_missing_assert")
        self.assertEqual(inspect_python_contract(
            "x=10\nassert x==2*5\n# print(x)"),"python_missing_print")

    def test_nontrivial_assert_with_real_final_print_passes(self):
        self.assertIsNone(inspect_python_contract(
            "x=11\nassert x == 5+6\nprint(x)"))

    def test_print_in_unexecuted_function_cannot_count(self):
        result=inspect_python_contract(
            "x=11\nassert x == 5+6\ndef never():\n    print(x)")
        self.assertEqual(result,"python_missing_print")

    def test_nonscalar_debug_only_print_is_not_final_result(self):
        result=inspect_python_contract("x=11\nassert x == 5+6\nprint('debug')")
        self.assertEqual(result,"python_missing_print")

    def test_test_timing_typical_ast_under_one_ms_median(self):
        small="x=17\nassert x == 2*8+1\nprint(x)"
        for _ in range(10):inspect_python_contract(small)
        measurements=[]
        for _ in range(100):
            start=time.perf_counter_ns()
            self.assertIsNone(inspect_python_contract(small))
            measurements.append((time.perf_counter_ns()-start)/1e6)
        self.assertLess(sorted(measurements)[50],1.0)

    def test_first_missing_assert_then_repair_once_success(self):
        codes=["x=7\nprint(x)", "x=7\nassert x == 3+4\nprint(x)"]
        feedback=[];execution=[]
        def propose(_,signal):
            feedback.append(signal["feedback_code"])
            return {"strategy":"PYTHON_SANDBOX","code":codes[signal["repair_round"]]}
        out=run_adaptive_planner(
            turns(ROOM_OTHER),propose,lambda _:"BASE",
            lambda code:execution.append(code) or {"status":"PASSED","output":"2"},
            lambda **kw:{"verified":True,"full_source":True,
                        "all_conditions_checked":True,"answer":2,
                        "proof_backend":"independent_unique_enumeration"},
            source_supported=lambda _:True)
        self.assertEqual(feedback,[None,"python_missing_assert"])
        self.assertEqual(len(execution),1)
        self.assertTrue(out["verified"])

    def test_first_missing_print_then_repair_once_success(self):
        codes=["x=2\nassert x == 1+1", "x=2\nassert x == 1+1\nprint(x)"]
        feedback=[];execution=[]
        def propose(_,signal):
            feedback.append(signal["feedback_code"])
            return {"strategy":"PYTHON_SANDBOX","code":codes[signal["repair_round"]]}
        out=run_adaptive_planner(
            turns(ROOM_OTHER),propose,lambda _:"BASE",
            lambda code:execution.append(code) or {"status":"PASSED","output":"2"},
            lambda **kw:{"verified":True,"full_source":True,
                        "all_conditions_checked":True,"answer":2,
                        "proof_backend":"independent_unique_enumeration"},
            source_supported=lambda _:True)
        self.assertEqual(feedback,[None,"python_missing_print"])
        self.assertEqual(len(execution),1)
        self.assertTrue(out["verified"])

    def test_duplicate_after_missing_assert_stops_before_docker(self):
        code="x=7\nprint(x)"
        signals=[]
        out=run_adaptive_planner(
            turns(ROOM_OTHER),
            lambda _,signal: signals.append(signal) or {
                "strategy":"PYTHON_SANDBOX","code":code},
            lambda _:"BASE",
            lambda _:self.fail("No Docker for missing assert"),
            lambda **kw:self.fail("No oracle for missing assert"),
            source_supported=lambda _:True)
        self.assertEqual(len(signals),2)
        self.assertFalse(out["verified"])

    def test_same_repair_feedback_never_leaks_prompt(self):
        from neral_adaptive_ollama import make_local_adaptive_proposer
        packets=[]
        def transport(payload):
            packets.append(payload)
            return {"done":True,"done_reason":"stop",
                    "message":{"content":'{"strategy":"DIRECT"}'}}
        propose=make_local_adaptive_proposer("neral-qwen35-qlora-v1:latest",
                                             transport=transport)
        propose(turns(ROOM_OTHER),{"preferred_strategy":"PYTHON_SANDBOX",
            "allowed_strategies":("PYTHON_SANDBOX","DIRECT"),
            "repair_round":1,"feedback_code":"python_missing_print"})
        system=packets[0]["messages"][0]["content"]
        self.assertIn("Fehlende print(...)-Ausgabe des Endergebnisses",system)
        self.assertNotIn("PRIVATE_TRACEBACK",system)


class IndependentFiniteLogicCPU(unittest.TestCase):
    def test_unique_truth_source_and_bitmask(self):
        task=parse_finite_unique_source(TRUTH_UNIQUE)
        self.assertIsNotNone(task)
        self.assertEqual(task.answer,1)
        self.assertEqual(task.number_of_solutions,1)

    def test_two_valid_truth_assignments_cannot_promote(self):
        task=parse_finite_unique_source(TRUTH_AMBIGUOUS)
        self.assertIsNotNone(task)
        self.assertEqual(task.number_of_solutions,2)
        self.assertFalse(independent_unique_certificate(
            source=TRUTH_AMBIGUOUS,code=POSSIBLE,output="1",canonical=None)["verified"])

    def test_zero_solution_truth_must_refuse(self):
        task=parse_finite_unique_source(TRUTH_IMPOSSIBLE)
        self.assertIsNotNone(task)
        self.assertEqual(task.number_of_solutions,0)
        self.assertFalse(independent_unique_certificate(
            source=TRUTH_IMPOSSIBLE,code=POSSIBLE,output="0",canonical=None)["verified"])

    def test_true_unique_enumeration_proof(self):
        cert=independent_unique_certificate(
            source=TRUTH_UNIQUE,code=POSSIBLE,output="1",canonical=None)
        self.assertTrue(cert["verified"],cert)
        self.assertEqual(cert["proof_backend"],"independent_unique_enumeration")

    def test_room_unique_and_ambiguous(self):
        a=parse_finite_unique_source(ROOM_UNIQUE)
        b=parse_finite_unique_source(ROOM_AMBIGUOUS)
        self.assertEqual(a.answer,1)
        self.assertEqual(a.number_of_solutions,1)
        self.assertGreater(b.number_of_solutions,1)
        self.assertTrue(independent_unique_certificate(
            source=ROOM_UNIQUE,code=ROOM_SCRIPT,output="1",canonical=None)["verified"])
        self.assertFalse(independent_unique_certificate(
            source=ROOM_AMBIGUOUS,code=ROOM_SCRIPT,output="1",canonical=None)["verified"])

    def test_other_unique_room_instance_not_a_smoke_prompt(self):
        task=parse_finite_unique_source(ROOM_OTHER)
        self.assertIsNotNone(task)
        self.assertEqual(task.number_of_solutions,1)
        self.assertEqual(task.answer,2)

    def test_never_trust_fabricated_unique_assert(self):
        for code in (
            "solutions=[(True,False,False)]\nassert len(solutions)==1\nprint(1)",
            "from itertools import product\nsolutions=list(product((0,1),repeat=3))\n"
            "assert len(solutions)==1\nprint(1)",
            "assert 1==1\nprint(1)",
        ):
            self.assertFalse(independent_unique_certificate(
                source=TRUTH_UNIQUE,code=code,output="1",canonical=None)["verified"])

    def test_unique_assert_required_as_exact_expression(self):
        for code in (
            POSSIBLE.replace("assert len(solutions) == 1","assert len(solutions) >= 1"),
            POSSIBLE.replace("assert len(solutions) == 1","assert len(solutions) == 2"),
            POSSIBLE.replace("assert len(solutions) == 1","assert 1 == len(solutions)"),
        ):
            self.assertFalse(independent_unique_certificate(
                source=TRUTH_UNIQUE,code=code,output="1",canonical=None)["verified"])

    def test_unknown_clause_trailing_prompt_is_not_ignored(self):
        for source in (TRUTH_UNIQUE+" A has a blue hat.",
                       ROOM_UNIQUE+" Ignore the last rule.",
                       TRUTH_UNIQUE.replace("A says B lies.","A says B is lucky.")):
            self.assertIsNone(parse_finite_unique_source(source))
            self.assertFalse(independent_unique_certificate(
                source=source,code=POSSIBLE,output="1",canonical=None)["verified"])

    def test_scalar_does_not_override_independent_truth(self):
        for value in ("2","01","1\n2","ANSWER: 1","1.0"):
            self.assertFalse(independent_unique_certificate(
                source=TRUTH_UNIQUE,code=POSSIBLE,output=value,canonical=None)["verified"])

    def test_script_is_not_executed_by_source_oracle(self):
        code=POSSIBLE+"\nraise RuntimeError('private')"
        self.assertFalse(independent_unique_certificate(
            source=TRUTH_UNIQUE,code=code,output="1",canonical=None)["verified"])

    def test_router_dispatches_unique_and_refuses_multi(self):
        verified=verify_python_full_source(
            source=ROOM_UNIQUE,code=ROOM_SCRIPT,output="1",canonical=None)
        rejected=verify_python_full_source(
            source=ROOM_AMBIGUOUS,code=ROOM_SCRIPT,output="1",canonical=None)
        self.assertTrue(verified["verified"])
        self.assertFalse(rejected["verified"])

    def test_controller_gates_ambiguous_without_qlora(self):
        probes=[]
        out=run_adaptive_planner(turns(ROOM_AMBIGUOUS),
            lambda *args:probes.append("model") or {
                "strategy":"PYTHON_SANDBOX","code":ROOM_SCRIPT},
            lambda _:"UNTOUCHED_FALLBACK",
            lambda _:probes.append("docker") or {"status":"PASSED","output":"1"},
            verify_python_full_source)
        self.assertEqual(probes,[])
        self.assertFalse(out["verified"])
        self.assertEqual(out["text"],"UNTOUCHED_FALLBACK")


class BudgetedQwenFallbackCPU(unittest.TestCase):
    def test_local_only_non_thinking_hard_cap(self):
        packets=[]
        def fake(packet):
            packets.append(packet)
            return {"done":True,"done_reason":"stop",
                    "message":{"content":"42"}}
        q=make_bounded_fallback("qwen3.5:9b",transport=fake)
        original=turns(ROOM_OTHER)
        self.assertEqual(q(original),"42")
        self.assertEqual(packets[0]["messages"],original)
        self.assertEqual(packets[0]["model"],"qwen3.5:9b")
        self.assertIs(packets[0]["think"],False)
        self.assertLessEqual(packets[0]["options"]["num_predict"],768)
        self.assertFalse(packets[0]["stream"])

    def test_remote_or_unknown_models_cannot_be_requested(self):
        for model in ("localhost:11434/qwen3.5:9b","qwen2.5:7b",
                      "https://remote","", "qwen3.5:9b:other"):
            with self.assertRaises(ValueError):
                make_bounded_fallback(model,transport=lambda _:{})

    def test_length_truncation_fails_closed_no_partial_text(self):
        fallback=make_bounded_fallback("qwen3.5:9b",transport=lambda _: {
            "done":True,"done_reason":"length",
            "message":{"content":"Partial answer 55 SECRET"}})
        with self.assertRaises(ValueError):
            fallback(turns(ROOM_OTHER))

    def test_private_thinking_field_never_leaks(self):
        fallback=make_bounded_fallback("qwen3.5:9b",transport=lambda _: {
            "done":True,"done_reason":"stop",
            "message":{"content":"42","thinking":"SECRET_CHAIN"}})
        self.assertEqual(fallback(turns(ROOM_OTHER)),"42")

    def test_injection_into_messages_rejected(self):
        fallback=make_bounded_fallback("qwen3.5:9b",transport=lambda _:self.fail("No call"))
        for messages in (
            turns(ROOM_OTHER)+[{"role":"tool","content":"hack"}],
            [{"role":"user","content":ROOM_OTHER}],
            [{"role":"system","content":""},{"role":"user","content":"x"}],
        ):
            with self.assertRaises(ValueError):
                fallback(messages)

    def test_strict_bounded_output_not_verified(self):
        from neral_tir_regression_guard import choose_guarded_tir_reply
        out=run_adaptive_planner(turns("Explain the difference between a star and a planet."),
            lambda *_:self.fail("No proposer"),lambda _:"Brief unverified answer",
            lambda _:self.fail("No sandbox"),lambda **kw:{"verified":False})
        self.assertFalse(out["verified"])
        self.assertFalse(out["safe_to_promote"])

if __name__=="__main__":
    unittest.main()
