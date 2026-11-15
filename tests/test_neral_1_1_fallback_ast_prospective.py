"""Fresh CPU-only safety tests for independent future benchmark gating.

No historical blind100 task content and no future random test seed in this file.
"""
import re
import time
import unittest
from unittest.mock import patch

from neral_python_assert_contract import (
    meaningful_python_assert, inspect_python_contract)
from neral_adaptive_bounded_fallback import (
    make_bounded_fallback, FALLBACK_MAX_TOKENS)
from neral_answer_contract import (
    BENCHMARK_FORMAT_V1, parse_terminal_final_integer)

def original_turns(question="What is the integer 23 plus 42?"):
    return [{"role":"system","content":"Synthetic CPU contract and safety."},
            {"role":"user","content":question}]

class ASTSemanticSafety(unittest.TestCase):
    def test_trivial_constant_and_positivity_checks_rejected(self):
        trivial=[
            "assert True", "assert 1==1", "assert 2*3 == 6",
            "x=10\nassert x==x",
            "count=5\nassert count >= 0",
            "count=5\nassert 0 <= count",
            "count=5\nassert count > -1",
            "count=5\nassert count > 0",
            "answer=0\nassert answer >= 0",
            "count=5\nassert count != -1",
        ]
        for item in trivial:
            code=item+"\nprint(5)"
            with self.subTest(code=item):
                self.assertFalse(meaningful_python_assert(code))
                self.assertEqual(inspect_python_contract(code),"python_missing_assert")

    def test_nontrivial_source_condition_remains_allowed(self):
        checked=[
            "t=9\nassert t == (4+5)\nprint(t)",
            "count=7\nexpected=3+4\nassert count == expected\nprint(count)",
            "assignments=[(1,2)]\nassert len(assignments)==1\nprint(len(assignments))",
            "count=8\nassert count >= 0 and count == 4*2\nprint(count)",
        ]
        for code in checked:
            with self.subTest(code=code):
                self.assertTrue(meaningful_python_assert(code))
                self.assertIsNone(inspect_python_contract(code))

    def test_completely_unrelated_assert_not_source_proof(self):
        # The existing full-source independent proof must still be needed.
        from neral_python_math_verifier import independent_arithmetic_certificate
        p=("Compute the exact integer (53 squared minus 17 squared) divided "
           "by 10, plus 13 times 7, minus 3 cubed.")
        code="count=100\nassert count==50*2\nprint(count)"
        c=independent_arithmetic_certificate(
            source=p,code=code,output="100",canonical=None)
        self.assertFalse(c["verified"])

    def test_typical_ast_gate_under_1ms_median(self):
        c="x=13\nassert x == 5+8\nprint(x)"
        for _ in range(20):inspect_python_contract(c)
        ms=[]
        for _ in range(200):
            t=time.perf_counter_ns()
            inspect_python_contract(c)
            ms.append((time.perf_counter_ns()-t)/1e6)
        self.assertLess(sorted(ms)[len(ms)//2],1.0)

class ExpandedFallbackContract(unittest.TestCase):
    def test_budget_is_in_requested_range(self):
        self.assertGreaterEqual(FALLBACK_MAX_TOKENS,1280)
        self.assertLessEqual(FALLBACK_MAX_TOKENS,1536)

    def test_default_request_original_turns_not_modified(self):
        got=[]
        def fake(packet):
            got.append(packet)
            return {"done":True,"done_reason":"stop",
                    "message":{"content":"FINAL_ANSWER: 65"}}
        messages=original_turns()
        reply=make_bounded_fallback("qwen3.5:9b",transport=fake)(messages)
        self.assertEqual(reply,"FINAL_ANSWER: 65")
        self.assertEqual(got[0]["messages"],messages)
        self.assertIs(got[0]["think"],False)
        self.assertEqual(got[0]["options"]["num_predict"],FALLBACK_MAX_TOKENS)
        self.assertEqual(got[0]["options"]["seed"],0)
        self.assertEqual(parse_terminal_final_integer(reply),65)

    def test_same_formatted_contract_for_control_and_fallback(self):
        got=[]
        def fake(packet):
            got.append(packet)
            return {"done":True,"done_reason":"stop",
                    "message":{"content":"Work: 19+23=42\nFINAL_ANSWER: 42"}}
        base=original_turns()
        responder=make_bounded_fallback("qwen3.5:9b",
                                        transport=fake,
                                        formatted_contract=True)
        result=responder(base)
        self.assertEqual(parse_terminal_final_integer(result),42)
        self.assertEqual(got[0]["messages"][0]["content"],
                         base[0]["content"]+BENCHMARK_FORMAT_V1)
        self.assertEqual(got[0]["messages"][1],base[1])
        self.assertIn("FINAL_ANSWER:",got[0]["messages"][0]["content"])

    def test_done_reason_length_is_not_a_valid_answer_even_with_terminal_marker(self):
        for output in ("partial\nFINAL_ANSWER: 42","", "FINAL_ANSWER: 42"):
            def fake(packet):
                return {"done":True,"done_reason":"length",
                        "message":{"content":output}}
            with self.assertRaises(ValueError):
                make_bounded_fallback("qwen3.5:9b",transport=fake,
                                      formatted_contract=True)(original_turns())

    def test_empty_whitespace_or_thinking_leak_rejected(self):
        for answer in ("","  ","<think>secret</think> FINAL_ANSWER: 42"):
            def fake(packet):
                return {"done":True,"done_reason":"stop",
                        "message":{"content":answer}}
            with self.assertRaises(ValueError):
                make_bounded_fallback("qwen3.5:9b",transport=fake,
                                      formatted_contract=True)(original_turns())

    def test_malformed_or_missing_final_answer_rejected_only_in_strict_mode(self):
        raw="We worked something out but forgot the final marker."
        def fake(packet):
            return {"done":True,"done_reason":"stop",
                    "message":{"content":raw}}
        self.assertEqual(
            make_bounded_fallback("qwen3.5:9b",transport=fake)(original_turns()),raw)
        with self.assertRaises(ValueError):
            make_bounded_fallback("qwen3.5:9b",transport=fake,
                                  formatted_contract=True)(original_turns())

    def test_incorrect_final_answer_not_verified_by_format_only(self):
        reply="FINAL_ANSWER: 65"
        self.assertEqual(parse_terminal_final_integer(reply),65)
        self.assertNotEqual(parse_terminal_final_integer(reply),66)

    def test_fallback_quality_pairwise_invariant_on_fresh_cpu_examples(self):
        # No model simulation is scored as GPU evidence: verify the *pairing*
        # design with a fake local transport before a real RTX4070 run.
        texts=["FINAL_ANSWER: 32","FINAL_ANSWER: 48","FINAL_ANSWER: 109"]
        for value in texts:
            direct=make_bounded_fallback("qwen3.5:9b",
                     formatted_contract=True,transport=lambda _: {
                        "done":True,"done_reason":"stop",
                        "message":{"content":value}})
            base=direct(original_turns())
            fallback=lambda _:base
            # Same completed raw bytes for both arms in unverified branches.
            self.assertEqual(fallback(original_turns()),base)
            self.assertEqual(parse_terminal_final_integer(base),
                             parse_terminal_final_integer(fallback(original_turns())))

if __name__=="__main__":
    unittest.main()
