"""RED/GREEN: source-agnostic planning, independent post verification and final format.

No case from the burned e80c6017... GPU corpus is embedded here.
"""
import unittest
from neral_adaptive_planner import run_adaptive_planner,deterministic_prefilter
from neral_python_source_router import verify_python_full_source
from scripts.neral_blind_100_offline_score import conservative_final_integer

def turns(q):
    return [{"role":"system","content":"Local benchmark response"},
            {"role":"user","content":q}]

class OpenRouterCPU(unittest.TestCase):
    def test_computational_paraphrases_route_to_qlora_without_source_preparse(self):
        for q in (
           "What's 43 raised to power 58, modulo 97?",
           "What exact number comes out of this Python program?",
           "Evaluate the number of arrangements of eight labelled rooks.",
           "How many combinations satisfy these Boolean restrictions?",
        ):
            calls=[]
            result=run_adaptive_planner(
                turns(q),
                lambda messages,signal:(calls.append(signal) or
                  {"strategy":"DIRECT"}),
                lambda messages:"RAW_QWEN",
                lambda code:self.fail("No Docker for DIRECT"),
                lambda **kwargs:{"verified":False})
            self.assertEqual(len(calls),1,q)
            self.assertEqual(result["text"],"RAW_QWEN")

    def test_generated_assertion_is_never_independent_evidence(self):
        question="Count all bounded arrangements not matching an input grammar."
        calls=[]
        code="answer=41\nassert answer==40+1\nprint(answer)"
        result=run_adaptive_planner(
            turns(question),
            lambda messages,signal:{"strategy":"PYTHON_SANDBOX","code":code},
            lambda messages:"RAW_QWEN",
            lambda script:calls.append(script) or {
                "status":"PASSED","exit_code":0,"output":"41"},
            lambda **kwargs:{"verified":False,"reason":"unsupported_source"})
        self.assertEqual(len(calls),1)
        self.assertEqual(result["text"],"RAW_QWEN")
        self.assertFalse(result["safe_to_promote"])

    def test_model_z3_without_source_certificate_must_not_promote(self):
        calls=[]
        result=run_adaptive_planner(
            turns("Count Boolean assignments with some unknown condition."),
            lambda m,s:calls.append(s) or {
               "strategy":"Z3_SCHEMA","schema":{"version":2}},
            lambda m:"RAW_QWEN",
            lambda c:self.fail("No code for Z3"),
            lambda **kw:{"verified":True,"answer":99},
            verify_z3=lambda **kw:{"verified":True,"answer":99,
                 "proof_backend":"z3_unsat",
                 "verification_scope":"full_source_grammar_exact",
                 "verification_status":"source_canonical_z3_verified"})
        self.assertEqual(len(calls),2)  # exactly one bounded Z3->Python repair
        self.assertFalse(result["verified"])
        self.assertEqual(result["text"],"RAW_QWEN")

    def test_noncomputational_dialogue_remains_direct(self):
        calls=[]
        result=run_adaptive_planner(turns("Good evening, how are you?"),
            lambda *a:calls.append("qlora") or {"strategy":"DIRECT"},
            lambda m:"RAW_QWEN",
            lambda c:calls.append("docker") or {},lambda **kw:{})
        self.assertEqual(calls,[])
        self.assertEqual(result["text"],"RAW_QWEN")

class FormatParserCPU(unittest.TestCase):
    def test_final_answer_terminal_parses_with_or_without_reasons(self):
        for v in ("FINAL_ANSWER: 91",
                  "step 1: 8+12=20\nFINAL_ANSWER: 91",
                  "Earlier 73, now 91.\nFINAL_ANSWER: 91"):
            self.assertEqual(conservative_final_integer(v),91)
        self.assertEqual(conservative_final_integer("FINAL_ANSWER: -7"),-7)

    def test_final_answer_never_accepts_conflicts_or_trailing_content(self):
        for v in ("FINAL_ANSWER: 13\nFINAL_ANSWER: 14",
                  "FINAL_ANSWER: 13\nAnd another answer 13",
                  "FINAL_ANSWER: 13\nmaybe",
                  "We might have FINAL_ANSWER: 13",
                  "<think>secret</think>\nFINAL_ANSWER: 13"):
            with self.subTest(v=v):
                self.assertIsNone(conservative_final_integer(v))

class PostSourceCPU(unittest.TestCase):
    def test_nontrivial_known_math_paraphrase_has_independent_oracle(self):
        q="Find the remainder when 17 to the power 45 is divided by 97."
        code="x=pow(17,45,97)\nassert 0<=x<97\nprint(x)"
        result=verify_python_full_source(
            source=q,code=code,output=str(pow(17,45,97)),canonical=None)
        self.assertTrue(result["verified"],result)

    def test_programming_original_program_runs_with_separate_docker(self):
        code="values=[2,3,4]\ntotal=sum(values)\nassert total==9\nprint(total)"
        q=("What exact integer does this unmodified Python 3 code print?\n"
           "```python\nvalues=[2,3,4]\nprint(sum(values))\n```")
        runs=[]
        from unittest.mock import patch
        with patch("neral_adaptive_planner.run_python_in_docker",
                   side_effect=lambda p:runs.append(p) or {
                       "status":"PASSED","exit_code":0,"output":"9"}):
            cert=verify_python_full_source(
                source=q,code=code,output="9",canonical=None)
        self.assertTrue(cert["verified"],cert)
        self.assertEqual(len(runs),1)
        self.assertIn("print(sum(values))",runs[0])

    def test_incorrect_script_or_unsourced_source_cannot_be_promoted(self):
        q="Find the remainder when 17 to the power 45 is divided by 97."
        code="x=25\nassert x==5*5\nprint(x)"
        cert=verify_python_full_source(source=q,code=code,output="25",canonical=None)
        self.assertFalse(cert["verified"])

if __name__=="__main__":unittest.main()
