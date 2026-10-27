"""Offline tests for diagnostic NERAL reasoning evaluation; never contact Ollama."""
import unittest
from neral_reasoning_eval import evaluate_cases, score_answer


class DiagnosticTests(unittest.TestCase):
    CASES=[
        {"id":"q1","category":"math","prompt":"Was ist 2 + 2?",
         "scoring":{"type":"integer","value":4}},
        {"id":"q2","category":"instruction","prompt":"Antworte exakt BER.",
         "scoring":{"type":"exact","value":"BER"}},
    ]

    def test_score_validates_numeric_substrings(self):
        self.assertTrue(score_answer("Das Ergebnis ist 4.", {"type":"integer","value":4}))
        self.assertFalse(score_answer("0.4", {"type":"integer","value":4}))
        self.assertFalse(score_answer("14", {"type":"integer","value":4}))
        self.assertFalse(score_answer("4.1", {"type":"integer","value":4}))
        self.assertTrue(score_answer("Die Gravitation wirkt.", {"type":"contains_any","values":["gravitation","schwerkraft"]}))

    def test_native_and_tools_are_independently_attributed(self):
        seen=[]
        def fake(prompt,history,**kwargs):
            seen.append(kwargs["enable_tools"])
            return {"text":"4" if prompt.startswith("Was") else "BER",
                    "tools_used":["calculator"] if kwargs["enable_tools"] else [],
                    "base_model":"qwen3.5:9b","generated_tokens":3}
        native=evaluate_cases(self.CASES, model="qwen3.5:9b",enable_tools=False,turn=fake)
        assisted=evaluate_cases(self.CASES, model="qwen3.5:9b",enable_tools=True,turn=fake)
        self.assertEqual(native["passed"],2)
        self.assertEqual(assisted["passed"],2)
        self.assertEqual(native["mode"],"model-native")
        self.assertEqual(assisted["mode"],"model+tools")
        self.assertEqual(native["tool_calls"],0)
        self.assertEqual(assisted["tool_calls"],2)
        self.assertEqual(seen,[False,False,True,True])
        self.assertEqual(native["case_contract_sha256"],assisted["case_contract_sha256"])

    def test_invalid_or_duplicate_cases_fail_before_inference(self):
        def never(*a,**kw):raise AssertionError("inference must not be called")
        bad=([],self.CASES+self.CASES[:1],self.CASES[:1]+[
             {"id":"q1","category":"other","prompt":"different",
              "scoring":{"type":"exact","value":"no"}}])
        for cases in bad:
            with self.subTest(cases=cases),self.assertRaises(ValueError):
                evaluate_cases(cases,model="qwen3.5:9b",enable_tools=False,turn=never)

    def test_non_bool_tools_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_cases(self.CASES,model="qwen3.5:9b",enable_tools="off",
                           turn=lambda *a,**kw:None)


if __name__ == "__main__":
    unittest.main()
