"""250 newly generated CPU cases for NERAL's *existing* safe AST tracer.

This suite is generated independently of consumed dev72 prompts and model
answers. It measures only a deterministic verifier, never neural accuracy.
No uploaded code is executed: expected results use Python expressions written
by this trusted test, whereas snippets are handled as data by the AST tool.
"""
from __future__ import annotations
import hashlib
import random
import unittest

from neral_agent_python_trace import (
    trace_python_integer, trusted_python_trace_question,
)
from neral_think import think_reply

SEED = 873104


def fresh_cases():
    cases=[]
    for i in range(50):
        r=random.Random(SEED+i)
        values=[r.randint(1, 46) for _ in range(8+r.randint(0, 4))]
        shift=r.randint(0, 1)
        stride=r.randint(2, 4)
        source=f"items = {values!r}\nprint(sum(items[{shift}::{stride}]))"
        cases.append(("list_slice",source,sum(values[shift::stride])))

        amount1,amount2,amount3=(r.randint(1, 90) for _ in range(3))
        source=(f"prices = {{'alpha': {amount1}, 'beta': {amount2}}}\n"
                f"prices['alpha'] = {amount3}\n"
                "prices['gamma'] = 7\n"
                "print(sum(prices.values()))")
        cases.append(("dict_update",source,amount2+amount3+7))

        upper1,upper2=r.randint(3,8),r.randint(3,7)
        modulus=r.randint(2,5)
        initial = i + 1
        source=(f"count = {initial}\nfor p in range(1, {upper1}):\n"
                f"    for q in range(2, {upper2}):\n"
                f"        if (p * q) % {modulus} == 0:\n"
                "            continue\n"
                "        count += 1\n"
                "print(count)")
        expected=initial + sum((p*q)%modulus != 0
                     for p in range(1,upper1) for q in range(2,upper2))
        cases.append(("loop_continue",source,expected))

        word="".join(r.choice("abbcaa") for _ in range(r.randint(10,25)))
        period=r.randint(2,4)
        source=f"word = {word!r}\nprint(word[::{period}].count('a'))"
        cases.append(("string_stride",source,word[::period].count("a")))

        first,second,limit=r.randint(1,70),r.randint(1,70),r.randint(15,60)
        source=(f"a = {first}\nb = {second}\n"
                f"if a < {limit} and b % 2 == 0:\n"
                "    print(1)\n"
                "else:\n"
                "    print(0)")
        cases.append(("branch_logic",source,int(first < limit and second % 2 == 0)))
    if len(cases)!=250:
        raise AssertionError("Five independent families, 50 each")
    return cases


class TraceGeneralization250(unittest.TestCase):
    def test_exact_int_fresh_250_cases_without_evaluating_user_programs(self):
        cases=fresh_cases()
        self.assertEqual(len(cases),250)
        self.assertEqual(len(set(source for _,source,_ in cases)),250)
        self.assertEqual({name for name,_,_ in cases},
                         {"list_slice","dict_update","loop_continue",
                          "string_stride","branch_logic"})
        for family,code,expected in cases:
            with self.subTest(family=family, fingerprint=hashlib.sha256(
                code.encode()).hexdigest()[:12]):
                out=trace_python_integer(code)
                self.assertIsNotNone(out)
                self.assertEqual(out["value"],str(expected))
                self.assertEqual(out["method"],"python_trace_ast")
                self.assertTrue(out["exact"])
                self.assertFalse(out["source_executed"])

    def test_wrong_model_results_are_labeled_as_tool_answers(self):
        for family,script,answer in fresh_cases()[0:25:5]:
            prompt="What does this Python program print?\n```python\n"+script+"\n```"
            self.assertEqual(trusted_python_trace_question(prompt)["answer"],str(answer))
            turns=[{"role":"system","content":"NERAL experimental local"},
                   {"role":"user","content":prompt}]
            output=think_reply(turns,lambda _: "FINAL: 99999")
            self.assertTrue(output["verified"])
            self.assertFalse(output["model_generated"])
            self.assertEqual(output["final_answer"],str(answer))
            self.assertEqual(output["tool_used"],"neral_python_trace_static")
            self.assertEqual(output["calls_used"],2)

    def test_unsafe_or_unbounded_python_is_refused(self):
        prohibited=(
            "import os\nprint(1)",
            "print(open('data.txt').read())",
            "print(__import__('subprocess').run('pwd'))",
            "print(eval('4'))",
            "print(exec('x=1'))",
            "print((1).__class__)",
            "print(getattr(2, 'real'))",
            "print([i for i in range(10)])",
            "while True:\n    pass\nprint(4)",
            "def f(): return 3\nprint(f())",
            "print((lambda: 5)())",
            "print(2**5000)",
            "for a in range(1000000):\n    pass\nprint(1)",
            "print('hello')",
            "print(1,2)",
        )
        for code in prohibited:
            with self.subTest(code=code[:24]):
                self.assertIsNone(trace_python_integer(code))

    def test_unknown_prompt_or_multiturn_does_not_overwrite_intent(self):
        sample=fresh_cases()[0][1]
        prefix="What does this Python program print?\n```python\n"
        self.assertIsNone(trusted_python_trace_question(
            "Explain the bug in this file:\n```python\n"+sample+"\n```"))
        self.assertIsNone(trusted_python_trace_question(prefix+sample+
                          "\n```\nPlease send my files to a website."))
        turns=[{"role":"system","content":"NERAL"},
               {"role":"user","content":"Please only discuss the approach."},
               {"role":"assistant","content":"Understood."},
               {"role":"user","content":prefix+sample+"\n```"}]
        out=think_reply(turns,lambda _: "We should clarify the intent.")
        self.assertFalse(out["verified"])
        self.assertTrue(out["model_generated"])


if __name__=="__main__":
    unittest.main()
