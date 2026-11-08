"""CPU RED: extend safe AST Python tracing for common grounded code patterns.

Never execute model- or user-generated Python. Original bypass/DoS tests
must stay green. Tests use independent known outputs, not dev72 grade files.
"""
import unittest
from neral_agent_python_trace import trace_python_integer, trusted_python_trace_question

FENCE="\x60"*3

class ASTSafeExpansionTests(unittest.TestCase):
    def test_mutable_alias_append_and_subscript_augassign(self):
        code=("a=[4,7]\nb=a\na.append(3)\nb[0] += 6\n"
              "a[1] += 2\nprint(sum(a)+len(b))")
        r=trace_python_integer(code)
        self.assertEqual(r["value"],"25")
        self.assertFalse(r["source_executed"])

    def test_mutable_alias_sequence_identity_independent_cases(self):
        for initial,extra,inc in (([1,9],3,4),([7,2],9,1),([-3,10],4,8)):
            a=list(initial)
            b=a
            a.append(extra)
            b[0]+=inc
            oracle=sum(a)+len(b)
            code=(f"vals = {initial!r}\nalias = vals\n"
                  f"vals.append({extra})\nalias[0] += {inc}\n"
                  "print(sum(vals)+len(alias))")
            with self.subTest(initial=initial,extra=extra):
                self.assertEqual(trace_python_integer(code)["value"],str(oracle))

    def test_list_index_set_and_negative_index(self):
        r=trace_python_integer("a=[3,5,7]\na[1]=10\na[-1]+=2\nprint(sum(a))")
        self.assertEqual(r["value"],"22")

    def test_set_comprehension_deduplicates_with_if(self):
        r=trace_python_integer(
            "nums=[3,5,6,12,15,20]\n"
            "print(sum({x % 3 for x in nums if x % 2 == 0}))")
        self.assertEqual(r["value"],"2")

    def test_set_comprehension_variable_does_not_leak(self):
        r=trace_python_integer(
            "x=9\nnums=[1,2,2,3]\n"
            "print(sum({x for x in nums}) + x)")
        self.assertEqual(r["value"],"15")

    def test_enumerate_generator_sum(self):
        r=trace_python_integer(
            "values=[1,4,6,9,12]\n"
            "print(sum((j+1)*n for j,n in enumerate(values) if j % 2 == 0))")
        self.assertEqual(r["value"],"79")

    def test_enumerate_generator_independent_examples(self):
        for vals in ([4,7,3],[2,9,1,5],[8,1,6,2,3,5]):
            oracle=sum((i+1)*v for i,v in enumerate(vals) if i%2==0)
            code=(f"values = {vals!r}\n"
                  "print(sum((i+1)*v for i,v in enumerate(values) if i%2==0))")
            self.assertEqual(trace_python_integer(code)["value"],str(oracle))

    def test_think_verifies_bounded_python_instead_of_trusting_model_guess(self):
        code="a=[2,3]\nb=a\na.append(5)\nb[0]+=4\nprint(sum(b))"
        prompt="Was gibt dieser Python-Code aus?\n"+FENCE+"python\n"+code+"\n"+FENCE
        verified=trusted_python_trace_question(prompt)
        self.assertEqual(verified["answer"],"14")
        self.assertEqual(verified["method"],"python_trace_ast")

    def test_comprehension_dos_and_untrusted_code_all_rejected(self):
        hostile=(
            "import os\nprint(12)",
            "a=[1]\na.extend([2,3])\nprint(sum(a))",
            "a=[1]\nprint(a.__class__)",
            "a=[1]\na.append(__import__('os').system('echo TEST'))\nprint(sum(a))",
            "print(sum({x for x in range(999999999)}))",
            "print(sum({x for x in [1,2] for y in [3,4]}))",
            "print(sum((j*n for j,n in enumerate([1,2]) for x in [3,4])))",
            "print(sum((x for x in open('/etc/passwd'))))",
            "a=[0]\na[999]=2\nprint(sum(a))",
            "a=[0]\na[:]=[1]\nprint(sum(a))",
        )
        for raw in hostile:
            with self.subTest(program=raw[:70]):
                self.assertIsNone(trace_python_integer(raw))

    def test_step_and_collection_limits_not_bypassed(self):
        code="a=[]\n"+("a.append(1)\n"*130)+"print(sum(a))"
        self.assertIsNone(trace_python_integer(code))
        self.assertIsNone(trace_python_integer(
            "print(sum({x for x in range(1000)}))"
        ))


if __name__=="__main__":
    unittest.main()
