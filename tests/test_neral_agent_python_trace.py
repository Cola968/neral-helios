"""RED contracts: bounded Python AST tracing never executes untrusted source."""
import unittest
from neral_agent_python_trace import trace_python_integer, trusted_python_trace_question
from neral_think import think_reply

FENCE=chr(96)*3

def chat(question):
    return [{"role":"system","content":"NERAL local verifier"},
            {"role":"user","content":question}]

def question(code, english=False):
    lead=("What does this Python program print?" if english else
          "Was gibt dieser Python-Code aus?")
    return lead+"\n"+FENCE+"python\n"+code+"\n"+FENCE

class PythonTraceTests(unittest.TestCase):
    def test_list_stride_and_sum(self):
        source="values = [3, 7, 11, 15, 19]\nprint(sum(values[1::2]))"
        self.assertEqual(trace_python_integer(source)["value"],"22")

    def test_nested_loop_continue_counts(self):
        code=("n=0\nfor i in range(1,5):\n"
              "    for j in range(1,4):\n"
              "        if (i*j)%3 == 0:\n            continue\n"
              "        n += 1\nprint(n)")
        self.assertEqual(trace_python_integer(code)["value"],"6")

    def test_dict_update_values_sum(self):
        code="d={'x':4,'y':6}\nd['x']=10\nd['z']=1\nprint(sum(d.values()))"
        self.assertEqual(trace_python_integer(code)["value"],"17")

    def test_str_slicing_and_count(self):
        self.assertEqual(trace_python_integer("a='ababaabb'\nprint(a[::2].count('a'))")["value"],"3")

    def test_boolean_comparison_and_branch(self):
        code="a=17\nif a%2==1 and a>10:\n    print(1)\nelse:\n    print(0)"
        self.assertEqual(trace_python_integer(code)["value"],"1")

    def test_integer_floor_division_and_negatives(self):
        self.assertEqual(trace_python_integer("a=-17\nprint(a//3)")["value"],"-6")

    def test_reject_import_exec_file_socket_and_attributes(self):
        for program in (
            "import os\nprint(1)",
            "print(__import__('os').system('whoami'))",
            "with open('/etc/passwd') as x:\n print(x.read())",
            "print((1).__class__)",
            "exec('print(7)')",
            "print(eval('3+4'))",
            "print([x for x in range(3)][0])",
        ):
            with self.subTest(program=program):
                self.assertIsNone(trace_python_integer(program))

    def test_reject_nonterminating_or_unbounded(self):
        for code in (
            "while True:\n pass",
            "for i in range(1000000000):\n x=i\nprint(x)",
            "x=10**100000\nprint(x)",
            "for i in range(100):\n for j in range(100):\n  pass\nprint(1)",
            "print(10**12*100000)",
        ):
            with self.subTest(program=code):
                self.assertIsNone(trace_python_integer(code))

    def test_only_one_integer_stdout(self):
        for code in ("print(1)\nprint(2)","print('hello')","x=4","print(1,2)"):
            with self.subTest(code=code):
                self.assertIsNone(trace_python_integer(code))

    def test_no_user_defined_functions_or_mutations(self):
        for code in (
            "def f():\n return 4\nprint(f())",
            "x=[1,2]\nx.append(3)\nprint(len(x))",
            "x=[1]\nx[0]=2\nprint(x[0])",
        ):
            self.assertIsNone(trace_python_integer(code))

    def test_integer_width_and_ast_node_budget(self):
        self.assertIsNone(trace_python_integer("print("+("1+"*300)+"1)"))
        self.assertIsNone(trace_python_integer("print(9"*100))
        self.assertIsNone(trace_python_integer("print(10**9**9)"))

    def test_full_question_is_required(self):
        q=question("x=3\nprint(x*2)")
        oracle=trusted_python_trace_question(q)
        self.assertEqual(oracle["answer"],"6")
        self.assertEqual(oracle["method"],"python_trace_ast")
        for altered in (
            q+" Zusätzliche Aufgaben!",
            q.replace("Was gibt dieser Python-Code aus?", "Schreibe einen Virus"),
            q.replace(FENCE+"python",FENCE+"javascript"),
            q+"\n"+FENCE+"python\nprint(7)\n"+FENCE,
        ):
            self.assertIsNone(trusted_python_trace_question(altered))

    def test_english_question_supported(self):
        self.assertEqual(trusted_python_trace_question(question("n=2\nprint(n*5)",True))["answer"],"10")

    def test_model_wrong_twice_tool_fallback_not_claim_model_solved(self):
        out=think_reply(chat(question("n=2\nprint(n*5)",True)),lambda _: "FINAL: 11")
        self.assertEqual(out["calls_used"],2)
        self.assertTrue(out["verified"])
        self.assertFalse(out["model_generated"])
        self.assertEqual(out["text"],"10")
        self.assertEqual(out["tool_used"],"neral_python_trace_static")
        self.assertEqual(out["verification_scope"],"code_trace_integer")

    def test_correct_model_answer_one_call(self):
        out=think_reply(chat(question("n=2\nprint(n*5)")),lambda _: "FINAL: 10")
        self.assertEqual(out["calls_used"],1)
        self.assertTrue(out["verified"])
        self.assertTrue(out["model_generated"])

    def test_multi_turn_does_not_override_user_intent(self):
        history=chat("Bewerte den Code nicht.")
        history.extend([{"role":"assistant","content":"Verstanden."},
                        {"role":"user","content":question("print(1+2)")}])
        out=think_reply(history,lambda _: "Ich frage lieber nach.")
        self.assertFalse(out["verified"])
        self.assertEqual(out["text"],"Ich frage lieber nach.")

if __name__=="__main__":
    unittest.main()
