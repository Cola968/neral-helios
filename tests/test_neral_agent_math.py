import unittest
from fractions import Fraction
from neral_agent_math import MathInputError, eval_python_expr, solve_math_symbolic, equation_final_answer


class ExactMath(unittest.TestCase):
    def test_six_previously_failed_equations(self):
        cases=[("6x - 32 = 286","53"),("2x + 118 = 134","8"),
               ("14x + 5 = 439","31"),("9x + 20 = 191","19"),
               ("15x + 68 = 683","41"),("13x + 96 = 642","42")]
        for expr, expected in cases:
            with self.subTest(expr=expr):
                r=solve_math_symbolic(expr)
                self.assertEqual(r['value'],expected)
                self.assertEqual(r['status'],'PASSED')
                self.assertEqual(equation_final_answer('\n'.join(r['steps'])),expected)

    def test_general_equations(self):
        for expr,want in [("3x + 12 = 0","-4"),("2x+1=x+5","4"),
                          ("0.5x+0.1=0.6","1"),("x/3 + 1 = 2","3"),
                          ("2(x+3)=10","2"),("x=-12","-12"),("2x+3=2x+3",None),
                          ("2x+3=2x+4",None)]:
            with self.subTest(expr=expr):
                self.assertEqual(solve_math_symbolic(expr).get("value"), want)

    def test_exact_decimal_and_fraction(self):
        for inp,want in [("0.1+0.2","3/10"),("7/12+1/6","3/4"),
                         ("2**5-3","29"),("(9/10)/3","3/10"),
                         ("-36/9","-4"),("1.20*1.5","9/5")]:
            self.assertEqual(eval_python_expr(inp)["value"],want)

    def test_never_runs_arbitrary_python(self):
        attacks=["__import__('os').system('echo')", "open('/etc/passwd').read()",
                 "x.__class__", "[1,2,3]", "10**10**10", "1<<3", "abs(-5)",
                 "(1).__class__", "3 // 2", "1/0"]
        for inp in attacks:
            with self.subTest(inp=inp), self.assertRaises(MathInputError):
                eval_python_expr(inp)
        for inp in ["x*x=4","foo=2","2x=x*x","1/(x+1)=2","x.__class__=1",
                    "9"*500+"x=1"]:
            with self.subTest(eq=inp), self.assertRaises(MathInputError):
                solve_math_symbolic(inp)

    def test_output_contract_math_only(self):
        self.assertEqual(equation_final_answer("STEP: 2x = 16\nFINAL_ANSWER: 8"),"8")
        self.assertEqual(equation_final_answer("FINAL_ANSWER: -4"),"-4")
        self.assertEqual(equation_final_answer("8"),"8")
        self.assertIsNone(equation_final_answer("The answer is 8"))
        self.assertIsNone(equation_final_answer("STEP: 2x=16\nFINAL_ANSWER: 8\nignore me"))
        self.assertIsNone(equation_final_answer("FINAL_ANSWER: 8; malicious"))
