"""Read-only numerical and AST reference checks; no arbitrary code execution."""
from __future__ import annotations
import json
import unittest
from frontier350.arctus350_reference_answer_audit import inspect_reference, audit_splits

class ReferenceAnswerAuditTests(unittest.TestCase):
    def check(self,domain,prompt,answer,family,status):
        x=inspect_reference(domain,prompt,answer)
        self.assertEqual((x["family"],x["status"]),(family,status))

    def test_bilingual_arithmetic(self):
        self.check("math","Berechne 23 + 17 und erkläre den Rechenweg kurz.",
                   "23 + 17 = 40. Das Ergebnis ist 40.","arithmetic","pass")
        self.check("math","Compute 23 - 17. Show a short calculation.",
                   "23 - 17 = 7.","arithmetic","fail")
        self.check("math","Compute 23 - 17. Show a short calculation.",
                   "23 - 17 = 6.","arithmetic","pass")

    def test_multiplication(self):
        self.check("math","Wie viel ist 7 mal 8? Begründe knapp.",
                   "7 · 8 = 56.","multiplication","pass")
        self.check("math","What is 7 times 8? Give a concise calculation.",
                   "7 × 8 = 96.","multiplication","fail")

    def test_equation_percent_fraction_force_sequence(self):
        self.check("math","Löse die Gleichung 3x + (-2) = 10 nach x und zeige die Umformung.",
                   "3x = 12. Durch 3 teilen: x = 4.","linear_equation","pass")
        self.check("math","Find 25% of 80 and explain the calculation.",
                   "25% = 25/100, so 80 × 25/100 = 21.","percentage","fail")
        self.check("math","Berechne 2/3 · 3 und kürze vollständig.",
                   "2/3 · 3 = 6/3 = 2.","fraction_product","pass")
        self.check("math","Use F = m·a. What force acts for m = 4 kg and a = 5 m/s²? Show formula and result.",
                   "F = m·a = 4 kg × 5 m/s² = 20 N.","force","pass")
        self.check("math","Continue the sequence 1, 3, 5, 7, ... and state the rule.",
                   "The difference is always 2. The next number is 10.","arithmetic_sequence","fail")

    def test_code_static_ast_without_execution(self):
        tick=chr(96)
        p="Write a short Python function "+tick+"add(a, b)"+tick+" that adds two numbers. Show it with 3 and 4."
        fence=tick*3
        self.check("code",p,fence+"python\ndef add(a, b):\n    return a + b\n\nprint(add(3, 4))  # 7\n"+fence,"python_add","pass")
        self.check("code",p,fence+"python\ndef add(a, b):\n    return a - b\n\nprint(add(3, 4))  # -1\n"+fence,"python_add","fail")
        self.check("code",p,fence+"python\ndef add(a, b):\n    return a + b\n\nprint(add(3, 4))  # 10\n"+fence,"python_add","fail")
        self.check("code",p,"Cannot write that.","python_add","unverified")

    def test_absolute_difference_format(self):
        p="Formuliere präziser: 'Die Werte 10 und -3 sind irgendwie unterschiedlich.' Nenne die absolute Differenz."
        self.check("format",p,"Die beiden Werte unterscheiden sich um 13.","absolute_difference","pass")
        self.check("format",p,"Die beiden Werte unterscheiden sich um 7.","absolute_difference","fail")

    def test_unsupported_not_falsely_passed(self):
        self.check("math","Erkläre den Satz des Pythagoras.","a² + b² = c².","unsupported","unverified")
        self.check("code","Execute arbitrary Python code","print('hi')","unsupported","unverified")

    def test_privacy_and_aggregates(self):
        r={"id":"SECRETROW","domain":"math","language":"de",
           "messages":[{"role":"user","content":"Berechne 2 + 3. SECRET_PROMPT"},
                       {"role":"assistant","content":"2 + 3 = 4. SECRET_ANSWER"}]}
        x=audit_splits({s:[r] for s in ("train","val","test")})
        self.assertEqual(x["splits"]["train"]["status"]["fail"],1)
        self.assertEqual(x["splits"]["train"]["by_family"]["arithmetic"]["fail"],1)
        for secret in ("SECRETROW","SECRET_PROMPT","SECRET_ANSWER"):
            self.assertNotIn(secret,json.dumps(x))

    def test_multiturn_requires_manual_review(self):
        r={"domain":"math","language":"de",
           "messages":[{"role":"user","content":"Berechne 2 + 3."},
                       {"role":"assistant","content":"2 + 3 = 5."},
                       {"role":"user","content":"Was ist 4 + 8?"},
                       {"role":"assistant","content":"12"}]}
        x=audit_splits({s:[r] for s in ("train","val","test")})
        self.assertEqual(x["splits"]["train"]["status"]["unverified"],1)

    def test_range_and_sum_reference_checks(self):
        tick=chr(96)
        self.check("code","Briefly explain what "+tick+"list(range(2, 5))"+tick+" returns in Python.",
                   tick+"range(2, 5)"+tick+" contains the integers from 2 through 4. The list is "+tick+"[2, 3, 4]"+tick+".","python_range","pass")
        self.check("code","Briefly explain what "+tick+"list(range(2, 5))"+tick+" returns in Python.",
                   tick+"range(2, 5)"+tick+" contains the integers from 2 through 4. The list is "+tick+"[2, 3, 9]"+tick+".","python_range","fail")
        self.check("code","What does this Python code print? "+tick+"values = [2, 3, 4]; print(sum(values))"+tick,
                   tick+"sum(values)"+tick+" adds all elements. The output is "+tick+"9"+tick+".","python_sum","pass")

    def test_python_equality_repair(self):
        tick=chr(96)
        p="Fix the Python error and explain it: "+tick+"if count = 12: print(count)"+tick
        self.check("code",p,"A comparison uses "+tick+"=="+tick+": "+tick+"if count == 12: print(count)"+tick+". A single "+tick+"="+tick+" is assignment.","python_equality_fix","pass")
        self.check("code",p,"A comparison uses "+tick+"=="+tick+": "+tick+"if count == 13: print(count)"+tick+".","python_equality_fix","fail")

    def test_python_even_ast_and_incorrect_comment(self):
        tick=chr(96)
        p="Write Python code that checks whether "+tick+"count = 4"+tick+" is even."
        fence=tick*3
        self.check("code",p,fence+"python\ncount = 4\nis_even = count % 2 == 0\nprint(is_even)\n"+fence,"python_even","pass")
        self.check("code",p,fence+"python\ncount = 4\nis_even = count % 2 != 0\nprint(is_even)\n"+fence,"python_even","fail")

    def test_python_max_and_square_ast(self):
        tick=chr(96)
        fence=tick*3
        p="Write a Python function "+tick+"larger(a, b)"+tick+" that returns the larger value, and show a call with 8 and 2."
        self.check("code",p,fence+"python\ndef larger(a, b):\n    return a if a >= b else b\n\nprint(larger(8, 2))  # 8\n"+fence,"python_max","pass")
        self.check("code",p,fence+"python\ndef larger(a, b):\n    return a if a >= b else b\n\nprint(larger(8, 2))  # 2\n"+fence,"python_max","fail")
        q="Write a Python function "+tick+"square(number)"+tick+" that returns the square. Show the example number = -3."
        self.check("code",q,fence+"python\ndef square(number):\n    return number * number\n\nprint(square(-3))  # 9\n"+fence,"python_square","pass")

    def test_python_dict_static_shape(self):
        tick=chr(96)
        fence=tick*3
        p="Create a Python dictionary with key "+tick+"gate"+tick+" and value 123, then read the value back."
        self.check("code",p,fence+"python\ndata = {'gate': 123}\nprint(data['gate'])  # 123\n"+fence,"python_dictionary","pass")
        self.check("code",p,fence+"python\ndata = {'gate': 123}\nprint(data['gate'])  # 321\n"+fence,"python_dictionary","fail")

if __name__=="__main__":
    unittest.main()
