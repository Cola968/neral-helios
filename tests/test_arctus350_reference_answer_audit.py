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

if __name__=="__main__":
    unittest.main()
