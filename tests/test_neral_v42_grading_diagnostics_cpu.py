"""Grader diagnostic regression: retain strict scoring while naming failure mode."""
import unittest
from unittest.mock import patch
from scripts.neral_v41_blind200_live import grade_python, grade_response


class GraderDiagnosticsCPU(unittest.TestCase):
    def test_unparseable_numeric_answer_is_not_credited(self):
        case={"category":"math_logic","grader":"independent_exact",
              "sealed_reference":{"kind":"integer","value":"391"}}
        self.assertEqual(grade_response(case,{"text":"17 × 23 = 391. Ergebnis: 391."}),
                         ("fail","exact_output_unparseable"))
        self.assertEqual(grade_response(case,{"text":"Antwort: 391"}),
                         ("pass","independent_exact"))
        self.assertEqual(grade_response(case,{"text":"Antwort: 392"}),
                         ("fail","exact_mismatch"))

    def test_invalid_python_reports_syntax_without_docker(self):
        case={"grader":"test_execution","sealed_reference":{
            "function":"is_prime","tests":[{"args":[2],"expected":True}]}}
        with patch("scripts.neral_v41_blind200_live.subprocess.run",
                   side_effect=AssertionError("Should not launch Docker for invalid syntax")):
            self.assertEqual(
                grade_python(case, "def is_prime(n):\n  return ("),
                ("fail","python_syntax_invalid"))

    def test_code_still_requires_correct_function(self):
        case={"grader":"test_execution","sealed_reference":{
            "function":"is_prime","tests":[]}}
        self.assertEqual(grade_python(case,"def foo(n): return True"),
                         ("fail","no_required_function"))

if __name__=="__main__":
    unittest.main()
