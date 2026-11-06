import tempfile
import unittest
from pathlib import Path
from solve import solve, arithmetic
from neral_agent_autonomy import AutonomousAgent


class ImplicitMultiplicationTests(unittest.TestCase):
    def test_eight_verified_development_equations(self):
        from scripts.neral_math_v2_eval import cases as original_cases
        import re
        frozen = [r for r in original_cases() if r["family"] == "equation"]
        self.assertEqual(len(frozen), 8)
        for row in frozen:
            match = re.search(r"-?\\d+x \\+ \\(-?\\d+\\) = -?\\d+", row["prompt"])
            self.assertIsNotNone(match, msg=row["id"])
            with self.subTest(id=row["id"]):
                self.assertEqual(solve(match.group(0))["result"], "x = " + row["expected"])

    def test_signed_coefficient_and_brackets(self):
        cases = [
            ("-x = 7", "x = -7"),
            ("3(x - 2) = 15", "x = 7"),
            ("-3(x - 2) = 15", "x = -3"),
            ("2(x + 3) = x + 9", "x = 3"),
            ("0.5x + 1 = 2", "x = 2"),
            ("3 x + 2 = 11", "x = 3"),
            ("3(2x - 1) = 9", "x = 2"),
            ("x/3 = 1/2", "x = 3/2"),
        ]
        for expression, expected in cases:
            with self.subTest(expression=expression):
                self.assertEqual(solve(expression)["result"], expected)

    def test_arithmetic_without_eval(self):
        self.assertEqual(arithmetic("3(4+2)")["result"], "18")
        self.assertEqual(arithmetic("2(3+4)+5")["result"], "19")
        self.assertEqual(arithmetic("2**3 + 4")["result"], "12")
        self.assertEqual(solve("2*x + 3 = 11")["result"], "x = 4")

    def test_rejects_scripting_and_nonlinear_expressions(self):
        bad = [
            "2x=__import__('os').system('echo hello')", "2xy=4",
            "3sin(x)=0", "x(3)=3", "x*x=16", "2x=3=4",
            "3(2x+1", "3)x=1", "3(2x)**2=12", "2x=1;exit()",
            "2x=1\nprint(1)", "9"*181 + "x=1",
        ]
        for expr in bad:
            with self.subTest(expression=expr[:24]):
                with self.assertRaises(ValueError):
                    solve(expr)

    def test_agent_has_explicit_readonly_solve_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent = AutonomousAgent(Path(tmp), backend=None)
            self.assertIn("solve", agent.TOOL_NAMES)
            row = agent._execute("solve", {"expression": "6x - 32 = 286"}, {})
            self.assertEqual(row["result"], "x = 53")
            self.assertFalse(row["model_generated"])
            with self.assertRaises(Exception):
                agent._execute("solve", {"expression": "__import__('os')"}, {})


if __name__ == "__main__":
    unittest.main()
