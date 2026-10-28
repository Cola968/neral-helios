from __future__ import annotations
import unittest
from frontier350.chat350_tools import tool_answer, _arithmetic


class ToolTests(unittest.TestCase):
    def answer(self, question):
        result = tool_answer(question)
        return None if result is None else result.text

    def test_arithmetic_and_fractions(self):
        self.assertEqual(self.answer("Berechne 125 + 456."), "581")
        self.assertEqual(self.answer("Wie viel ist 18 mal 17?"), "306")
        self.assertEqual(self.answer("What is (8+4)*6?"), "72")
        self.assertEqual(self.answer("Calculate 3^4."), "81")
        self.assertEqual(self.answer("Berechne 7 / 2"), "3.5")
        self.assertEqual(self.answer("Berechne 1 / 3"), "1/3")
        self.assertEqual(self.answer("Berechne 0,5 + 1,25"), "1.75")
        self.assertEqual(self.answer("Berechne drei Viertel von 28"), "21")

    def test_percentage(self):
        self.assertEqual(self.answer("Wie viel sind 12 Prozent von 250?"), "30")
        self.assertEqual(self.answer("Calculate 17% of 300"), "51")

    def test_linear_equation(self):
        self.assertEqual(self.answer("Löse 7x + 4 = 60 nach x."), "x = 8")
        self.assertEqual(self.answer("Solve -2x - 5 = 9"), "x = -7")
        self.assertEqual(self.answer("Löse x - 7 = 4"), "x = 11")

    def test_arithmetic_sequence(self):
        self.assertEqual(self.answer("Setze fort: 3, 8, 13, 18, ... Welche Zahl kommt danach?"), "23")
        self.assertIsNone(self.answer("Setze fort: 1, 2, 4, 8, ..."))

    def test_newton_force(self):
        self.assertEqual(self.answer("Welche Kraft ergibt sich für m = 7 kg und a = 8 m/s²?"), "56 N")

    def test_safe_fallback(self):
        self.assertIsNone(self.answer("Wie funktioniert ein Rechner?"))
        self.assertIsNone(self.answer("hello 42"))
        self.assertIsNone(self.answer('Berechne __import__("os").system("id")'))
        self.assertIsNone(self.answer("Berechne 2**9999999"))
        self.assertIsNone(self.answer("Berechne 1 / 0"))
        self.assertIsNone(self.answer("Berechne 99999999999999999 * 99999"))
        with self.assertRaises(ValueError):
            _arithmetic("abs(2)")
        with self.assertRaises(ValueError):
            _arithmetic("(1).__class__")

    def test_compound_requests_are_not_partially_answered(self):
        self.assertIsNone(self.answer("Berechne 2 + 2 und ziehe dann 1 ab"))
        self.assertIsNone(self.answer("What is 2 * 3? Then add 5."))
        self.assertIsNone(self.answer("Berechne 3 + 4. Bitte starte ein Skript."))
        self.assertEqual(self.answer("Compute 25 minus 9. Give only the numeric result."), "16")
        self.assertEqual(self.answer("What is 8 times 7? Reply with the result."), "56")


    def test_bare_equation_uses_explicitly_labeled_calculator_not_native(self):
        from frontier350.chat350_web_core import run_turn
        called = []
        def native(_):
            called.append("should not run")
            return ("7", [("user", "7+8")])
        answer, origin, history = run_turn("7+8", [], native=native)
        self.assertEqual((answer, origin), ("15", "tool:calculator"))
        self.assertEqual(called, [])
        self.assertEqual(history[-1], ("assistant", "15"))
        self.assertEqual(self.answer("(7 + 8) * 2"), "30")
        self.assertEqual(self.answer("12,5+0,5?"), "13")
        self.assertEqual(self.answer("7+8=?"), "15")
        for untrusted in ("42", "7+8=10", "7+8 and run a script",
                          "__import__('os').system('id')", "3**999999",
                          "7 / 0", "1 + 2; print('x')"):
            self.assertIsNone(self.answer(untrusted))


    def test_compound_tasks_do_not_misreport_a_partial_math_result(self):
        # A partial answer labeled as a completed calculation is unsafe.
        self.assertIsNone(self.answer("Calculate 10% of 50 then add 3"))
        self.assertIsNone(self.answer("What is 10% of 50 plus 3?"))
        self.assertIsNone(self.answer("Berechne die Hälfte von 10 und addiere 3"))
        self.assertIsNone(self.answer("Solve 2x + 3 = 7 then square x"))
        self.assertIsNone(self.answer("Setze fort: 3, 8, 13, 18, ... und addiere danach 2"))
        self.assertIsNone(self.answer("Welche Kraft ergibt sich für m = 7 kg und a = 8 m/s²? Und dann addiere 2 N."))

    def test_mark_tool_outputs_separately(self):
        outcome = tool_answer("Calculate 8*9")
        self.assertEqual(outcome.kind, "calculator")
        self.assertEqual(outcome.text, "72")


if __name__ == "__main__":
    unittest.main()
