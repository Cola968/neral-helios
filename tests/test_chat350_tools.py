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

    def test_mark_tool_outputs_separately(self):
        outcome = tool_answer("Calculate 8*9")
        self.assertEqual(outcome.kind, "calculator")
        self.assertEqual(outcome.text, "72")


if __name__ == "__main__":
    unittest.main()
