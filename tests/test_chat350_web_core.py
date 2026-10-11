from __future__ import annotations

import unittest

from frontier350.chat350_web_core import run_turn


class BrowserCoreTests(unittest.TestCase):
    def test_no_checkpoint_does_not_pretend_to_infer(self):
        answer, kind, history = run_turn("Erkläre mir Python.", [], native=None)
        self.assertEqual(kind, "no-model")
        self.assertIn("Kein Neral-350M-Modell", answer)
        self.assertEqual(history[-2][0], "user")

    def test_tool_is_attributed_separately(self):
        answer, kind, history = run_turn("Berechne 12 * 14", [], native=None)
        self.assertEqual(answer, "168")
        self.assertEqual(kind, "tool:calculator")
        self.assertEqual(history[-1], ("assistant", "168"))

    def test_native_receives_actual_user_history(self):
        def native(pending):
            self.assertEqual(pending[-1], ("user", "Hello!"))
            return "Hello there.", pending
        answer, kind, history = run_turn("Hello!", [], native=native)
        self.assertEqual((answer, kind), ("Hello there.", "native-model"))
        self.assertEqual(history[-1], ("assistant", "Hello there."))

    def test_no_tool_in_native_only_mode(self):
        def native(pending):
            return "Model output", pending
        answer, kind, _ = run_turn("Berechne 2 + 2", [],
                                   native=native, enable_tools=False)
        self.assertEqual((answer, kind), ("Model output", "native-model"))

    def test_compound_math_not_misrouted(self):
        answer, kind, _ = run_turn("Berechne 2 + 2 und ziehe 1 ab", [])
        self.assertEqual(kind, "no-model")
        self.assertIn("Kein", answer)

    def test_input_bounds_and_history(self):
        with self.assertRaises(ValueError):
            run_turn(" ", [])
        with self.assertRaises(ValueError):
            run_turn("x" * 2001, [])
        old = [("user", "a"), ("assistant", "b")] * 20
        _, _, updated = run_turn("Hi", old)
        self.assertLessEqual(len(updated), 26)


if __name__ == "__main__":
    unittest.main()
