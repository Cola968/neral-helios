"""Ensure a two-tool answer is attributed honestly by the GPT-3.5 development gate."""
import unittest

from neral_2022_gate import _verified_response, GateError


def output(tools):
    return {
        "mode": "neral-assistant", "source": "external-model",
        "base_model": "qwen3.5:9b", "model_generated": True,
        "quality_mode": "standard", "reasoning_enabled": False,
        "text": "Der Code hat einen Fehler, und 2+2 ist 4.",
        "tools_used": tools,
    }


class DualToolGateTests(unittest.TestCase):
    def test_gate_accepts_exactly_two_distinct_allowlisted_tools(self):
        used = _verified_response(output(["code_check", "calculator"]),
                                  model="qwen3.5:9b", quality="standard", enable_tools=True)
        self.assertEqual(used, ["code_check", "calculator"])

    def test_gate_rejects_repeat_and_third_call(self):
        for used in (["calculator", "calculator"],
                     ["calculator", "code_check", "calculator"],
                     ["calculator", "shell"]):
            with self.subTest(used=used), self.assertRaises(GateError):
                _verified_response(output(used), model="qwen3.5:9b",
                                   quality="standard", enable_tools=True)

    def test_gate_rejects_tools_during_model_native_only_comparison(self):
        with self.assertRaises(GateError):
            _verified_response(output(["code_check", "calculator"]),
                               model="qwen3.5:9b", quality="standard", enable_tools=False)


if __name__ == "__main__":
    unittest.main()
