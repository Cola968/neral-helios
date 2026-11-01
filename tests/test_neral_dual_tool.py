"""Bounded sequential Qwen tools: two *different* tools and a final answer, never shell."""
import json
import unittest

from neral_assistant import assistant_turn, AssistantError


def result(text="Fertig.", *, calls=None, done_reason="stop"):
    msg = {"role": "assistant", "content": text}
    if calls is not None:
        msg["tool_calls"] = calls
    return {"model": "qwen3.5:9b", "done": True, "done_reason": done_reason, "message": msg}


def tool(name, params):
    return [{"function": {"name": name, "arguments": params}}]


class DualToolTests(unittest.TestCase):
    def test_code_review_then_calculation_in_one_turn(self):
        requests = []

        def fake(payload):
            requests.append(payload)
            if len(requests) == 1:
                return result("", calls=tool("code_check", {"code": "def f(x):\n  return eval(x)\n"}))
            if len(requests) == 2:
                self.assertEqual(payload["messages"][-1]["tool_name"], "code_check")
                self.assertIn("dynamic-code-execution", payload["messages"][-1]["content"])
                self.assertIn("executed", payload["messages"][-1]["content"])
                return result("", calls=tool("calculator", {"expression": "17 * 23"}))
            self.assertEqual(payload["messages"][-1]["tool_name"], "calculator")
            self.assertIn("391", payload["messages"][-1]["content"])
            return result("eval ist unsicher, und 17 mal 23 ist 391.")

        output = assistant_turn("Prüfe Code und rechne.", transport=fake)
        self.assertEqual(len(requests), 3)
        self.assertEqual(output["tools_used"], ["code_check", "calculator"])
        self.assertEqual(output["source"], "external-model")
        self.assertEqual(output["base_model"], "qwen3.5:9b")

    def test_calculation_then_code_review_in_same_turn(self):
        requests = []

        def fake(payload):
            requests.append(payload)
            if len(requests) == 1:
                return result("", calls=tool("calculator", {"expression": "9+8"}))
            if len(requests) == 2:
                return result("", calls=tool("code_check", {"code": "def x(:\n  pass"}))
            self.assertIn('"syntax_ok": false', payload["messages"][-1]["content"])
            return result("17, und der Code enthält einen Syntaxfehler.")

        output = assistant_turn("Rechne und prüfe Code.", transport=fake)
        self.assertEqual(output["tools_used"], ["calculator", "code_check"])
        self.assertEqual(len(requests), 3)

    def test_repeated_same_tool_is_denied_before_second_execution(self):
        requests = []

        def fake(payload):
            requests.append(payload)
            return result("", calls=tool("calculator", {"expression": "2+2"}))

        with self.assertRaisesRegex(AssistantError, "Werkzeugaufrufe"):
            assistant_turn("Rechne zwei Mal.", transport=fake)
        self.assertEqual(len(requests), 2)

    def test_third_tool_is_denied_after_two_distinct_calls(self):
        requests = []

        def fake(payload):
            requests.append(payload)
            if len(requests) == 1:
                return result("", calls=tool("calculator", {"expression": "2+2"}))
            if len(requests) == 2:
                return result("", calls=tool("code_check", {"code": "print(1)"}))
            return result("", calls=tool("calculator", {"expression": "9+9"}))

        with self.assertRaisesRegex(AssistantError, "Werkzeugaufrufe"):
            assistant_turn("Prüfe mehrere Aufgaben.", transport=fake)
        self.assertEqual(len(requests), 3)

    def test_multiple_calls_in_one_model_message_stay_denied(self):
        def fake(payload):
            return result("", calls=tool("calculator", {"expression": "2+2"}) +
                          tool("code_check", {"code": "print(1)"}))

        with self.assertRaises(AssistantError):
            assistant_turn("Beide auf einmal?", transport=fake)

    def test_truncated_second_tool_call_never_executes(self):
        requests = []

        def fake(payload):
            requests.append(payload)
            if len(requests) == 1:
                return result("", calls=tool("calculator", {"expression": "4+4"}))
            return result("", done_reason="length",
                          calls=tool("code_check", {"code": "print(1)"}))

        with self.assertRaises(AssistantError):
            assistant_turn("Prüfe beides.", transport=fake)
        self.assertEqual(len(requests), 2)


    def test_static_code_tool_result_is_bounded_without_hiding_omissions(self):
        # Valid Python code with hundreds of reported unsafe calls.
        source = "eval('x')\\n".replace("\\\\n", "\\n") * 360
        self.assertLessEqual(len(source.encode("utf-8")), 4096)
        requests = []

        def fake(payload):
            requests.append(payload)
            if len(requests) == 1:
                return result("", calls=tool("code_check", {"code": source}))
            raw = payload["messages"][-1]["content"]
            self.assertLessEqual(len(raw.encode("utf-8")), 4096)
            audit = json.loads(raw)
            self.assertEqual(audit["finding_count"], 360)
            self.assertEqual(audit["severity_counts"]["high"], 360)
            self.assertFalse(audit["executed"])
            self.assertTrue(audit["truncated"])
            self.assertGreater(audit["omitted_findings"], 0)
            self.assertEqual(audit["omitted_findings"] + len(audit["findings"]), 360)
            return result("Die statische Prüfung meldet viele riskante eval-Aufrufe.")

        assistant_turn("Analysiere den Code.", transport=fake)
        self.assertEqual(len(requests), 2)

    def test_static_code_tool_prioritizes_high_risk_findings_when_shortened(self):
        # Hundreds of low-severity 'bare except' findings precede the high-risk eval.
        source = ("try:\\n  pass\\nexcept:\\n  pass\\n" * 115) + "eval('x')\\n"
        self.assertLessEqual(len(source.encode("utf-8")), 4096)

        def fake(payload):
            if payload["messages"][-1]["role"] == "tool":
                raw = payload["messages"][-1]["content"]
                self.assertLessEqual(len(raw.encode("utf-8")), 4096)
                audit = json.loads(raw)
                self.assertEqual(audit["severity_counts"]["high"], 1)
                self.assertGreater(audit["omitted_findings"], 0)
                self.assertIn("dynamic-code-execution",
                              [x["code"] for x in audit["findings"]])
                return result("eval birgt ein hohes Risiko.")
            return result("", calls=tool("code_check", {"code": source}))

        output = assistant_turn("Prüfe die Fehler.", transport=fake)
        self.assertEqual(output["tools_used"], ["code_check"])


if __name__ == "__main__":
    unittest.main()
