"""TDD contracts: Qwen gets existing bounded Python AST checker, never code execution."""
import json
import unittest

from neral_assistant import assistant_turn, AssistantError


def _reply(text="Die Funktion hat einen Fehler."):
    return {"model": "qwen3.5:9b", "done": True,
            "message": {"role": "assistant", "content": text}}


class QwenStaticCodeToolTests(unittest.TestCase):
    def test_model_can_request_existing_python_checker_and_explain_its_findings(self):
        seen = []
        def fake(payload):
            seen.append(payload)
            if len(seen) == 1:
                self.assertEqual([t["function"]["name"] for t in payload["tools"]],
                                 ["calculator", "code_check"])
                return {"model": "qwen3.5:9b", "done": True,
                        "message": {"role": "assistant", "content": "",
                                    "tool_calls": [{"function": {
                                        "name": "code_check",
                                        "arguments": {"code": "def run(user_input):\n    return eval(user_input)\n"}
                                    }}]}}
            self.assertEqual(payload["messages"][-1]["role"], "tool")
            self.assertEqual(payload["messages"][-1]["tool_name"], "code_check")
            findings = json.loads(payload["messages"][-1]["content"])
            self.assertTrue(findings["syntax_ok"])
            self.assertFalse(findings["executed"])
            self.assertIn("dynamic-code-execution",
                          [x["code"] for x in findings["findings"]])
            return _reply("eval() kann gefährlich sein. Verwende eine sichere Alternative.")
        result = assistant_turn("Prüfe diesen Python-Code", transport=fake)
        self.assertEqual(result["tools_used"], ["code_check"])
        self.assertEqual(result["source"], "external-model")
        self.assertEqual(result["base_model"], "qwen3.5:9b")
        self.assertEqual(len(seen), 2)
        self.assertIn("eval()", result["text"])

    def test_syntax_error_is_reported_not_executed(self):
        def fake(payload):
            if payload["messages"][-1]["role"] != "tool":
                return {"model": "qwen3.5:9b", "done": True,
                        "message": {"role": "assistant", "tool_calls": [
                            {"function": {"name": "code_check",
                                          "arguments": {"code": "def broken(:\n    pass"}}}]}}
            findings = json.loads(payload["messages"][-1]["content"])
            self.assertFalse(findings["syntax_ok"])
            self.assertFalse(findings["executed"])
            return _reply("Der Python-Code enthält einen Syntaxfehler.")
        result = assistant_turn("Findest du den Syntaxfehler?", transport=fake)
        self.assertEqual(result["tools_used"], ["code_check"])

    def test_untrusted_code_is_never_run(self):
        # An arbitrary Python snippet may be submitted, but this tool must only AST-parse.
        submitted = "import os\nos.system('echo NEVER_RUN')\n"
        def fake(payload):
            if payload["messages"][-1]["role"] == "tool":
                result = json.loads(payload["messages"][-1]["content"])
                self.assertFalse(result["executed"])
                self.assertIn("shell-execution", [f["code"] for f in result["findings"]])
                return _reply("Das ist ein riskanter Shell-Aufruf.")
            return {"model": "qwen3.5:9b", "done": True,
                    "message": {"role": "assistant", "tool_calls": [
                        {"function": {"name": "code_check", "arguments": {"code": submitted}}}]}}
        result = assistant_turn("Prüfe Sicherheit.", transport=fake)
        self.assertEqual(result["tools_used"], ["code_check"])

    def test_tool_is_forbidden_in_native_only_benchmark(self):
        def fake(payload):
            self.assertEqual(payload["tools"], [])
            return {"model": "qwen3.5:9b", "done": True,
                    "message": {"role": "assistant", "tool_calls": [
                        {"function": {"name": "code_check",
                                      "arguments": {"code": "print(1)"}}}]}}
        with self.assertRaises(AssistantError):
            assistant_turn("Review", transport=fake, enable_tools=False)

    def test_invalid_code_check_calls_fail_closed(self):
        invalid = [
            {"code": "x" * 4097},
            {"code": "print(1)", "shell": True},
            {"code": 123},
            {"code": ""},
            {"expression": "2+2"},
        ]
        for args in invalid:
            with self.subTest(args=str(args)[:48]):
                def fake(payload):
                    return {"model": "qwen3.5:9b", "done": True,
                            "message": {"role": "assistant", "tool_calls": [
                                {"function": {"name": "code_check", "arguments": args}}]}}
                with self.assertRaises(AssistantError):
                    assistant_turn("Review", transport=fake)

    def test_malformed_falsy_tool_calls_are_rejected_not_treated_as_no_tools(self):
        # Ollama protocol: absent/null means no calls; present must be an array.
        for malformed in ("", 0, False, {}):
            with self.subTest(value=repr(malformed)):
                def fake(payload):
                    return {"model": "qwen3.5:9b", "done": True,
                            "message": {"role": "assistant", "content": "OK",
                                        "tool_calls": malformed}}
                with self.assertRaises(AssistantError):
                    assistant_turn("Prüfe Protokoll", transport=fake)

    def test_second_tool_call_cannot_escape_one_tool_budget(self):
        attempts = []
        def fake(payload):
            attempts.append(payload)
            return {"model": "qwen3.5:9b", "done": True,
                    "message": {"role": "assistant", "tool_calls": [
                        {"function": {"name": "code_check", "arguments": {"code": "print(1)"}}}]}}
        with self.assertRaises(AssistantError):
            assistant_turn("Review", transport=fake)
        self.assertEqual(len(attempts), 2)


    def test_ollama_metadata_on_code_check_envelope_is_accepted(self):
        # The real Qwen loopback reported code_check with dict arguments,
        # but our strict envelope rejected additional protocol metadata.
        calls = []
        def fake(payload):
            calls.append(payload)
            if len(calls) == 1:
                return {"model": "qwen3.5:9b", "done": True, "done_reason": "stop",
                        "message": {"role": "assistant", "content": "",
                                    "tool_calls": [{"id": "call_0-ABC", "type": "function",
                                                    "index": 0, "function": {
                                                        "name": "code_check",
                                                        "arguments": {"code": "def unsafe(x): return eval(x)"}}}]}}
            self.assertEqual(payload["messages"][-1]["tool_name"], "code_check")
            findings = json.loads(payload["messages"][-1]["content"])
            self.assertFalse(findings["executed"])
            self.assertIn("dynamic-code-execution", [f["code"] for f in findings["findings"]])
            return _reply("eval ist unsicher.")

        result = assistant_turn("Prüfe Python-Code", transport=fake)
        self.assertEqual(result["tools_used"], ["code_check"])
        self.assertEqual(len(calls), 2)

    def test_ollama_metadata_also_supported_for_calculator(self):
        calls = []
        def fake(payload):
            calls.append(payload)
            if len(calls) == 1:
                return {"model": "qwen3.5:9b", "done": True, "done_reason": "stop",
                        "message": {"role": "assistant", "content": "",
                                    "tool_calls": [{"id": "call_calculator_1", "type": "function",
                                                    "index": 0, "function": {
                                                        "name": "calculator",
                                                        "arguments": {"expression": "17*23"}}}]}}
            self.assertEqual(payload["messages"][-1]["tool_name"], "calculator")
            self.assertIn("391", payload["messages"][-1]["content"])
            return _reply("391")

        result = assistant_turn("Berechne 17 mal 23", transport=fake)
        self.assertEqual(result["tools_used"], ["calculator"])

    def test_tool_metadata_is_strictly_bounded(self):
        malicious_metadata = [
            {"unexpected": "shell"},
            {"id": "x" * 129},
            {"id": ""},
            {"id": "../outside"},
            {"type": "shell"},
            {"type": True},
            {"index": -1},
            {"index": True},
            {"index": 2},
        ]
        for extra in malicious_metadata:
            with self.subTest(extra=str(extra)[:60]):
                def fake(payload):
                    return {"model": "qwen3.5:9b", "done": True, "done_reason": "stop",
                            "message": {"role": "assistant", "tool_calls": [{
                                **extra, "function": {"name": "code_check",
                                                     "arguments": {"code": "print(1)"}}}]}}
                with self.assertRaises(AssistantError):
                    assistant_turn("Prüfe Python-Code", transport=fake)


if __name__ == "__main__":
    unittest.main()
