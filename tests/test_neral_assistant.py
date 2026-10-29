"""NERAL Assistant contract tests; all inference calls are fake/offline."""
import unittest

from neral_assistant import assistant_turn, AssistantError, AssistantUnavailable


def answer(text, *, model="qwen3.5:9b"):
    return {"model": model, "message": {"role": "assistant", "content": text},
            "done": True, "eval_count": 12, "eval_duration": 1000000000}


class PersonalityIntegrationTests(unittest.TestCase):
    def test_actual_ollama_payload_uses_versioned_neral_personality(self):
        from neral_assistant_persona import PERSONA_VERSION, SYSTEM_PROMPT
        from neral_assistant import SYSTEM_PROMPT as ACTIVE_PROMPT
        self.assertIs(ACTIVE_PROMPT, SYSTEM_PROMPT)
        self.assertEqual(PERSONA_VERSION, "NERAL-P1")
        self.assertIn("Qwen3.5", SYSTEM_PROMPT)
        self.assertIn("Deutsch", SYSTEM_PROMPT)
        self.assertIn("keine erfundenen Quellen", SYSTEM_PROMPT)
        captured=[]
        def fake(body):
            captured.append(body)
            return answer("Hallo! Ich bin NERAL.")
        result=assistant_turn("Wer bist du?",transport=fake)
        self.assertEqual(captured[0]["messages"][0]["content"],SYSTEM_PROMPT)
        self.assertEqual(result["persona_version"],PERSONA_VERSION)
        self.assertEqual(result["base_model"],"qwen3.5:9b")

    def test_local_tool_access_is_not_misrepresented(self):
        from neral_assistant_persona import SYSTEM_PROMPT
        self.assertIn("calculator",SYSTEM_PROMPT)
        self.assertIn("keine dauerhafte",SYSTEM_PROMPT)


class AssistantTests(unittest.TestCase):
    def test_local_model_and_branding_are_explicit(self):
        sent = []
        def fake(payload):
            sent.append(payload)
            return answer("Hallo! Ich bin NERAL.")
        result = assistant_turn("Hallo", transport=fake)
        self.assertEqual(result["mode"], "neral-assistant")
        self.assertEqual(result["base_model"], "qwen3.5:9b")
        self.assertEqual(result["text"], "Hallo! Ich bin NERAL.")
        self.assertTrue(result["model_generated"])
        self.assertEqual(sent[0]["model"], "qwen3.5:9b")
        self.assertEqual(sent[0]["stream"], False)
        self.assertEqual(sent[0]["options"]["num_ctx"], 4096)
        self.assertEqual(sent[0]["messages"][-1], {"role": "user", "content": "Hallo"})
        self.assertEqual(result["tools_used"], [])

    def test_history_and_temperature_are_bounded(self):
        def fake(payload):
            self.assertEqual([x["role"] for x in payload["messages"][-3:]],
                             ["user", "assistant", "user"])
            self.assertEqual(payload["options"]["temperature"], 0.5)
            return answer("Gut.")
        result = assistant_turn("Wie geht's?", history=[
            {"role":"user","content":"Hi"}, {"role":"assistant","content":"Hallo"}
        ], transport=fake, temperature=0.5)
        self.assertEqual(result["text"], "Gut.")

    def test_calculator_tool_is_executed_without_eval(self):
        calls = []
        def fake(payload):
            calls.append(payload)
            if len(calls) == 1:
                return {"model":"qwen3.5:9b","message":{"role":"assistant","content":"",
                    "tool_calls":[{"function":{"name":"calculator","arguments":{"expression":"12*14"}}}]},
                    "done":True}
            self.assertEqual(payload["messages"][-1]["role"], "tool")
            self.assertIn("168", payload["messages"][-1]["content"])
            return answer("12 × 14 = 168.")
        result=assistant_turn("Wie viel sind 12 mal 14?",transport=fake)
        self.assertEqual(result["tools_used"],["calculator"])
        self.assertIn("168", result["text"])
        self.assertEqual(len(calls), 2)

    def test_unknown_or_unsafe_tool_fails_closed(self):
        for name,args in (
            ("shell",{"command":"whoami"}),
            ("calculator",{"expression":"__import__('os').system('id')"}),
            ("calculator",{"expression":"2**9999999999999"}),
            ("calculator",{"expression":"1/0"}),
        ):
            def fake(payload):
                return {"model":"qwen3.5:9b","done":True,
                        "message":{"role":"assistant","tool_calls":[{"function":{
                            "name":name,"arguments":args}}]}}
            with self.subTest(name=name,args=args),self.assertRaises(AssistantError):
                assistant_turn("Test",transport=fake)

    def test_repeated_tool_call_is_rejected(self):
        def fake(payload):
            return {"model":"qwen3.5:9b","done":True,
                    "message":{"role":"assistant","tool_calls":[{"function":{
                        "name":"calculator","arguments":{"expression":"2+2"}}}]}}
        with self.assertRaises(AssistantError):
            assistant_turn("Berechne 2+2",transport=fake)

    def test_malformed_inputs_fail_before_network(self):
        def forbidden(_):
            raise AssertionError("transport must not be called")
        invalid = [
            {"prompt":" "}, {"prompt":"x"*4097},
            {"prompt":"Hi","history":[{"role":"assistant","content":"Hi"}]},
            {"prompt":"Hi","history":[{"role":"user","content":"Hi"}]},
            {"prompt":"Hi","history":[{"role":"user","content":"x"},
                                      {"role":"user","content":"y"}]},
            {"prompt":"Hi","model":"qwen3.5:27b"},
            {"prompt":"Hi","max_new":9999},
            {"prompt":"Hi","temperature":float("nan")},
        ]
        for params in invalid:
            with self.subTest(params=str(params)[:80]),self.assertRaises(AssistantError):
                assistant_turn(transport=forbidden,**params)

    def test_no_silent_model_substitution_or_empty_success(self):
        for response in (
            answer("Nice",model="some-other-model"),
            answer(""),
            {"model":"qwen3.5:9b","done":False,"message":{"role":"assistant","content":"partial"}},
            {"model":"qwen3.5:9b","done":True,"message":{"role":"tool","content":"fake"}},
        ):
            with self.subTest(response=response),self.assertRaises(AssistantError):
                assistant_turn("Hallo",transport=lambda _:response)

    def test_offline_ollama_failure_is_not_misrepresented(self):
        def offline(_):
            raise AssistantUnavailable("Ollama ist nicht erreichbar")
        with self.assertRaises(AssistantUnavailable):
            assistant_turn("Hallo",transport=offline)

    def test_model_will_not_receive_hidden_cloud_url(self):
        def fake(payload):
            self.assertEqual(set(payload["options"]), {"num_ctx","num_predict","temperature"})
            self.assertEqual([tool["function"]["name"] for tool in payload["tools"]],
                             ["calculator"])
            return answer("Antwort")
        assistant_turn("Hallo",transport=fake)

    def test_native_benchmark_mode_exposes_no_tool_calls(self):
        def native_only(payload):
            self.assertEqual(payload["tools"], [])
            return answer("Die Antwort lautet 4.")
        result = assistant_turn("Berechne 2+2", transport=native_only, enable_tools=False)
        self.assertEqual(result["tools_used"], [])
        with self.assertRaises(AssistantError):
            assistant_turn("Hi",transport=native_only,enable_tools="no")

    def test_native_only_rejects_unexpected_tool_request(self):
        def malicious(_):
            return {"model":"qwen3.5:9b","done":True,"message":{"role":"assistant",
                "tool_calls":[{"function":{"name":"calculator","arguments":{"expression":"2+2"}}}]}}
        with self.assertRaises(AssistantError):
            assistant_turn("Berechne 2+2", transport=malicious, enable_tools=False)


if __name__ == "__main__":
    unittest.main()

