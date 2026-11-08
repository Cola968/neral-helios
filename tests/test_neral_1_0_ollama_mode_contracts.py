"""Release-targeted RED contracts: Ollama FAST vs THINK, no truncated answers."""
from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from flask import Flask

from neral_agent_autonomy import AgentModelError, OllamaLocalBackend
from neral_local_chat_api import register_neral_local_chat_api


class _Response:
    def __init__(self, packet):
        self.raw = json.dumps(packet).encode("utf8")
    def __enter__(self): return self
    def __exit__(self, *_): return False
    def read(self, size): return self.raw


class _MockLocalOnly:
    def __init__(self, packet=None):
        self.packet = packet or {"done": True,"done_reason":"stop",
                                 "message":{"role":"assistant","thinking":"HIDDEN_PRIVATE_COT",
                                            "content":"Die Antwort ist 42."}}
        self.calls=[]
    def open(self,request,timeout):
        self.calls.append((request.full_url, json.loads(request.data), timeout))
        return _Response(self.packet)


class NeralReleaseInferenceTests(unittest.TestCase):
    @staticmethod
    def messages():
        return [{"role":"system","content":"Nur lokal."},
                {"role":"user","content":"Was ist die Antwort?"}]

    def test_fast_chat_and_agent_json_explicitly_disable_hidden_thinking(self):
        local=_MockLocalOnly()
        with patch("neral_agent_autonomy.local_only_opener",return_value=local):
            backend=OllamaLocalBackend("qwen3.5:9b")
            self.assertEqual(backend.respond_chat(self.messages()),"Die Antwort ist 42.")
            self.assertEqual(backend.respond(self.messages()),"Die Antwort ist 42.")
        self.assertEqual(len(local.calls),2)
        for url,payload,timeout in local.calls:
            self.assertEqual(url,"http://127.0.0.1:11434/api/chat")
            self.assertEqual(payload["think"],False)
            self.assertEqual(payload["stream"],False)
            self.assertEqual(payload["model"],"qwen3.5:9b")
            self.assertLessEqual(timeout,90)
        self.assertEqual(local.calls[0][1]["options"]["num_predict"],1024)
        self.assertEqual(local.calls[1][1]["options"]["num_predict"],512)
        self.assertNotIn("format",local.calls[0][1])
        self.assertEqual(local.calls[1][1]["format"],"json")

    def test_explicit_think_uses_separate_reasoning_budget_and_never_exposes_it(self):
        local=_MockLocalOnly()
        with patch("neral_agent_autonomy.local_only_opener",return_value=local):
            backend=OllamaLocalBackend("qwen3:8b")
            output=backend.respond_think(self.messages())
        self.assertEqual(output,"Die Antwort ist 42.")
        self.assertNotIn("HIDDEN_PRIVATE_COT",output)
        payload=local.calls[0][1]
        self.assertEqual(payload["think"],True)
        self.assertEqual(payload["options"]["num_predict"],2048)
        self.assertNotIn("format",payload)

    def test_length_censored_outputs_and_nonterminal_packets_fail_closed(self):
        for packet in (
            {"done":True,"done_reason":"length","message":{"content":"FINAL: 42"}},
            {"done":False,"message":{"content":"partial"}},
            {"done":True,"done_reason":"error","message":{"content":"not true"}},
            {"done":True,"done_reason":"stop","message":{"content":"",
                                                      "thinking":"not done"}},
        ):
            with self.subTest(packet=packet):
                local=_MockLocalOnly(packet)
                with patch("neral_agent_autonomy.local_only_opener",return_value=local):
                    with self.assertRaises(AgentModelError):
                        OllamaLocalBackend("qwen3:8b").respond_chat(self.messages())

    def test_http_think_chooses_explicit_backend_reasoning_while_chat_stays_fast(self):
        class DualBackend:
            model="qwen3.5:9b"
            def __init__(self): self.calls=[]
            def respond(self,messages):
                self.calls.append("json")
                return "{}"
            def respond_chat(self,messages):
                self.calls.append("fast")
                return "FAST_NORMAL"
            def respond_think(self,messages):
                self.calls.append("think")
                return "THINK_OPTIN"
        local=DualBackend()
        app=Flask("neral1-0-inference")
        register_neral_local_chat_api(app,local)
        client=app.test_client()
        normal=client.post("/api/neral/chat/generate",json={"prompt":"Erkläre Sternentwicklung."})
        deep=client.post("/api/neral/chat/think",json={"prompt":"Erkläre Sternentwicklung."})
        self.assertEqual(normal.status_code,200)
        self.assertEqual(normal.json["text"],"FAST_NORMAL")
        self.assertEqual(deep.status_code,200)
        self.assertEqual(deep.json["text"],"THINK_OPTIN")
        self.assertEqual(local.calls,["fast","think"])
        self.assertFalse(deep.json["verified"])
        self.assertEqual(deep.json["model_origin"],"local_foundation_not_neral_adapter")

    def test_hidden_model_thinking_is_not_accepted_as_empty_final_answer(self):
        local=_MockLocalOnly({"done":True,"done_reason":"stop",
                              "message":{"content":"", "thinking":"secret work"}})
        with patch("neral_agent_autonomy.local_only_opener",return_value=local):
            with self.assertRaises(AgentModelError) as error:
                OllamaLocalBackend("qwen3.5:9b").respond_think(self.messages())
        self.assertNotIn("secret work",str(error.exception))


if __name__=="__main__":
    unittest.main()
