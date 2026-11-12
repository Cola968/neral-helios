"""No false completed Ollama answers; public-safe 503 failure categories.

These tests are CPU-only; no network, GPU inference, prompt logging or weights.
"""
import json
import unittest
from unittest.mock import patch

from neral_agent_autonomy import (
    OllamaLocalBackend, OllamaCompletionError,
)
from neral_preview_server import create_preview_app


class Response:
    def __init__(self, reply=None, lines=None):
        self.reply=reply
        self.lines=lines
    def __enter__(self): return self
    def __exit__(self,*_): return False
    def read(self,_):
        return json.dumps(self.reply).encode("utf8")
    def __iter__(self):
        return iter(json.dumps(x).encode("utf8")+b"\n" for x in self.lines)


class FakeOllama:
    def __init__(self, reply=None, lines=None, fail=None):
        self.reply=reply
        self.lines=lines
        self.fail=fail
        self.calls=[]
    def open(self,request,timeout=0):
        self.calls.append((request.full_url,json.loads(request.data),timeout))
        if self.fail is not None:
            raise self.fail
        return Response(self.reply,self.lines)


MESSAGES=[{"role":"user","content":"Berechne die Lösung ohne frühzeitig abzubrechen."}]
MODEL="qwen3.5:9b"


class CompletionDiagnostics(unittest.TestCase):
    def model(self):
        return OllamaLocalBackend(MODEL,inference_profile="measured_nonthinking_2048")

    def test_length_stop_is_classified_without_leaking_model_text(self):
        f=FakeOllama(reply={"done":True,"done_reason":"length",
                            "message":{"content":"PRIVATE partial answer"}})
        with patch("neral_agent_autonomy.local_only_opener",return_value=f):
            with self.assertRaises(OllamaCompletionError) as ctx:
                self.model().respond_chat(MESSAGES)
        self.assertEqual(ctx.exception.reason,"token_limit")
        self.assertNotIn("PRIVATE",str(ctx.exception))
        self.assertEqual(len(f.calls),1,"Do not automatically retry an unverified result")

    def test_missing_terminal_metadata_is_not_an_accepted_measured_answer(self):
        cases=[
          {"message":{"content":"plausible fake answer"}},
          {"done":True,"message":{"content":"plausible fake answer"}},
          {"done":False,"done_reason":"stop","message":{"content":"partial"}},
          {"done":True,"done_reason":"load","message":{"content":"partial"}},
        ]
        for packet in cases:
            with self.subTest(packet=packet):
                with patch("neral_agent_autonomy.local_only_opener",
                           return_value=FakeOllama(reply=packet)):
                    with self.assertRaises(OllamaCompletionError) as ctx:
                        self.model().respond_chat(MESSAGES)
                self.assertEqual(ctx.exception.reason,"incomplete")

    def test_empty_final_is_classified_without_exposing_hidden_thinking(self):
        f=FakeOllama(reply={"done":True,"done_reason":"stop",
             "message":{"content":"","thinking":"PRIVATE internal text"}})
        with patch("neral_agent_autonomy.local_only_opener",return_value=f):
            with self.assertRaises(OllamaCompletionError) as ctx:
                self.model().respond_think(MESSAGES)
        self.assertEqual(ctx.exception.reason,"empty_answer")
        self.assertNotIn("PRIVATE",str(ctx.exception))

    def test_timeout_has_stable_nonsecret_code(self):
        f=FakeOllama(fail=TimeoutError("C:/SECRET/internal_generative_thoughts.txt"))
        with patch("neral_agent_autonomy.local_only_opener",return_value=f):
            with self.assertRaises(OllamaCompletionError) as ctx:
                self.model().respond_chat(MESSAGES)
        self.assertEqual(ctx.exception.reason,"timeout")
        self.assertNotIn("SECRET",str(ctx.exception))

    def test_stream_with_missing_finish_reason_never_sends_verified_final(self):
        f=FakeOllama(lines=[
          {"done":False,"message":{"content":"Partial"}},
          {"done":True,"message":{"content":""}}
        ])
        with patch("neral_agent_autonomy.local_only_opener",return_value=f):
            stream=self.model().respond_chat_stream(MESSAGES)
            self.assertEqual(next(stream),"Partial")
            with self.assertRaises(OllamaCompletionError) as ctx:
                list(stream)
        self.assertEqual(ctx.exception.reason,"incomplete")

    def test_local_http_only_returns_safe_failure_code_and_never_partial_answer(self):
        scenarios=[
           ({"done":True,"done_reason":"length",
             "message":{"content":"PRIVATE 42"}},"token_limit"),
           ({"done":True,"done_reason":"stop",
             "message":{"content":""}},"empty_answer"),
        ]
        for packet,reason in scenarios:
            with self.subTest(reason=reason):
                with patch("neral_agent_autonomy.local_only_opener",
                           return_value=FakeOllama(reply=packet)):
                    backend=self.model()
                    client=create_preview_app(
                        backend,secret_key=b"Z"*32).test_client()
                    base="http://127.0.0.1:9690"
                    client.get("/",base_url=base)
                    session=client.get("/api/neral/preview/session",base_url=base)
                    result=client.post("/api/neral/chat/generate",
                        json={"prompt":"Erkläre den Inhalt der Aufgabe."},
                        headers={"X-Neral-CSRF":session.json["csrf"]},
                        base_url=base)
                self.assertEqual(result.status_code,503)
                self.assertEqual(result.json["error_code"],reason)
                self.assertNotIn("PRIVATE",str(result.json))
                self.assertNotIn("text",result.json)


    def test_high_http_failure_reason_is_safe_and_not_success(self):
        f=FakeOllama(reply={"done":True,"done_reason":"length",
                            "message":{"content":"PRIVATE final-looking answer"}})
        with patch("neral_agent_autonomy.local_only_opener",return_value=f):
            client=create_preview_app(self.model(),secret_key=b"H"*32).test_client()
            base="http://127.0.0.1:9690"
            client.get("/",base_url=base)
            token=client.get("/api/neral/preview/session",base_url=base).json["csrf"]
            result=client.post("/api/neral/chat/think",
                json={"prompt":"Erkläre die Bedeutung der Keplerschen Gesetze."},
                headers={"X-Neral-CSRF":token},base_url=base)
        self.assertEqual(result.status_code,503)
        self.assertEqual(result.json["error_code"],"token_limit")
        self.assertNotIn("PRIVATE",str(result.json))
        self.assertNotIn("text",result.json)

    def test_streaming_http_error_never_emits_a_final_success_event(self):
        f=FakeOllama(lines=[
            {"done":False,"message":{"content":"PRIVATE partial"}},
            {"done":True,"done_reason":"length","message":{"content":""}}
        ])
        with patch("neral_agent_autonomy.local_only_opener",return_value=f):
            client=create_preview_app(self.model(),secret_key=b"S"*32).test_client()
            base="http://127.0.0.1:9690"
            client.get("/",base_url=base)
            token=client.get("/api/neral/preview/session",base_url=base).json["csrf"]
            result=client.post("/api/neral/chat/generate-stream",
                json={"prompt":"Erkläre ein komplexes Problem."},
                headers={"X-Neral-CSRF":token},base_url=base)
            events=[json.loads(line) for line in result.data.splitlines()]
        self.assertEqual(result.status_code,200)
        self.assertEqual(events[-1]["type"],"error")
        self.assertEqual(events[-1]["error_code"],"token_limit")
        self.assertFalse(any(x["type"]=="final" for x in events))
        self.assertNotIn("PRIVATE",str(events[-1]))


    def test_token_limit_message_blames_output_budget_not_input_length(self):
        from neral_local_chat_api import _SAFE_OLLAMA_ERROR_TEXT
        explanation=_SAFE_OLLAMA_ERROR_TEXT["token_limit"]
        self.assertIn("Ausgabelimit",explanation)
        self.assertNotIn("kürzer fragen",explanation.lower())
        self.assertIn("Antwort",explanation)

if __name__=="__main__":
    unittest.main()
