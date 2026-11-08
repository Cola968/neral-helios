"""REAL Ollama streaming, safe NDJSON endpoint and lock/CSRF regression."""
import json
import unittest
from flask import Flask
from unittest.mock import patch

from neral_agent_autonomy import AgentModelError, OllamaLocalBackend
from neral_local_chat_api import register_neral_local_chat_api
from neral_preview_server import create_preview_app


def lines(*packets):
    return [json.dumps(packet, ensure_ascii=False).encode("utf-8") + b"\n" for packet in packets]


class StreamResponse:
    def __init__(self, packets):
        self.packets=packets
        self.closed=False
    def __enter__(self): return self
    def __exit__(self, *_): self.closed=True
    def __iter__(self): return iter(self.packets)


class FakeOpener:
    def __init__(self, packets):
        self.packets=packets
        self.last=None
        self.response=None
    def open(self, request, timeout=0):
        self.last=json.loads(request.data)
        self.response=StreamResponse(self.packets)
        return self.response


class FakeStreamingBackend:
    model="qwen3:8b"
    def __init__(self, fail=False):
        self.fail=fail
        self.calls=[]
    def respond_chat_stream(self, messages):
        self.calls.append(messages)
        yield "Guten"
        yield " Morgen"
        if self.fail:
            raise AgentModelError("Cannot produce complete answer")
    def respond_chat(self, messages):
        raise AssertionError("Streaming endpoint cannot fall back to non-streaming for a streaming backend")
    def respond(self, messages):
        return "{}"


class StreamingBackendTests(unittest.TestCase):
    def messages(self):
        return [{"role":"system","content":"Nur lokal"},
                {"role":"user","content":"Sag hallo"}]

    def test_ollama_real_chunks_never_expose_hidden_thinking(self):
        fake=FakeOpener(lines(
            {"done":False,"message":{"content":"Hallo","thinking":"PRIVATE"}},
            {"done":False,"message":{"content":" Welt","thinking":"SECRET"}},
            {"done":True,"done_reason":"stop","message":{"content":""}},
        ))
        with patch("neral_agent_autonomy.local_only_opener",return_value=fake):
            output=list(OllamaLocalBackend("qwen3:8b").respond_chat_stream(self.messages()))
        self.assertEqual(output,["Hallo"," Welt"])
        self.assertTrue(fake.response.closed)
        self.assertTrue(fake.last["stream"])
        self.assertFalse(fake.last["think"])
        self.assertNotIn("format",fake.last)
        self.assertEqual(fake.last["options"]["num_predict"],1024)

    def test_length_limit_does_not_signal_a_successful_final(self):
        fake=FakeOpener(lines(
            {"done":False,"message":{"content":"partial"}},
            {"done":True,"done_reason":"length","message":{"content":""}},
        ))
        with patch("neral_agent_autonomy.local_only_opener",return_value=fake):
            with self.assertRaises(AgentModelError):
                list(OllamaLocalBackend("qwen3:8b").respond_chat_stream(self.messages()))

    def test_missing_done_or_content_is_failure(self):
        for packets in [
            lines({"done":False,"message":{"content":"PARTIAL"}}),
            lines({"done":True,"done_reason":"stop","message":{"content":""}}),
            lines({"done":False,"message":{"thinking":"PRIVATE"}},
                  {"done":True,"done_reason":"stop","message":{}}),
        ]:
            with self.subTest(packets=packets):
                fake=FakeOpener(packets)
                with patch("neral_agent_autonomy.local_only_opener",return_value=fake):
                    with self.assertRaises(AgentModelError):
                        list(OllamaLocalBackend("qwen3:8b").respond_chat_stream(self.messages()))


class StreamingHTTPTests(unittest.TestCase):
    @staticmethod
    def parse(response):
        return [json.loads(line) for line in response.data.splitlines() if line.strip()]

    def test_low_stream_delivers_real_deltas_then_verified_terminal_event(self):
        fake=FakeStreamingBackend()
        app=Flask("neral-streaming-unit")
        register_neral_local_chat_api(app,fake)
        client=app.test_client()
        res=client.post("/api/neral/chat/generate-stream",json={"prompt":"Sag hallo"})
        self.assertEqual(res.status_code,200)
        self.assertTrue(res.content_type.startswith("application/x-ndjson"))
        self.assertEqual(res.headers.get("X-Accel-Buffering"),"no")
        events=self.parse(res)
        self.assertEqual([e["type"] for e in events],["delta","delta","final"])
        self.assertEqual("".join(e["text"] for e in events if e["type"]=="delta"),"Guten Morgen")
        self.assertEqual(events[-1]["result"]["text"],"Guten Morgen")
        self.assertEqual(events[-1]["result"]["model_origin"],"local_foundation_not_neral_adapter")
        self.assertEqual(events[-1]["result"]["mode"],"LOW")
        self.assertEqual(len(fake.calls),1)

    def test_bad_stream_emits_error_but_never_a_final(self):
        fake=FakeStreamingBackend(fail=True)
        app=Flask("neral-streaming-error")
        register_neral_local_chat_api(app,fake)
        client=app.test_client()
        result=client.post("/api/neral/chat/generate-stream",json={"prompt":"Sag hallo"})
        events=self.parse(result)
        self.assertEqual(events[-1]["type"],"error")
        self.assertNotIn("result",events[-1])
        self.assertFalse(any(e["type"]=="final" for e in events))
        self.assertNotIn("Cannot produce",events[-1]["error"])

    def test_csrf_origin_and_request_contract_still_apply(self):
        app=create_preview_app(FakeStreamingBackend(),secret_key=b"R"*32)
        client=app.test_client()
        base="http://127.0.0.1:9690"
        client.get("/",base_url=base)
        session=client.get("/api/neral/preview/session",base_url=base)
        headers={"X-Neral-CSRF":session.json["csrf"]}
        for h in ({},{"X-Neral-CSRF":"wrong"},{"X-Neral-CSRF":headers["X-Neral-CSRF"],"Origin":"https://wrong.example"}):
            r=client.post("/api/neral/chat/generate-stream",json={"prompt":"Hallo"},headers=h,base_url=base)
            self.assertEqual(r.status_code,403)
        ok=client.post("/api/neral/chat/generate-stream",json={"prompt":"Hallo"},headers=headers,base_url=base)
        self.assertEqual(ok.status_code,200)
        self.assertEqual(self.parse(ok)[-1]["type"],"final")
        unknown=client.post("/api/neral/chat/generate-stream",json={"prompt":"Hallo","secret":"oops"},headers=headers,base_url=base)
        self.assertEqual(unknown.status_code,400)

    def test_lock_is_released_after_response_completion(self):
        fake=FakeStreamingBackend()
        app=Flask("neral-stream-lock")
        register_neral_local_chat_api(app,fake)
        client=app.test_client()
        first=client.post("/api/neral/chat/generate-stream",json={"prompt":"Frage 1"}, buffered=False)
        it=iter(first.response)
        next(it)
        second=client.post("/api/neral/chat/generate-stream",json={"prompt":"Frage 2"})
        self.assertEqual(second.status_code,409)
        first.close()
        third=client.post("/api/neral/chat/generate-stream",json={"prompt":"Frage 3"})
        self.assertEqual(third.status_code,200)
        self.assertEqual(self.parse(third)[-1]["type"],"final")


if __name__=="__main__":
    unittest.main()
