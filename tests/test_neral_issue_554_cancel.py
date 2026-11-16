"""Issue 554: safely cancel only the matching active Ollama request."""
import json
import threading
import unittest
from unittest.mock import patch

from neral_agent_autonomy import OllamaCompletionError, OllamaLocalBackend
from neral_preview_server import create_preview_app

REQUEST=[{"role":"system","content":"Local assistant"},
         {"role":"user","content":"Erkläre ausführlich Keplers Gesetz."}]

def line(text, *, done=False, reason=None):
    obj={"done":done,"message":{"content":text}}
    if done:obj["done_reason"]=reason or "stop"
    return (json.dumps(obj)+"\n").encode("utf8")

class FixtureStream:
    def __init__(self,event,*, trigger_stop):
        self.event=event
        self.trigger_stop=trigger_stop
        self.closed=False
    def __enter__(self):return self
    def __exit__(self,*args):
        self.closed=True
        return False
    def __iter__(self):
        yield line("Schritt 1: ")
        if self.trigger_stop:
            self.event.set()
        yield line("Schritt 2: ")
        yield line("",done=True)

class LocalTransport:
    def __init__(self,response):
        self.response=response
        self.payloads=[]
    def open(self,request,timeout=0):
        self.payloads.append(json.loads(request.data))
        return self.response

class Cancellation554(unittest.TestCase):
    def test_high_opt_in_stream_closes_socket_and_does_not_return_partial(self):
        event=threading.Event()
        response=FixtureStream(event,trigger_stop=True)
        transport=LocalTransport(response)
        model=OllamaLocalBackend("qwen3.5:9b",inference_profile="measured_nonthinking_2048")
        with patch("neral_agent_autonomy.local_only_opener",return_value=transport):
            with self.assertRaises(OllamaCompletionError) as caught:
                model.respond_high_final(REQUEST,cancel_event=event)
        self.assertEqual(caught.exception.reason,"cancelled")
        self.assertTrue(response.closed)
        self.assertEqual(len(transport.payloads),1)
        self.assertTrue(transport.payloads[0]["stream"])

    def test_cancellable_high_success_reassembles_chunks_without_changing_model_settings(self):
        event=threading.Event()
        response=FixtureStream(event,trigger_stop=False)
        transport=LocalTransport(response)
        model=OllamaLocalBackend("qwen3.5:9b",inference_profile="measured_nonthinking_2048")
        with patch("neral_agent_autonomy.local_only_opener",return_value=transport):
            result=model.respond_high_final(REQUEST,cancel_event=event)
        self.assertEqual(result,"Schritt 1: Schritt 2: ")
        self.assertTrue(response.closed)
        self.assertEqual(transport.payloads[0]["options"]["num_predict"],2048)
        self.assertFalse(transport.payloads[0]["think"])

    def test_cancel_endpoint_requires_matching_id_and_local_csrf(self):
        class Fake:
            model="qwen3:8b"
            def respond_chat(self,_):return "ok"
        client=create_preview_app(Fake(),secret_key=b"S"*32).test_client()
        host="http://127.0.0.1:9690"
        client.get("/",base_url=host)
        csrf=client.get("/api/neral/preview/session",base_url=host).json["csrf"]
        headers={"X-Neral-CSRF":csrf}
        rid="f"*32
        wrong=client.post("/api/neral/chat/cancel",
                          json={"request_id":rid},base_url=host)
        self.assertEqual(wrong.status_code,403)
        status=client.post("/api/neral/chat/cancel",json={"request_id":rid},
                          headers=headers,base_url=host)
        self.assertEqual(status.status_code,409)
        registry=client.application._neral_chat_cancel_registry
        active=registry.begin(rid)
        self.assertFalse(active.is_set())
        other=client.post("/api/neral/chat/cancel",
                          json={"request_id":"e"*32},headers=headers,base_url=host)
        self.assertEqual(other.status_code,409)
        self.assertFalse(active.is_set())
        ok=client.post("/api/neral/chat/cancel",
                       json={"request_id":rid},headers=headers,base_url=host)
        self.assertEqual(ok.status_code,202)
        self.assertTrue(active.is_set())
        registry.end(active)
        retry=client.post("/api/neral/chat/cancel",
                          json={"request_id":rid},headers=headers,base_url=host)
        self.assertEqual(retry.status_code,409)

if __name__=="__main__":
    unittest.main()
