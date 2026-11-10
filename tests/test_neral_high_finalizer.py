"""CPU-only finalizer contracts: no invented finals, no leaking partial text."""
from __future__ import annotations
import json
import unittest
from unittest.mock import patch

from neral_agent_autonomy import OllamaCompletionError, OllamaLocalBackend
from neral_think import think_reply

REQUEST = [
    {"role":"system","content":"You are a local assistant."},
    {"role":"user","content":"Count the possible arrangements.\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."}
]

class Reply:
    def __init__(self, packet):
        self.packet=packet
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self,_): return json.dumps(self.packet).encode("utf8")

class Transport:
    def __init__(self, packets):
        self.packets=list(packets);self.calls=[]
    def open(self,request,timeout=0):
        self.calls.append(json.loads(request.data))
        return Reply(self.packets.pop(0))

class HighFinalizerTests(unittest.TestCase):
    def test_private_partial_is_available_only_inside_exception(self):
        model=OllamaLocalBackend("qwen3.5:9b",inference_profile="measured_nonthinking_2048")
        wire=Transport([{"done":True,"done_reason":"length",
            "message":{"content":"SECRET local model partial = 42"}}])
        with patch("neral_agent_autonomy.local_only_opener",return_value=wire):
            with self.assertRaises(OllamaCompletionError) as caught:
                model.respond_think(REQUEST)
        self.assertEqual(caught.exception.reason,"token_limit")
        self.assertEqual(caught.exception.partial_content,"SECRET local model partial = 42")
        self.assertNotIn("SECRET",str(caught.exception))
        self.assertEqual(len(wire.calls),1)

    def test_finalizer_has_exact_64_token_zero_temperature_payload(self):
        model=OllamaLocalBackend("qwen3.5:9b",inference_profile="measured_nonthinking_2048")
        wire=Transport([{"done":True,"done_reason":"stop",
            "message":{"content":"FINAL_ANSWER: 42"}}])
        with patch("neral_agent_autonomy.local_only_opener",return_value=wire):
            actual=model.respond_final(REQUEST)
        self.assertEqual(actual,"FINAL_ANSWER: 42")
        opts=wire.calls[0]["options"]
        self.assertEqual(opts["num_predict"],64)
        self.assertEqual(opts["temperature"],0)
        self.assertFalse(wire.calls[0]["think"])
        self.assertEqual(len(wire.calls),1)

    def test_terminal_length_is_recovered_only_by_strict_bounded_final(self):
        count=[]
        def first(_):
            count.append("first")
            raise OllamaCompletionError("token_limit",partial_content=
                "First I consider cases.\nThere are exactly 42 arrangements.")
        def final(turns):
            count.append("final")
            self.assertEqual(turns[-2]["role"],"assistant")
            self.assertIn("42 arrangements",turns[-2]["content"])
            self.assertIn("FINAL_ANSWER",turns[-1]["content"])
            return "FINAL_ANSWER: 42"
        packet=think_reply(REQUEST,first,deep_review=True,finalize=final)
        self.assertEqual(packet["text"],"FINAL_ANSWER: 42")
        self.assertFalse(packet["verified"])
        self.assertEqual(packet["calls_used"],2)
        self.assertTrue(packet["initial_length_truncated"])
        self.assertTrue(packet["finalizer_used"])
        self.assertEqual(count,["first","final"])

    def test_missing_final_on_complete_response_gets_one_bounded_attempt(self):
        used=[]
        packet=think_reply(REQUEST,lambda _: "The count is 42, after enumerating carefully.",
                           deep_review=True,finalize=lambda turns: (used.append(1) or "FINAL_ANSWER: 42"))
        self.assertEqual(packet["text"],"FINAL_ANSWER: 42")
        self.assertEqual(len(used),1)
        self.assertFalse(packet["initial_length_truncated"])
        self.assertTrue(packet["finalizer_used"])

    def test_no_unverified_final_on_failed_extraction(self):
        def first(_):
            raise OllamaCompletionError("token_limit",partial_content="No result derived.")
        with self.assertRaises(OllamaCompletionError) as caught:
            think_reply(REQUEST,first,deep_review=True,finalize=lambda _: "UNVERIFIED")
        self.assertEqual(caught.exception.reason,"token_limit")

    def test_short_regular_chat_never_triggers_finalizer(self):
        plain=[REQUEST[0],{"role":"user","content":"Explain turbofans."}]
        packet=think_reply(plain,lambda _: "Turbofans accelerate air.",deep_review=True,
                           finalize=lambda _: self.fail("Unexpected finalizer"))
        self.assertEqual(packet["calls_used"],1)
        self.assertEqual(packet["text"],"Turbofans accelerate air.")

if __name__=="__main__":
    unittest.main()
