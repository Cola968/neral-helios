"""CPU-only regression: the second 64-token call is HIGH-only, never verified."""
import json
import unittest
from unittest.mock import patch
from neral_agent_autonomy import OllamaCompletionError, OllamaLocalBackend
from neral_think import think_reply

REQUEST=[
    {"role":"system","content":"You are a local assistant."},
    {"role":"user","content":"Count the arrangements.\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."}
]

class Reply:
    def __init__(self,packet): self.packet=packet
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self,_): return json.dumps(self.packet).encode("utf8")

class Transport:
    def __init__(self,*packets): self.packets=list(packets);self.calls=[]
    def open(self,request,timeout=0):
        self.calls.append(json.loads(request.data))
        return Reply(self.packets.pop(0))

def packet(reason,content):
    return {"done":True,"done_reason":reason,"message":{"content":content}}

class HighFinalizerTests(unittest.TestCase):
    def model(self):
        return OllamaLocalBackend("qwen3.5:9b",inference_profile="measured_nonthinking_2048")

    def test_raw_qwen_length_still_fails_closed_with_no_private_partial_in_error(self):
        wire=Transport(packet("length","PRIVATE reasoning the count is 42"))
        with patch("neral_agent_autonomy.local_only_opener",return_value=wire):
            with self.assertRaises(OllamaCompletionError) as err:
                self.model().respond_chat(REQUEST)
        self.assertEqual(err.exception.reason,"token_limit")
        self.assertNotIn("PRIVATE",str(err.exception))
        self.assertEqual(len(wire.calls),1)

    def test_high_only_finalizer_has_64_tokens_and_zero_temperature(self):
        model=self.model()
        wire=Transport(packet("length","I counted all the cases; the result is 42."),
                       packet("stop","FINAL_ANSWER: 42"))
        with patch("neral_agent_autonomy.local_only_opener",return_value=wire):
            result=model.respond_high_final(REQUEST)
        self.assertEqual(result,"FINAL_ANSWER: 42")
        self.assertEqual(len(wire.calls),2)
        self.assertEqual(wire.calls[0]["options"]["num_predict"],2048)
        self.assertEqual(wire.calls[0]["options"]["temperature"],0.7)
        self.assertEqual(wire.calls[1]["options"]["num_predict"],64)
        self.assertEqual(wire.calls[1]["options"]["temperature"],0)
        self.assertFalse(wire.calls[1]["think"])
        self.assertEqual(wire.calls[1]["messages"][-2]["role"],"assistant")
        self.assertEqual(wire.calls[1]["messages"][-1]["role"],"user")
        self.assertTrue(any(e["final_recovery_succeeded"] for e in model.inference_diagnostic_events))
        self.assertNotIn("I counted all the cases",str(model.inference_diagnostic_events))

    def test_complete_long_reply_with_missing_final_can_be_closed_once(self):
        model=self.model()
        wire=Transport(packet("stop","After enumerating cases I obtain the value 42.\nFurther notes."),
                       packet("stop","FINAL_ANSWER: 42"))
        with patch("neral_agent_autonomy.local_only_opener",return_value=wire):
            result=model.respond_high_final(REQUEST)
        self.assertEqual(result,"FINAL_ANSWER: 42")
        self.assertEqual(len(wire.calls),2)

    def test_unverifiable_rescue_is_not_silently_claimed_as_correct(self):
        wire=Transport(packet("length","I am still calculating the value."),
                       packet("stop","UNSURE"))
        with patch("neral_agent_autonomy.local_only_opener",return_value=wire):
            with self.assertRaises(OllamaCompletionError) as err:
                self.model().respond_high_final(REQUEST)
        self.assertEqual(err.exception.reason,"token_limit")
        self.assertEqual(len(wire.calls),2)

    def test_non_benchmark_chat_never_gets_a_second_final_turn(self):
        plain=[REQUEST[0],{"role":"user","content":"Explain turbofan aircraft engines."}]
        wire=Transport(packet("stop","Turbofans accelerate bypass air."))
        with patch("neral_agent_autonomy.local_only_opener",return_value=wire):
            result=self.model().respond_high_final(plain)
        self.assertEqual(result,"Turbofans accelerate bypass air.")
        self.assertEqual(len(wire.calls),1)
        result=think_reply(plain,lambda _: "Turbofans accelerate bypass air.",deep_review=True)
        self.assertEqual(result["calls_used"],1)

if __name__=="__main__": unittest.main()
