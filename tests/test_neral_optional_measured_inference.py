"""Opt-in benchmark-matched inference; baseline release behavior stays unchanged."""
import json
import unittest
from unittest.mock import patch
from neral_agent_autonomy import AgentModelError, OllamaLocalBackend

class _Reply:
    def __init__(self, packet):
        self.raw=json.dumps(packet).encode("utf8")
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def read(self,n): return self.raw

class _OfflineOllama:
    def __init__(self, packet=None):
        self.packet=packet or {"done":True,"done_reason":"stop",
                                "message":{"content":"FINAL_ANSWER: 42",
                                           "thinking":"NEVER_EXPOSE_PRIVATE_THOUGHT"}}
        self.calls=[]
    def open(self,request,timeout):
        self.calls.append((request.full_url,json.loads(request.data),timeout))
        return _Reply(self.packet)

MSG=[{"role":"system","content":"You are local."},
     {"role":"user","content":"Find the answer."}]

class BenchmarkProfileTests(unittest.TestCase):
    def test_baseline_remains_identical_by_default(self):
        stub=_OfflineOllama()
        with patch("neral_agent_autonomy.local_only_opener",return_value=stub):
            b=OllamaLocalBackend("qwen3:8b")
            self.assertEqual(b.respond_chat(MSG),"FINAL_ANSWER: 42")
            self.assertEqual(b.respond_think(MSG),"FINAL_ANSWER: 42")
        low,high=(c[1] for c in stub.calls)
        self.assertEqual(low["model"],"qwen3:8b")
        self.assertEqual(low["options"]["temperature"],0)
        self.assertEqual(low["options"]["num_predict"],1024)
        self.assertEqual(low["think"],False)
        self.assertEqual(high["options"]["temperature"],0)
        self.assertEqual(high["options"]["num_predict"],2048)
        self.assertEqual(high["think"],True)

    def test_explicit_benchmark_profile_matches_successful_2048_probe(self):
        for model in ("qwen3:8b","qwen3.5:9b"):
            with self.subTest(model=model):
                stub=_OfflineOllama()
                with patch("neral_agent_autonomy.local_only_opener",return_value=stub):
                    b=OllamaLocalBackend(model,inference_profile="measured_nonthinking_2048")
                    self.assertEqual(b.respond_chat(MSG),"FINAL_ANSWER: 42")
                    self.assertEqual(b.respond_think(MSG),"FINAL_ANSWER: 42")
                self.assertEqual(b.inference_profile,"measured_nonthinking_2048")
                self.assertEqual(len(stub.calls),2)
                for _,data,_ in stub.calls:
                    self.assertEqual(data["model"],model)
                    self.assertEqual(data["think"],False)
                    self.assertNotIn("format",data)
                    opt=data["options"]
                    self.assertEqual(opt["temperature"],0.7)
                    self.assertEqual(opt["top_p"],0.8)
                    self.assertEqual(opt["top_k"],20)
                    self.assertEqual(opt["min_p"],0.0)
                    self.assertEqual(opt["presence_penalty"],1.5)
                    self.assertEqual(opt["repeat_penalty"],1.0)
                    self.assertEqual(opt["num_predict"],2048)
                    self.assertEqual(opt["num_ctx"],4096)

    def test_agent_json_is_unchanged_even_with_opt_in(self):
        stub=_OfflineOllama()
        with patch("neral_agent_autonomy.local_only_opener",return_value=stub):
            b=OllamaLocalBackend("qwen3:8b",inference_profile="measured_nonthinking_2048")
            b.respond(MSG)
        data=stub.calls[0][1]
        self.assertEqual(data["options"]["temperature"],0)
        self.assertEqual(data["options"]["num_predict"],512)
        self.assertEqual(data["format"],"json")
        self.assertFalse(data["think"])

    def test_reject_unknown_profiles_and_no_implicit_model_switch(self):
        for profile in ("",None,"surprise","MEASURED_NONTHINKING_2048"):
            with self.subTest(profile=profile),self.assertRaises(ValueError):
                OllamaLocalBackend("qwen3:8b",inference_profile=profile)
        b=OllamaLocalBackend("qwen3:8b",inference_profile="measured_nonthinking_2048")
        self.assertEqual(b.model,"qwen3:8b")
        self.assertFalse(hasattr(b,"high_model"))

    def test_incomplete_answer_still_refused_with_opt_in(self):
        stub=_OfflineOllama({"done":True,"done_reason":"length",
                             "message":{"content":"FINAL_ANSWER: 42"}})
        with patch("neral_agent_autonomy.local_only_opener",return_value=stub):
            b=OllamaLocalBackend("qwen3.5:9b",inference_profile="measured_nonthinking_2048")
            with self.assertRaises(AgentModelError):
                b.respond_chat(MSG)

    def test_http_provenance_reports_opt_in_and_does_not_claim_neural_gain(self):
        from neral_local_chat_api import _provenance
        b=OllamaLocalBackend("qwen3:8b",inference_profile="measured_nonthinking_2048")
        status=_provenance(b)
        self.assertEqual(status["inference_profile"],"measured_nonthinking_2048")
        self.assertFalse(status["quality_improvement_proven"])
        self.assertFalse(status["adapter_loaded"])

if __name__=="__main__":
    unittest.main()
