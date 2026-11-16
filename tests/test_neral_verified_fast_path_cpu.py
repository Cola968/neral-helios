"""Opt-in fast proof-first path: no model-dependent choice before safe source check.

The original full Dual-Routing remains untouched and default.
"""
import os
import unittest
from unittest.mock import patch
from neral_agent_autonomy import OllamaLocalBackend
from neral_preview_server import create_preview_app
from neral_dual_routing import BASE_TAG,TUNED_TAG,run_verified_source_fast

MESSAGES=[{"role":"system","content":"NERAL test"},
          {"role":"user","content":"Count permutations for an unfamiliar problem"}]

class FastPathCPU(unittest.TestCase):
    def test_arbitrary_unstructured_question_requires_only_base_no_adapter(self):
        called=[]
        def respond(model,messages,seed,*,structured=False,repair=False):
            called.append((model,messages,structured))
            if model!=BASE_TAG:raise AssertionError("QLoRA must not be called")
            return "FINAL_ANSWER: 8",{"model":model,"load_ms":1,
                    "model_residency_after":{"models_vram_mib":{BASE_TAG:5600}}}
        with patch("neral_dual_routing.require_exact_models",
                   return_value={BASE_TAG:"digest",TUNED_TAG:"digest2"}), \
             patch("neral_dual_routing.require_exact_guard",return_value="pinned"), \
             patch("neral_dual_routing.sample_residency",return_value={"both_resident":False}), \
             patch("neral_dual_routing._ollama_chat",side_effect=respond):
            result=run_verified_source_fast(MESSAGES,sample_gpu=False)
        self.assertEqual(len(called),1)
        self.assertEqual(called[0][0],BASE_TAG)
        self.assertEqual(called[0][1],MESSAGES)
        self.assertEqual(result["text"],"FINAL_ANSWER: 8")
        self.assertFalse(result["verified"])
        self.assertEqual(result["response_source"],"raw_qwen")
        self.assertEqual(result["selection_policy"],"base_fast_unverified")
        self.assertIsNone(result["tool_model"])
        self.assertEqual(result["fallback_model"],BASE_TAG)
        self.assertEqual(result["structured_schema_attempts"],0)
        self.assertEqual(result["tool_planning_calls"],0)
        self.assertEqual(result["base_raw_calls"],1)

    def test_explicit_verified_source_stays_guard_verified_without_any_model(self):
        candidate={"text":"19","mode":"HIGH","model_generated":False,
             "verified":True,"verification_status":"verified",
             "verification_scope":"exact_user_python_execution",
             "review_method":"exact_user_ast_match",
             "tool_used":"neral_python_sandbox",
             "tool_calls_succeeded":1,
             "tool_calls_triggered":1,
             "calls_used":0,"seconds":.1}
        with patch("neral_dual_routing.require_exact_models",return_value={}), \
             patch("neral_dual_routing.require_exact_guard",return_value="pinned"), \
             patch("neral_dual_routing.sample_residency",return_value={}), \
             patch("neral_dual_routing.run_experimental_tir",return_value=candidate), \
             patch("neral_dual_routing._ollama_chat",
                   side_effect=AssertionError("No model needed")):
            result=run_verified_source_fast(MESSAGES,sample_gpu=False)
        self.assertTrue(result["verified"])
        self.assertEqual(result["selection_policy"],"independently_verified_tir")
        self.assertEqual(result["text"],"19")
        self.assertEqual(result["base_raw_calls"],0)

    def test_explicit_source_unavailable_is_not_replaced_by_untrusted_raw(self):
        refused={"text":"Source could not be safely executed","mode":"HIGH",
             "model_generated":False,"verified":False,
             "verification_status":"not_available","verification_scope":"none",
             "tool_calls_succeeded":0,"tool_calls_triggered":1,
             "calls_used":0,"seconds":.1}
        with patch("neral_dual_routing.require_exact_models",return_value={}), \
             patch("neral_dual_routing.require_exact_guard",return_value="pinned"), \
             patch("neral_dual_routing.sample_residency",return_value={}), \
             patch("neral_dual_routing.run_experimental_tir",return_value=refused), \
             patch("neral_dual_routing._ollama_chat",
                   side_effect=AssertionError("Cannot guess after safe refusal")):
            result=run_verified_source_fast(MESSAGES,sample_gpu=False)
        self.assertFalse(result["verified"])
        self.assertEqual(result["selection_policy"],"explicit_safe_refusal")

class FastApiContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=create_preview_app(OllamaLocalBackend("qwen3:8b"),
            experimental_high_backend=OllamaLocalBackend(
                "qwen3.5:9b",inference_profile="measured_nonthinking_2048"))
    def request(self,payload):
        with self.app.test_client() as c:
            c.get("/",base_url="http://127.0.0.1:9690")
            csrf=c.get("/api/neral/preview/session",
                       base_url="http://127.0.0.1:9690").json["csrf"]
            return c.post("/api/neral/chat/think",json=payload,
                    headers={"X-Neral-CSRF":csrf},base_url="http://127.0.0.1:9690")
    def test_fast_requires_dual_and_both_explicit_switches(self):
        payload={"prompt":"Hello","experimental_tir":True,
                 "conservative_tir":True,"dual_routing":True,
                 "fast_verified_only":True}
        with patch.dict(os.environ,{"NERAL_EXPERIMENTAL_TIR":"1",
            "NERAL_EXPERIMENTAL_DUAL_ROUTE":"1",
            "NERAL_EXPERIMENTAL_FAST_VERIFIED_ONLY":"0"}):
            self.assertEqual(self.request(payload).status_code,403)
        self.assertEqual(self.request({**payload,"fast_verified_only":"yes"}).status_code,400)
        self.assertEqual(self.request({"prompt":"Hello","fast_verified_only":True}).status_code,400)

    def test_opted_in_fast_returns_true_base_provenance_without_qlora_claim(self):
        answer={"text":"8","verified":False,"mode":"HIGH",
                "response_source":"raw_qwen",
                "selection_policy":"base_fast_unverified",
                "fallback_model":BASE_TAG,"tool_model":None}
        with patch.dict(os.environ,{"NERAL_EXPERIMENTAL_TIR":"1",
              "NERAL_EXPERIMENTAL_DUAL_ROUTE":"1",
              "NERAL_EXPERIMENTAL_FAST_VERIFIED_ONLY":"1"}),patch(
              "neral_dual_routing.run_verified_source_fast",
              return_value=answer) as route:
            r=self.request({"prompt":"Hello","experimental_tir":True,
                "conservative_tir":True,"dual_routing":True,
                "fast_verified_only":True})
        self.assertEqual(r.status_code,200,r.json)
        self.assertEqual(r.json["model"],BASE_TAG)
        self.assertEqual(r.json["model_origin"],"local_qwen_verified_fast")
        self.assertEqual(route.call_count,1)

if __name__=="__main__":unittest.main()
