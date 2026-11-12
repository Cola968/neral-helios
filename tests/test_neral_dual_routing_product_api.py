"""Opt-in productive dual route contract: no default change or silent model substitution."""
import os
import unittest
from unittest.mock import patch
from neral_agent_autonomy import OllamaLocalBackend
from neral_preview_server import create_preview_app

class DualApiContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=create_preview_app(
            OllamaLocalBackend("qwen3:8b"),
            experimental_high_backend=OllamaLocalBackend(
                "qwen3.5:9b",inference_profile="measured_nonthinking_2048"))
    def request(self,payload):
        with self.app.test_client() as c:
            c.get("/",base_url="http://127.0.0.1:9690")
            tok=c.get("/api/neral/preview/session",
                      base_url="http://127.0.0.1:9690").json["csrf"]
            return c.post("/api/neral/chat/think",json=payload,
                headers={"X-Neral-CSRF":tok},
                base_url="http://127.0.0.1:9690")
    def test_reject_dual_without_both_conservative_and_tir_flags(self):
        for fields in [
          {"dual_routing":True},
          {"dual_routing":True,"experimental_tir":True},
          {"dual_routing":True,"conservative_tir":True},
          {"dual_routing":"true","experimental_tir":True,"conservative_tir":True}
        ]:
            res=self.request({"prompt":"Calculate the number of paths",**fields})
            self.assertEqual(res.status_code,400,res.json)
    def test_dual_must_be_opted_in_by_environment(self):
        with patch.dict(os.environ,{"NERAL_EXPERIMENTAL_TIR":"1","NERAL_EXPERIMENTAL_DUAL_ROUTE":"0"}):
            res=self.request({"prompt":"Count 13 items","experimental_tir":True,
                "conservative_tir":True,"dual_routing":True})
        self.assertEqual(res.status_code,403,res.json)
    def test_explicit_dual_routes_via_pinned_adapter_and_base(self):
        mock={"text":"42","mode":"HIGH","verified":True,
              "verification_status":"verified","verification_scope":"exact_user_python_execution",
              "response_source":"verified_tir","selection_policy":"independently_verified_tir",
              "dual_routing":True,"tool_model":"neral-qwen35-qlora-v1:latest",
              "fallback_model":"qwen35-9b-qlora-control-20261112:latest",
              "gpu_device_peak_mib":10400}
        with patch.dict(os.environ,{"NERAL_EXPERIMENTAL_TIR":"1","NERAL_EXPERIMENTAL_DUAL_ROUTE":"1"}),patch(
            "neral_dual_routing.run_dual_route",return_value=mock) as run:
            res=self.request({"prompt":"How many paths","experimental_tir":True,
                 "conservative_tir":True,"dual_routing":True})
        self.assertEqual(res.status_code,200,res.json)
        self.assertEqual(res.json["text"],"42")
        self.assertEqual(res.json["selection_policy"],"independently_verified_tir")
        self.assertEqual(run.call_count,1)
        self.assertEqual(run.call_args.args[0][-1]["content"],"How many paths")
    def test_no_unvalidated_claim_if_model_digest_wrong(self):
        with patch.dict(os.environ,{"NERAL_EXPERIMENTAL_TIR":"1","NERAL_EXPERIMENTAL_DUAL_ROUTE":"1"}),patch(
            "neral_dual_routing.run_dual_route",
            side_effect=RuntimeError("secret raw prompt / wrong digest")):
            res=self.request({"prompt":"How many paths","experimental_tir":True,
                 "conservative_tir":True,"dual_routing":True})
        self.assertEqual(res.status_code,503)
        self.assertNotIn("secret",str(res.json))
    def test_normal_think_legacy_unaffected(self):
        with patch.dict(os.environ,{"NERAL_EXPERIMENTAL_TIR":"0","NERAL_EXPERIMENTAL_DUAL_ROUTE":"1"}):
            res=self.request({"prompt":"Count 13 items","experimental_tir":True,
                "conservative_tir":True,"dual_routing":True})
        self.assertEqual(res.status_code,403)
if __name__=="__main__":unittest.main()
