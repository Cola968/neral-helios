"""CPU-only telemetry contracts. No model weights or private datasets."""
import threading
import unittest
from unittest.mock import patch
from flask import Flask
from neral_v42_telemetry import (
    RequestTelemetry, FirstTokenStopper, current, begin_request, end_request,
    summarize_placement, timed_tool,
)
from neral_three_tier_runtime import register_three_tier_api
from test_neral_three_tier_api import make_runtime


class TelemetryCPU(unittest.TestCase):
    def test_scoped_load_ttft_tokens_rate_and_no_private_text(self):
        metric, token = begin_request()
        try:
            self.assertIs(current(), metric)
            metric.load(False, 1234, type("Fake", (), {"loaded":False})())
            metric.generation(input_tokens=420, system_tokens=200, max_new_tokens=256,
                              first_token_at=2.0, started_at=1.0, finished_at=4.0, count=21)
            diag = metric.summary()
            self.assertEqual(diag["model_load_ms"], 1234)
            self.assertEqual(diag["model_load_cold"], 1)
            self.assertEqual(diag["ttft_ms"],1000)
            self.assertEqual(diag["gen_tokens"],21)
            self.assertAlmostEqual(diag["tokens_per_sec"],10.)
            self.assertEqual(diag["system_prompt_tokens"],200)
            self.assertNotIn("text",diag)
            self.assertNotIn("prompt",diag)
            self.assertFalse(any(isinstance(v,str) and "secret" in v for v in diag.values()))
        finally:
            end_request(token)
        self.assertIsNone(current())

    def test_first_token_callback_observes_sampled_token_and_preserves_stopping(self):
        times=iter([3.5,7.0])
        first=FirstTokenStopper(lambda *args,**kwargs: False)
        with patch("neral_v42_telemetry.time.monotonic",side_effect=lambda:next(times)):
            self.assertFalse(first(None,None))
            self.assertFalse(first(None,None))
        self.assertEqual(first.first_at,3.5)

    def test_tool_timing_and_exception_does_not_hide_failure(self):
        metric,token=begin_request()
        try:
            fn=timed_tool("python_z3",lambda n: n+1)
            self.assertEqual(fn(10),11)
            self.assertGreaterEqual(metric.summary()["tool_ms"],0)
            def broken():
                raise RuntimeError("internal")
            with self.assertRaises(RuntimeError):
                timed_tool("web_v4",broken)()
            self.assertIn("web_v4",metric.summary()["tool_breakdown_ms"])
        finally:
            end_request(token)

    def test_device_counts_dont_log_names(self):
        class Param:
            def __init__(self,t):
                self.device=type("Device",(),{"type":t})()
        class M:
            hf_device_map={"model.layers.0":"cuda:0", "x":"cpu"}
            def parameters(self):
                return iter((Param("cuda"),Param("cpu"),Param("meta")))
        adapter=type("Adapter",(),{"_model":M(),"_base_model":None})()
        result=summarize_placement(adapter)
        self.assertEqual(result["gpu_parameter_tensors"],1)
        self.assertEqual(result["cpu_parameter_tensors"],1)
        self.assertEqual(result["disk_parameter_tensors"],1)
        self.assertEqual(result["hf_gpu_module_entries"],1)
        self.assertNotIn("model.layers",str(result))

    def test_telemetry_is_opt_in_no_training_only(self):
        runtime,*_=make_runtime()
        app=Flask(__name__)
        app.secret_key="secret"
        register_three_tier_api(app,runtime,busy=threading.Lock())
        client=app.test_client()
        body={"tier":"Quartz","prompt":"Hallo","diagnostic_telemetry":True}
        self.assertEqual(client.post("/api/neral/chat/three-tier",json=body).status_code,400)
        # Existing owner middleware and model-free fake runtime are separately
        # tested by the full three-tier API suite.

if __name__=="__main__":
    unittest.main()
