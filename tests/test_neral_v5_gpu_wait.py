"""Regression contracts for GPU recovery between 320-case Qwen and v5 inference."""
import subprocess
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.neral_v5_gpu_wait import (
    GPUReadyError, parse_gpu_csv, wait_until_gpu_ready,
    MIN_FREE_MIB, MAX_GPU_UTIL_PCT, MAX_CHECKS, POLL_SECONDS,
)

class V5GpuWaitTests(unittest.TestCase):
    def test_thresholds_and_bounded_budget(self):
        self.assertEqual(MIN_FREE_MIB, 9000)
        self.assertEqual(MAX_GPU_UTIL_PCT, 20)
        self.assertEqual(MAX_CHECKS, 19)
        self.assertEqual(POLL_SECONDS, 10)

    def test_strict_sensor_parsing_fails_closed(self):
        self.assertEqual(parse_gpu_csv("9000, 20\n"),(9000,20))
        self.assertEqual(parse_gpu_csv("12176, 0\r\n"),(12176,0))
        for value in ("", "NA, NA", "9000", "9000,20\n8000,0", "9000,-1",
                      "9000,101", "-4,0", "9000,3,2", "nan, 0"):
            with self.subTest(value=value):
                with self.assertRaises(GPUReadyError):
                    parse_gpu_csv(value)

    def test_recovers_after_prior_base_model_releases_gpu(self):
        readings=iter([(8450, 30),(8900, 0),(9000, 20)])
        delays=[]
        messages=[]
        result=wait_until_gpu_ready(
            "neral_v5",lambda:next(readings),delays.append,messages.append)
        self.assertEqual(result,(9000,20))
        self.assertEqual(delays,[10,10])
        self.assertTrue(any("8450" in message for message in messages))
        self.assertTrue(any("neral_v5" in message for message in messages))

    def test_failure_after_bounded_attempts_and_no_relaxed_threshold(self):
        delays=[]
        calls=[]
        def busy():
            calls.append(True)
            return (8999, 21)
        with self.assertRaisesRegex(GPUReadyError,"still busy"):
            wait_until_gpu_ready("neral_v5",busy,delays.append,lambda _:None)
        self.assertEqual(len(calls),19)
        self.assertEqual(delays,[10]*18)

    def test_sensor_error_not_silently_treated_as_ready(self):
        for sensor in (lambda: (_ for _ in ()).throw(GPUReadyError("sensor failed")),
                       lambda: (8999, 0, 1)):
            with self.assertRaises(GPUReadyError):
                wait_until_gpu_ready("base",sensor,lambda _:None,lambda _:None)

    def test_cli_is_present_in_both_inference_variants_only(self):
        yaml=Path(".github/workflows/neral-v5-style-eval.yml").read_text(encoding="utf8")
        self.assertIn("foreach($variant in @('base','neral_v5'))",yaml)
        self.assertIn("scripts\\neral_v5_gpu_wait.py",yaml)
        self.assertIn("--variant $variant",yaml)
        self.assertIn("GPU readiness check failed",yaml)
        self.assertNotIn("GPU busy / under 9000 MiB before $variant",yaml)
        self.assertNotIn("schedule:",yaml)
        self.assertIn("workflow_dispatch:",yaml)
        self.assertIn("neral-qwen7b-r32-gpu-exclusive",yaml)

if __name__=="__main__":
    unittest.main()
