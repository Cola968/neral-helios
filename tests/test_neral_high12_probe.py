"""CPU-only contracts for the one-task HIGH12 Ollama VRAM/timing preflight."""
import json
import unittest
from unittest.mock import patch

from scripts.neral_high12_probe import (
    measured_chat_payload, running_model_summary, model_fully_on_gpu,
    sanitized_timings, select_probe_case, PreflightError
)


class RequestParity(unittest.TestCase):
    def test_exact_measured_request_think_and_stream_are_top_level(self):
        turns = [{"role": "system", "content": "NERAL"},
                 {"role": "user", "content": "What is 6*7?"}]
        payload = measured_chat_payload(turns)
        self.assertIs(payload["think"], False)
        self.assertIs(payload["stream"], False)
        self.assertNotIn("think", payload["options"])
        self.assertNotIn("stream", payload["options"])
        self.assertEqual(payload["model"], "qwen3.5:9b")
        self.assertEqual(payload["options"], {
            "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5,
            "repeat_penalty": 1.0, "num_predict": 2048,
            "num_ctx": 4096,
        })
        self.assertEqual(payload["keep_alive"], "5m")
        self.assertEqual(payload["messages"], turns)

    def test_model_backend_sends_exact_same_wire_options(self):
        from neral_agent_autonomy import OllamaLocalBackend
        turns = [{"role": "system", "content": "NERAL"},
                 {"role": "user", "content": "Explain a star."}]
        captured = []
        class Reply:
            status = 200
            def __enter__(self): return self
            def __exit__(self, *_): pass
            def read(self, _max): return json.dumps({
                "done": True, "done_reason": "stop",
                "message": {"role": "assistant", "content": "A star."},
            }).encode()
        class Local:
            def open(self, request, timeout):
                captured.append((json.loads(request.data), timeout))
                return Reply()
        backend = OllamaLocalBackend(
            "qwen3.5:9b", timeout=90,
            inference_profile="measured_nonthinking_2048")
        with patch("neral_agent_autonomy.local_only_opener", return_value=Local()):
            backend.respond_chat(turns)
        self.assertEqual(len(captured), 1)
        self.assertEqual(captured[0][0], measured_chat_payload(turns))
        self.assertEqual(captured[0][1], 90.0)


class GpuPsContracts(unittest.TestCase):
    def test_detects_low_model_and_full_high_gpu(self):
        ps = {"models": [
            {"name": "qwen3:8b", "size": 5000, "size_vram": 5000},
            {"name": "qwen3.5:9b", "size": 7000, "size_vram": 7000}
        ]}
        data = running_model_summary(ps)
        self.assertEqual(set(data), {"qwen3:8b", "qwen3.5:9b"})
        self.assertTrue(model_fully_on_gpu(data["qwen3.5:9b"]))
        self.assertFalse(model_fully_on_gpu({"size": 7000, "size_vram": 6000}))

    def test_fail_closed_if_model_offloaded_or_unknown_or_ps_malformed(self):
        for ps in (
            {"models": [{"name": "qwen3.5:9b", "size": 7000, "size_vram": 6899}]},
            {"models": [{"name": "unknown-model:4b", "size": 6000, "size_vram": 6000}]},
            {"models": [{"name": "qwen3.5:9b", "size": "7000", "size_vram": 7000}]},
            {"models": [{"name": "qwen3.5:9b", "size": 7000, "size_vram": 7000},
                        {"name": "qwen3.5:9b", "size": 7000, "size_vram": 7000}]},
            {"models": "wrong"},
        ):
            with self.subTest(ps=ps), self.assertRaises(PreflightError):
                running_model_summary(ps)

    def test_missing_gpu_bytes_or_zero_is_not_fully_gpu(self):
        for item in ({"size": 7000, "size_vram": 0},
                     {"size": 7000, "size_vram": 6000},
                     {"size": 0, "size_vram": 0}):
            self.assertFalse(model_fully_on_gpu(item))


class TimingDiagnostics(unittest.TestCase):
    def test_ollama_nanos_are_renamed_to_seconds_without_model_content(self):
        packet = {"done": True, "done_reason": "stop",
                  "message": {"role": "assistant", "content": "PRIVATE ANSWER"},
                  "load_duration": 2_000_000_000,
                  "prompt_eval_duration": 500_000_000,
                  "eval_duration": 4_000_000_000,
                  "total_duration": 6_500_000_000,
                  "eval_count": 40,
                  "prompt_eval_count": 25}
        values = sanitized_timings(packet)
        self.assertEqual(values["load_seconds"], 2.0)
        self.assertEqual(values["prompt_eval_seconds"], 0.5)
        self.assertEqual(values["eval_seconds"], 4.0)
        self.assertEqual(values["tokens_per_second"], 10.0)
        self.assertNotIn("PRIVATE ANSWER", str(values))

    def test_unknown_duration_and_unfinished_packet_fail(self):
        with self.assertRaises(PreflightError):
            sanitized_timings({"done": False, "done_reason": "length"})
        with self.assertRaises(PreflightError):
            sanitized_timings({"done": True, "done_reason": "stop",
                               "eval_duration": -1})



class GpuSamplerContracts(unittest.TestCase):
    def test_gpu_monitor_sanitizes_utilization_samples(self):
        from scripts.neral_high12_probe import parse_gpu_telemetry, summarize_gpu_samples
        self.assertEqual(parse_gpu_telemetry("92, 6194"), (92, 6194))
        summary = summarize_gpu_samples([(0, 6050), (94, 6030), (98, 6010)])
        self.assertEqual(summary["gpu_util_samples_percent"], [0, 94, 98])
        self.assertEqual(summary["gpu_util_peak_percent"], 98)
        self.assertEqual(summary["samples_count"], 3)
        self.assertEqual(summary["vram_used_peak_mib"], 6050)

    def test_invalid_gpu_samples_rejected(self):
        from scripts.neral_high12_probe import parse_gpu_telemetry
        for line in ("", "101, 6000", "-1, 1", "100, -1", "ten, 2",
                     "12, 5, 42", "12, 5\\nprivate", "0.1, 50"):
            with self.subTest(line=line):
                self.assertIsNone(parse_gpu_telemetry(line))



class PilotSelection(unittest.TestCase):
    def test_one_representative_math_case_deterministic(self):
        a = select_probe_case()
        b = select_probe_case()
        self.assertEqual(a, b)
        self.assertEqual(a["domain"], "math")
        self.assertIsInstance(a["answer"], int)
        self.assertTrue(a["prompt"])


if __name__ == "__main__":
    unittest.main()
