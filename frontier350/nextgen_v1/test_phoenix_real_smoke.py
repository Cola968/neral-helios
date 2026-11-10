"""CPU-only contracts; fake model outputs are not actual inference evidence."""
import io
import json
from contextlib import redirect_stdout
from unittest.mock import patch
import unittest

from phoenix_real_smoke import SMOKE_PROMPTS, parse_gpu_row, require_idle_gpu, smoke


class RealSmokeContracts(unittest.TestCase):
    def test_gpu_telemetry_parsing(self):
        self.assertEqual(parse_gpu_row(" 0, 1024, 58\n"), (0, 1024, 58))

    def test_malformed_or_multiple_gpus_fail_closed(self):
        for data in ("", "0, 1000", "0, 1000, 60\n1, 1000, 60", "not a gpu", "101, 0, 60"):
            with self.subTest(data=data), self.assertRaises(ValueError):
                parse_gpu_row(data)

    def test_busy_gpu_refuses_before_checkpoint_load(self):
        for data in ("16, 0, 50", "0, 2501, 50", "0, 0, 79"):
            with self.subTest(data=data):
                with patch("phoenix_real_smoke.subprocess.check_output", return_value=data):
                    with self.assertRaises(RuntimeError):
                        require_idle_gpu()

    def test_idle_gpu_admitted(self):
        with patch("phoenix_real_smoke.subprocess.check_output", return_value="3, 1200, 61"):
            require_idle_gpu()

    def test_eight_unique_development_prompts(self):
        self.assertEqual(len(SMOKE_PROMPTS), 8)
        self.assertEqual(len(set(SMOKE_PROMPTS)), 8)

    def test_smoke_does_not_log_answers_or_prompts(self):
        def fake_capture(model, tokenizer, bos, eos, torch, prompt, budget):
            self.assertIn(prompt, SMOKE_PROMPTS)
            self.assertEqual(budget, 1)
            return "PRIVATE_MODEL_ANSWER", "max_tokens", 1
        stream = io.StringIO()
        with patch("phoenix_real_smoke.capture_generation", side_effect=fake_capture):
            with redirect_stdout(stream):
                result = smoke(None, None, 1, 2, None)
        self.assertEqual(result["cases"], 8)
        self.assertEqual(result["actual_eos"], 0)
        self.assertEqual(result["nonempty"], 8)
        self.assertFalse(result["quality_evaluated"])
        self.assertFalse(result["training_started"])
        self.assertNotIn("PRIVATE_MODEL_ANSWER", stream.getvalue())
        for prompt in SMOKE_PROMPTS:
            self.assertNotIn(prompt, stream.getvalue())
        self.assertEqual(sum("case" in json.loads(line) for line in stream.getvalue().splitlines()), 8)

    def test_invalid_decoder_result_fails(self):
        with patch("phoenix_real_smoke.capture_generation", return_value=("x", "stop", 1)):
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                smoke(None, None, 1, 2, None)


if __name__ == "__main__":
    unittest.main()
