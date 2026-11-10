"""Synthetic bounded development-diagnostic contracts, NOT own-weight evidence."""
import io
from contextlib import redirect_stdout
from unittest.mock import patch
import unittest

from phoenix_real_smoke import smoke


class DiagnosticContracts(unittest.TestCase):
    def test_full_budget_aggregates_eos_repetition_without_text_leak(self):
        calls = 0
        def fake_capture(model, tok, bos, eos, torch, prompt, budget):
            nonlocal calls
            calls += 1
            self.assertEqual(budget, 64)
            if calls == 1:
                return "", "eos", 1
            return "PRIVATE repeat repeat repeat repeat repeat repeat repeat", "max_tokens", budget
        output = io.StringIO()
        with patch("phoenix_real_smoke.capture_generation", side_effect=fake_capture):
            with redirect_stdout(output):
                result = smoke(None, None, 1, 2, None, 64)
        self.assertEqual(result["cases"], 8)
        self.assertEqual(result["actual_eos"], 1)
        self.assertEqual(result["nonempty"], 7)
        self.assertEqual(result["eos_rate"], 0.125)
        self.assertGreater(result["repetition_rate"], 0)
        self.assertFalse(result["quality_evaluated"])
        self.assertFalse(result["nis2_evaluated"])
        self.assertEqual(result["generated_tokens_total"], 1 + 7 * 64)
        self.assertNotIn("PRIVATE", output.getvalue())
        self.assertNotIn("repeat", output.getvalue())

    def test_unreached_max_budget_claim_rejected(self):
        with patch("phoenix_real_smoke.capture_generation", return_value=("text", "max_tokens", 31)):
            with redirect_stdout(io.StringIO()), self.assertRaises(RuntimeError):
                smoke(None, None, 1, 2, None, 32)

    def test_invalid_budget_rejected_before_decoding(self):
        with patch("phoenix_real_smoke.capture_generation") as capture:
            with self.assertRaises(ValueError):
                smoke(None, None, 1, 2, None, 256)
            capture.assert_not_called()


if __name__ == "__main__":
    unittest.main()
