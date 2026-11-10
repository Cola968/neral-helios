"""Synthetic guarded diagnostic contract tests; no model/GPU/files."""
import unittest
from unittest.mock import patch

import helios_guarded_diagnostic as diagnostic


class GuardedDiagnosticTests(unittest.TestCase):
    def test_public_fixture_health_stays_non_approving(self):
        generated = [("Hello there.", 2, "eos"), ("", 1, "role_boundary"),
                     ("one two three " * 6, 20, "repetition_guard"),
                     ("Answer", 32, "max_tokens")] * 2
        with patch.object(diagnostic, "guarded_generate", side_effect=generated) as mock:
            result = diagnostic.evaluate(None, None, 1, 2, None, 32)
        self.assertEqual(mock.call_count, 8)
        self.assertEqual(result["cases"], 8)
        self.assertEqual(result["finish_reasons"], {
            "eos": 2, "max_tokens": 2, "repetition_guard": 2, "role_boundary": 2})
        self.assertFalse(result["correctness_evaluated"])
        self.assertFalse(result["raw_decoder_comparable"])
        self.assertFalse(result["checkpoint_modified"])
        self.assertFalse(result["training_started"])
        self.assertFalse(result["private_text_logged"])

    def test_invalid_counts_and_stops_cannot_pass(self):
        with self.assertRaises(ValueError):
            diagnostic.evaluate(None, None, 1, 2, None, 1)
        for response in (("oops", 0, "eos"), ("oops", 1, "invented_stop"),
                         ("oops", 4, "max_tokens")):
            with self.subTest(response=response):
                with patch.object(diagnostic, "guarded_generate", return_value=response):
                    with self.assertRaises(RuntimeError):
                        diagnostic.evaluate(None, None, 1, 2, None, 32)


if __name__ == '__main__':
    unittest.main()
