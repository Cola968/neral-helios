"""Safety checks for a real, immutable Arctus 350M checkpoint mini-inference."""
from __future__ import annotations
import unittest

from frontier350.roblox_checkpoint_probe350 import (
    probe_is_allowed, checkpoint_class, normalize_prompt_set,
)

class CheckpointSafetyTests(unittest.TestCase):
    def test_real_2b_must_be_confirmed_by_metadata_not_filename(self):
        self.assertEqual(checkpoint_class(2_073_000_000), "2b_plus")
        self.assertEqual(checkpoint_class(1_010_003_584), "1b_baseline")
        self.assertEqual(checkpoint_class(0), "unknown")
        self.assertEqual(checkpoint_class(150_000_000), "pre_1b")

    def test_malformed_processed_tokens_fail_closed(self):
        for v in (None, "2B", -2, 3.3, True):
            with self.subTest(v=v), self.assertRaises(ValueError):
                checkpoint_class(v)

    def test_gpu_policy_requires_explicit_allowance_and_headroom(self):
        self.assertFalse(probe_is_allowed(False, True, 8000, 7000))
        self.assertFalse(probe_is_allowed(True, False, 8000, 7000))
        self.assertFalse(probe_is_allowed(True, True, 4095, 7000))
        self.assertFalse(probe_is_allowed(True, True, 8000, 5999))
        self.assertTrue(probe_is_allowed(True, True, 7200, 7000))
        for a in ((True, True, -1, 7000), (True, True, 8000, 0), (1, True, 8000, 7000)):
            with self.subTest(a=a), self.assertRaises(ValueError):
                probe_is_allowed(*a)

    def test_only_short_non_heldout_probes_allowed(self):
        rows=normalize_prompt_set(["Hallo!", "Was ist zwei plus zwei?", "Nenne eine Farbe."])
        self.assertEqual(len(rows),3)
        with self.assertRaises(ValueError): normalize_prompt_set(["a" * 1000])
        with self.assertRaises(ValueError): normalize_prompt_set(["a"]*6)
        with self.assertRaises(ValueError): normalize_prompt_set(["", "Hallo"])

if __name__ == "__main__":
    unittest.main()
