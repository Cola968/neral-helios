"""Fail-closed CPU tests for ultra-small concurrent GPU diagnostic with Roblox running."""
from __future__ import annotations

import unittest

from frontier350.gpu_precision_probe350 import assess_free_vram, parse_gpu_memory_csv


class ConcurrentGpuSafetyTests(unittest.TestCase):
    def test_accepts_free_memory_above_threshold(self):
        total, free = parse_gpu_memory_csv("12282, 8240\n")
        self.assertEqual((total, free), (12282, 8240))
        self.assertEqual(assess_free_vram(total, free, min_free_mib=4096),
                         (True, "admitted"))

    def test_refuses_when_roblox_uses_too_much_vram(self):
        self.assertEqual(assess_free_vram(12282, 3000, min_free_mib=4096),
                         (False, "insufficient_free_vram"))

    def test_malformed_or_unknown_nvidia_smi_fails_closed(self):
        for text in ("N/A, N/A", "garbage", "", "12282", "12282, 20000",
                     "12282, -1", "12282, 4.5", "12282, 8000\n12282, 2000"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_gpu_memory_csv(text)

    def test_free_memory_policy_rejects_malformed_numbers(self):
        for values in ((True, 6000), (10000, 10001), (10000, -1),
                       ("10000", 6000), (0, 0), (10000, 6000, -1)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                assess_free_vram(*values[:2], min_free_mib=values[2] if len(values)>2 else 4096)

    def test_probe_has_explicit_synthetic_not_training_identity(self):
        from frontier350.gpu_precision_probe350 import PROBE_NAME
        self.assertEqual(PROBE_NAME, "synthetic_cuda_optimizer_precision_only")


if __name__ == "__main__":
    unittest.main()
