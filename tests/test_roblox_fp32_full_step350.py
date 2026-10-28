"""Strict CPU-only admission tests for full-parameter FP32 AdamW GPU smoke."""
import unittest
from frontier350.roblox_fp32_full_step350 import admit_full_step
class FullPrecisionAdmission(unittest.TestCase):
    def test_does_not_start_without_9_gib_headroom(self):
        self.assertTrue(admit_full_step(True,10000,10000,.50))
        self.assertFalse(admit_full_step(False,10000,10000,.50))
        self.assertFalse(admit_full_step(True,9215,11000,.50))
        self.assertFalse(admit_full_step(True,11000,9215,.50))
    def test_disallow_unsafe_or_invalid_allocator_caps(self):
        for frac in (0.,.60,float("nan"),"0.5"):
            with self.subTest(frac=frac),self.assertRaises(ValueError):
                admit_full_step(True,11000,11000,frac)
        with self.assertRaises(ValueError):admit_full_step(True,-1,11000,.5)
        with self.assertRaises(ValueError):admit_full_step(1,11000,11000,.5)
if __name__=="__main__":unittest.main()
