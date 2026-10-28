"""CPU-only safety tests before native RTX4070 FP32 norm-gradient micro-probe."""
import unittest
from frontier350.roblox_fp32_norm_probe350 import allow_norm_probe, summarize_update
class NormPrecisionGuard(unittest.TestCase):
    def test_admit_only_with_vram_headroom(self):
        self.assertTrue(allow_norm_probe(True,10240,11000,0.38))
        self.assertFalse(allow_norm_probe(False,10240,11000,0.38))
        self.assertFalse(allow_norm_probe(True,8100,11000,0.38))
        self.assertFalse(allow_norm_probe(True,10240,8100,0.38))
        self.assertFalse(allow_norm_probe(True,10240,11000,0.50))
        with self.assertRaises(ValueError):allow_norm_probe(True,-1,9000,0.38)
        with self.assertRaises(ValueError):allow_norm_probe(1,10000,10000,0.38)
        with self.assertRaises(ValueError):allow_norm_probe(True,10000,10000,float("nan"))

    def test_changed_fraction_and_norm_are_finite(self):
        r=summarize_update([1.,1.,1.],[1.00001,1.,0.99999])
        self.assertEqual(r["changed_fraction"],2/3)
        self.assertGreater(r["mean_abs_change"],0)
        for before,after in (([],[]),([0],[]),([1.],[float("nan")]),([1.,2.],[1.])):
            with self.subTest(before=before,after=after),self.assertRaises(ValueError):
                summarize_update(before,after)

if __name__=="__main__":unittest.main()
