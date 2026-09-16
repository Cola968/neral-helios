"""Synthetic configuration tests only; not evidence of GPU speed or model quality."""
import unittest

from probe import MAX_BATCH_TOKENS, SYNTHETIC_TARGET_TPS, validate_shape


class ProbeShapeTests(unittest.TestCase):
    def test_safe_batch8_batch12_batch16_shapes(self):
        for batch in (8, 12, 16):
            with self.subTest(batch=batch):
                self.assertIsNone(validate_shape(batch, 512, 4))

    def test_8192_ceiling_enforced(self):
        self.assertEqual(MAX_BATCH_TOKENS, 8192)
        self.assertIsNone(validate_shape(8, 1024, 4))
        for batch in (12, 16):
            with self.subTest(batch=batch):
                with self.assertRaisesRegex(ValueError, "ceiling"):
                    validate_shape(batch, 1024, 4)

    def test_large_batch_must_freeze_all_but_four_blocks(self):
        for blocks in (8, 24):
            with self.subTest(blocks=blocks):
                with self.assertRaisesRegex(ValueError, "four trainable"):
                    validate_shape(12, 512, blocks)

    def test_invalid_shapes_rejected(self):
        for args in ((0, 512, 4), (10, 512, 4), (16, 128, 4), (8, 512, 5)):
            with self.subTest(args=args):
                with self.assertRaises(ValueError):
                    validate_shape(*args)

    def test_50k_is_synthetic_only_target_constant(self):
        self.assertEqual(SYNTHETIC_TARGET_TPS, 50_000)


if __name__ == "__main__":
    unittest.main()
