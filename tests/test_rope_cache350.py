from __future__ import annotations

import unittest

import torch

from frontier350.model350 import _ROPE_CACHE, _rope_table, rope


class RoPEAutogradRegressionTests(unittest.TestCase):
    def test_inference_warmup_then_backward(self):
        # Identical sequence to the real runner: inference evaluation builds the
        # global cache, then SFT begins with autograd enabled.
        _ROPE_CACHE.clear()
        device = torch.device("cpu")
        with torch.inference_mode():
            c, s = _rope_table(64, 8, 10000.0, device, torch.float32)
            self.assertFalse(c.is_inference())
            self.assertFalse(s.is_inference())

        q = torch.randn(2, 2, 17, 8, requires_grad=True)
        k = torch.randn(2, 1, 17, 8, requires_grad=True)
        qr, kr = rope(q, k, 10000.0, 64)
        (qr.square().mean() + kr.square().mean()).backward()

        self.assertIsNotNone(q.grad)
        self.assertIsNotNone(k.grad)
        self.assertTrue(torch.isfinite(q.grad).all().item())
        self.assertTrue(torch.isfinite(k.grad).all().item())

    def test_legacy_inference_entry_is_rebuilt(self):
        # Defend against cached inference tensors from earlier module versions.
        _ROPE_CACHE.clear()
        device = torch.device("cpu")
        key = (device.type, device.index, str(torch.float32), 64, 8, 10000.0)
        with torch.inference_mode():
            _ROPE_CACHE[key] = (torch.ones(1, 1, 64, 8), torch.zeros(1, 1, 64, 8))
        c, s = _rope_table(64, 8, 10000.0, device, torch.float32)
        self.assertFalse(c.is_inference())
        self.assertFalse(s.is_inference())


if __name__ == "__main__":
    unittest.main()
