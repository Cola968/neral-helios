"""Tiny CPU precision regression; NOT a Neral quality/throughput benchmark."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phoenix_master_precision import require_fp32_master


class MasterPrecisionTests(unittest.TestCase):
    def test_real_tiny_adam_updates_fp32_master_at_small_lr(self):
        import torch
        reference = 0.1
        results = {}
        for dtype in (torch.bfloat16, torch.float32):
            model = torch.nn.Linear(1, 1, bias=False, dtype=dtype)
            with torch.no_grad():
                model.weight.fill_(reference)
            optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5,
                                           weight_decay=0., foreach=False)
            if dtype == torch.float32:
                require_fp32_master(model, optimizer, torch, require_moments=False)
            else:
                with self.assertRaisesRegex(RuntimeError, 'FP32 master'):
                    require_fp32_master(model, optimizer, torch, require_moments=False)
            for _ in range(10):
                optimizer.zero_grad(set_to_none=True)
                model.weight.grad = torch.full_like(model.weight, 0.01)
                optimizer.step()
            results[dtype] = model.weight.item()
            if dtype == torch.float32:
                require_fp32_master(model, optimizer, torch, require_moments=True)
                self.assertEqual(optimizer.state[model.weight]['exp_avg'].dtype, torch.float32)
            else:
                with self.assertRaisesRegex(RuntimeError, 'FP32 master'):
                    require_fp32_master(model, optimizer, torch, require_moments=True)
        self.assertLess(results[torch.float32], reference - 0.00015)
        self.assertGreater(results[torch.bfloat16], reference - 0.00005)

    def test_missing_or_wrong_dtype_moments_fail_closed(self):
        import torch
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
        with self.assertRaisesRegex(RuntimeError, 'moments missing'):
            require_fp32_master(model, optimizer, torch, require_moments=True)
        optimizer.zero_grad(set_to_none=True)
        model(torch.ones(1, 1)).sum().backward()
        optimizer.step()
        require_fp32_master(model, optimizer, torch, require_moments=True)
        optimizer.state[model.weight]['exp_avg'] = optimizer.state[model.weight]['exp_avg'].bfloat16()
        with self.assertRaisesRegex(RuntimeError, 'moments must be FP32'):
            require_fp32_master(model, optimizer, torch, require_moments=True)


if __name__ == '__main__':
    unittest.main()
