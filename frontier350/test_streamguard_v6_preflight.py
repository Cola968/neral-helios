from __future__ import annotations

import unittest

from frontier350.streamguard_v6_policy import (
    MemoryWatchdog,
    RetentionController,
    choose_profile,
    downgrade_profile,
    effective_tps,
    scaled_recipe,
)


class StreamGuardPolicyTests(unittest.TestCase):
    def test_profile_selection_and_downgrade(self):
        self.assertEqual(choose_profile(free_mib=4096, reserved_mib=7000).name, "P48")
        self.assertEqual(choose_profile(free_mib=1800, reserved_mib=7900).name, "P32")
        self.assertEqual(choose_profile(free_mib=900, reserved_mib=9000).name, "P24")
        self.assertEqual(downgrade_profile("P48").name, "P32")
        self.assertEqual(downgrade_profile("P32").name, "P24")
        self.assertEqual(downgrade_profile("P24").name, "P24")
        self.assertNotEqual(choose_profile(free_mib=4096, reserved_mib=7000, previous="P32").name, "P48")

    def test_memory_watchdog(self):
        w = MemoryWatchdog(baseline_reserved_mib=7000)
        self.assertEqual(w.needs_restart(reserved_mib=7200, free_mib=3000, segment_tokens=10_000_000), (False, None))
        self.assertEqual(w.needs_restart(reserved_mib=7800, free_mib=3000, segment_tokens=10_000_000)[1], "reserved_vram_growth")
        self.assertEqual(w.needs_restart(reserved_mib=7200, free_mib=1200, segment_tokens=10_000_000)[1], "free_vram_floor")
        self.assertEqual(w.needs_restart(reserved_mib=7200, free_mib=3000, segment_tokens=50_000_000)[1], "segment_token_limit")
        w.oom_retries = 2
        self.assertEqual(w.needs_restart(reserved_mib=7200, free_mib=3000, segment_tokens=10_000_000)[1], "repeated_oom_retry")

    def test_retention_hysteresis(self):
        c = RetentionController(pressure=2)
        for _ in range(2):
            self.assertEqual(c.observe(healthy=True, validation_delta=0.0, regressions=0, gains=0, margin_delta=0.0), 2)
        self.assertEqual(c.observe(healthy=True, validation_delta=0.0, regressions=0, gains=0, margin_delta=0.0), 1)
        self.assertEqual(c.observe(healthy=False, validation_delta=0.1, regressions=2, gains=0, margin_delta=-0.02), 2)
        self.assertEqual(c.observe(healthy=False, validation_delta=0.1, regressions=2, gains=0, margin_delta=-0.02), 3)

    def test_recipe_and_effective_tps(self):
        r = scaled_recipe(2.6e-5, 1.0e-5, 1.5, 2)
        self.assertLess(r["lr"], 2.6e-5)
        self.assertGreater(r["replay_boost"], 1.5)
        self.assertAlmostEqual(effective_tps(45_000_000, 1000.0), 45_000.0)


class ChunkedCEFidelityTests(unittest.TestCase):
    def test_chunked_ce_matches_reference_on_cpu(self):
        import torch
        from frontier350.v6_chunked_frozen_head_ce import reference_equivalence

        torch.manual_seed(20260914)
        hidden = torch.randn(3, 5, 24, dtype=torch.float32)
        weight = torch.randn(37, 24, dtype=torch.float32)
        weight.requires_grad_(False)
        targets = torch.randint(0, 37, (3, 5), dtype=torch.long)
        targets[0, 0] = -100
        targets[2, 4] = -100
        result = reference_equivalence(hidden, weight, targets, chunk_size=7, ignore_index=-100)
        self.assertLessEqual(result["loss_abs_diff"], 2e-4)
        self.assertGreaterEqual(result["grad_cosine"], 0.99999)
        self.assertLessEqual(result["grad_rel_norm_diff"], 5e-4)


if __name__ == "__main__":
    unittest.main()
