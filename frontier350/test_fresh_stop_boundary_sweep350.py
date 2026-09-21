"""Deterministic exhaustive small-grid native stop-boundary verification; CPU only."""
from __future__ import annotations
import unittest
from fresh_stop_boundary350 import require_fresh_stop_boundary


class FreshStopSweepTests(unittest.TestCase):
    def test_intermediate_and_terminal_stop_grid(self):
        tested = accepted = refused = 0
        for seq in (1, 2, 4, 128):
            for microbatch in (1, 2):
                for grad_accum in (1, 4, 16):
                    update = seq * microbatch * grad_accum
                    for target in (1, 2, 511, 512, 513, 4096, 4100, 10000):
                        for stop in range(1, target + 1):
                            should_pass = stop == target or stop % update == 0
                            tested += 1
                            try:
                                actual_update = require_fresh_stop_boundary(
                                    target=target, stop=stop, seq=seq,
                                    microbatch=microbatch, grad_accum=grad_accum)
                            except ValueError:
                                self.assertFalse(should_pass, (seq, microbatch,
                                                               grad_accum, target, stop))
                                refused += 1
                            else:
                                self.assertTrue(should_pass, (seq, microbatch,
                                                              grad_accum, target, stop))
                                self.assertEqual(actual_update, update)
                                accepted += 1
        self.assertEqual((tested, accepted, refused), (473640, 68384, 405256))


if __name__ == '__main__':
    unittest.main()
