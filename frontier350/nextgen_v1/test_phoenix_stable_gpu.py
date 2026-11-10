"""Synthetic CPU-only GPU readiness contracts; no device access or weights."""
from pathlib import Path
from unittest.mock import patch
import unittest

from phoenix_cuda_diagnostic import (
    IDLE_INTERVAL_SECONDS, IDLE_SAMPLES, main, require_stably_idle_gpu,
)


class StableGPUPreflightTests(unittest.TestCase):
    def test_three_separate_safe_snapshots_required(self):
        with patch('phoenix_cuda_diagnostic.require_idle_gpu') as guard, \
             patch('phoenix_cuda_diagnostic.time.sleep') as sleep:
            require_stably_idle_gpu()
        self.assertEqual(IDLE_SAMPLES, 3)
        self.assertEqual(guard.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertTrue(all(call.args == (IDLE_INTERVAL_SECONDS,)
                            for call in sleep.call_args_list))

    def test_any_sample_failure_stops_without_majority_or_averaging(self):
        for blocked_index in range(3):
            with self.subTest(blocked_index=blocked_index):
                responses = [None] * blocked_index + [RuntimeError('busy/hot')]
                with patch('phoenix_cuda_diagnostic.require_idle_gpu',
                           side_effect=responses) as guard, \
                     patch('phoenix_cuda_diagnostic.time.sleep') as sleep:
                    with self.assertRaisesRegex(RuntimeError,
                                                f'sample {blocked_index + 1}/3'):
                        require_stably_idle_gpu()
                self.assertEqual(guard.call_count, blocked_index + 1)
                self.assertEqual(sleep.call_count, blocked_index)

    def test_telemetry_errors_also_fail_closed(self):
        with patch('phoenix_cuda_diagnostic.require_idle_gpu',
                   side_effect=FileNotFoundError('nvidia-smi')) as guard, \
             patch('phoenix_cuda_diagnostic.time.sleep') as sleep:
            with self.assertRaisesRegex(RuntimeError, 'no CUDA inference started'):
                require_stably_idle_gpu()
        guard.assert_called_once_with()
        sleep.assert_not_called()

    def test_no_checkpoint_loading_or_torch_import_when_preflight_blocks(self):
        with patch('sys.argv', ['phoenix_cuda_diagnostic', '--execute',
                                '--assets-root', str(Path('.'))]), \
             patch('phoenix_cuda_diagnostic.require_stably_idle_gpu',
                   side_effect=RuntimeError('GPU not idle')) as guard, \
             patch('phoenix_cuda_diagnostic.load_verified_model') as load:
            with self.assertRaisesRegex(RuntimeError, 'GPU not idle'):
                main()
        guard.assert_called_once_with()
        load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
