"""Synthetic CPU-only thermal safety contracts; no GPU, checkpoint or training."""
from contextlib import redirect_stdout
import io
from unittest.mock import patch
import unittest

from phoenix_cuda_diagnostic import MAX_TEMPERATURE_C, require_safe_runtime_temperature
from phoenix_real_smoke import SMOKE_PROMPTS, smoke


class RuntimeTemperatureContracts(unittest.TestCase):
    def test_temperature_limit_is_unchanged_and_inclusive(self):
        self.assertEqual(MAX_TEMPERATURE_C, 78)
        with patch('phoenix_cuda_diagnostic.subprocess.check_output', return_value='78\n'):
            self.assertIsNone(require_safe_runtime_temperature())

    def test_above_limit_refuses_even_at_one_degree(self):
        with patch('phoenix_cuda_diagnostic.subprocess.check_output', return_value='79\n'):
            with self.assertRaisesRegex(RuntimeError, 'safety limit'):
                require_safe_runtime_temperature()

    def test_missing_or_bad_temperature_fails_closed(self):
        for text in ('', 'N/A', '65\n66\n', '-1', '121'):
            with self.subTest(text=text):
                with patch('phoenix_cuda_diagnostic.subprocess.check_output',
                           return_value=text):
                    with self.assertRaises(RuntimeError):
                        require_safe_runtime_temperature()
        with patch('phoenix_cuda_diagnostic.subprocess.check_output',
                   side_effect=FileNotFoundError('nvidia-smi')):
            with self.assertRaises(FileNotFoundError):
                require_safe_runtime_temperature()

    def test_guard_runs_before_and_after_each_bounded_prompt(self):
        with patch('phoenix_real_smoke.capture_generation',
                   return_value=('synthetic', 'max_tokens', 1)) as capture, \
             patch('phoenix_cuda_diagnostic.require_safe_runtime_temperature') as monitor, \
             redirect_stdout(io.StringIO()):
            result = smoke(None, None, 1, 2, None, budget=1,
                           before_case=monitor, after_case=monitor)
        self.assertEqual(result['cases'], len(SMOKE_PROMPTS))
        self.assertEqual(capture.call_count, len(SMOKE_PROMPTS))
        self.assertEqual(monitor.call_count, 2 * len(SMOKE_PROMPTS))
        self.assertFalse(result['training_started'])

    def test_hot_before_or_after_case_stops_further_generation(self):
        for fail_when in ('before', 'after'):
            with self.subTest(fail_when=fail_when):
                def fail():
                    raise RuntimeError('hot')
                with patch('phoenix_real_smoke.capture_generation',
                           return_value=('secret synthetic', 'max_tokens', 1)) as capture, \
                     redirect_stdout(io.StringIO()) as output:
                    with self.assertRaisesRegex(RuntimeError, 'hot'):
                        smoke(None, None, 1, 2, None, budget=1,
                              before_case=fail if fail_when == 'before' else None,
                              after_case=fail if fail_when == 'after' else None)
                self.assertEqual(capture.call_count, 0 if fail_when == 'before' else 1)
                self.assertNotIn('secret synthetic', output.getvalue())


if __name__ == '__main__':
    unittest.main()
