"""Synthetic, no-Torch tests: entry point must never skip stable GPU preflight."""
from __future__ import annotations

import sys
import unittest
from unittest.mock import patch

import train_sft_guarded_entry as entry


class GuardedEntryTests(unittest.TestCase):
    def test_no_execute_refuses_without_gpu_probe(self):
        with patch.object(sys, "argv", ["entry", "--trusted-local-checkpoint"]), \
             patch.object(entry.gpu_safety, "require_stably_idle_gpu") as check, \
             patch.object(entry.pilot, "main") as train:
            with self.assertRaises(SystemExit):
                entry.main()
            check.assert_not_called()
            train.assert_not_called()

    def test_no_trust_refuses_without_gpu_probe(self):
        with patch.object(sys, "argv", ["entry", "--execute"]), \
             patch.object(entry.gpu_safety, "require_stably_idle_gpu") as check, \
             patch.object(entry.pilot, "main") as train:
            with self.assertRaises(SystemExit):
                entry.main()
            check.assert_not_called()
            train.assert_not_called()

    def test_busy_or_hot_gpu_fails_closed_before_training(self):
        with patch.object(sys, "argv", ["entry", "--execute", "--trusted-local-checkpoint"]), \
             patch.object(entry.gpu_safety, "require_stably_idle_gpu", side_effect=RuntimeError("GPU busy")) as check, \
             patch.object(entry.pilot, "main") as train:
            with self.assertRaisesRegex(RuntimeError, "GPU busy"):
                entry.main()
            check.assert_called_once_with()
            train.assert_not_called()

    def test_only_successful_preflight_can_call_existing_pilot(self):
        with patch.object(sys, "argv", ["entry", "--execute", "--trusted-local-checkpoint"]), \
             patch.object(entry.gpu_safety, "require_stably_idle_gpu") as check, \
             patch.object(entry.pilot, "main", return_value=0) as train:
            self.assertEqual(entry.main(), 0)
            check.assert_called_once_with()
            train.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
