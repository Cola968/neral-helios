"""Synthetic CPU-only storage tests; no real checkpoint, CUDA or cleanup."""
from __future__ import annotations

from collections import namedtuple
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from storage_budget import GIB, check_storage_budget, physical_directory_bytes

Disk = namedtuple('Disk', 'total used free')


class StorageBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.resume = self.root / 'resume.pt'
        self.resume.write_bytes(b'original-checkpoint')

    def check(self, free=500 * GIB):
        with patch('storage_budget.shutil.disk_usage', return_value=Disk(700 * GIB, 700 * GIB - free, free)):
            return check_storage_budget(self.root, self.resume)

    def test_allows_small_run_with_reserved_space(self):
        report = self.check()
        self.assertEqual(report['directory_bytes'], len(b'original-checkpoint'))
        self.assertGreater(report['reserved_write_bytes'], GIB)
        self.assertEqual(report['directory_limit_bytes'], 40 * GIB)

    def test_counts_hardlinked_generations_only_once(self):
        linked = self.root / 'ckpt-old.pt'
        os.link(self.resume, linked)
        self.assertEqual(physical_directory_bytes(self.root), self.resume.stat().st_size)

    def test_refuses_low_free_space_before_writes(self):
        with self.assertRaisesRegex(RuntimeError, '200 GiB'):
            self.check(free=199 * GIB)
        self.assertEqual(self.resume.read_bytes(), b'original-checkpoint')

    def test_refuses_over_budget_before_writes(self):
        with patch('storage_budget.physical_directory_bytes', return_value=39 * GIB), \
             patch('storage_budget.shutil.disk_usage', return_value=Disk(700 * GIB, 100 * GIB, 600 * GIB)):
            with self.assertRaisesRegex(RuntimeError, '40 GiB'):
                check_storage_budget(self.root, self.resume)
        self.assertEqual(self.resume.read_bytes(), b'original-checkpoint')

    def test_refuses_relaxed_limits(self):
        with self.assertRaisesRegex(ValueError, '8..40'):
            check_storage_budget(self.root, self.resume, max_directory_bytes=41 * GIB)
        with self.assertRaisesRegex(ValueError, '200 GiB'):
            check_storage_budget(self.root, self.resume, min_free_bytes=100 * GIB)

    def test_refuses_resume_outside_its_directory(self):
        with self.assertRaisesRegex(ValueError, 'directly'):
            check_storage_budget(self.root, self.root.parent / 'resume.pt')

    def test_refuses_symlinks_inside_run_and_does_not_delete(self):
        linked = self.root / 'other.pt'
        try:
            linked.symlink_to(self.resume)
        except (OSError, NotImplementedError):
            self.skipTest('OS denies symlink creation')
        with self.assertRaisesRegex(ValueError, 'symbolic links'):
            self.check()
        self.assertTrue(linked.is_symlink())
        self.assertEqual(self.resume.read_bytes(), b'original-checkpoint')


if __name__ == '__main__':
    unittest.main()
