"""Synthetic CPU tests; never open or regenerate the sealed corpus."""
from __future__ import annotations
import tempfile
from pathlib import Path
import unittest
from scripts.neral_pr542_recovery_readonly import scan_named_raw_files

class OriginalRunRawRecoveryTests(unittest.TestCase):
    def test_finds_nested_original_run_file_beyond_fixed_77_paths(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)
            p=root/"unpredicted"/"nested"/"archive"/"private-results.jsonl"
            p.parent.mkdir(parents=True)
            p.write_text('{"synthetic":true}\n',encoding="utf8")
            found,meta=scan_named_raw_files([root],max_entries=1000,deadline_seconds=5)
            self.assertEqual(found,[p])
            self.assertEqual(meta["candidate_files"],1)

    def test_skips_unrelated_files_and_deduplicates_roots(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)
            (root/"other.jsonl").write_text('{"synthetic":true}\n',encoding="utf8")
            p=root/"private-results.jsonl"
            p.write_text('{"synthetic":true}\n',encoding="utf8")
            found,meta=scan_named_raw_files([root,root],max_entries=1000,deadline_seconds=5)
            self.assertEqual(found,[p])
            self.assertEqual(meta["candidate_files"],1)

    def test_does_not_follow_symlink_directories(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)
            data=root/"actual"
            data.mkdir()
            p=data/"private-results.jsonl"
            p.write_text('{"synthetic":true}\n',encoding="utf8")
            link=root/"linked"
            try:link.symlink_to(data,target_is_directory=True)
            except OSError:self.skipTest("symlinks not available")
            found,meta=scan_named_raw_files([link],max_entries=1000,deadline_seconds=5)
            self.assertEqual(found,[])
            self.assertEqual(meta["candidate_files"],0)

    def test_small_budget_reports_incomplete_not_false_negative(self):
        with tempfile.TemporaryDirectory() as home:
            root=Path(home)
            for k in range(12):
                p=root/str(k)
                p.mkdir()
                (p/"unrelated.txt").write_text("synthetic",encoding="utf8")
            _,meta=scan_named_raw_files([root],max_entries=3,deadline_seconds=5)
            self.assertTrue(meta["incomplete"])
            self.assertGreater(meta["visited_entries"],3)

if __name__=="__main__":
    unittest.main()
