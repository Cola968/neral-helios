"""Regression tests: integration web/UI gate follows main and fails closed."""
import subprocess
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scripts import neral_1_1_main_cpu_gate as gate

BASE = "a" * 40
CLEAN_BLOB = "b" * 40


class MainMergeBaseIntegrityTests(unittest.TestCase):
    def test_base_is_actual_merge_base_with_tracked_main_not_a_pinned_commit(self):
        with patch.object(gate.subprocess, "run",
                          return_value=SimpleNamespace(stdout=BASE+"\n")) as invoke:
            self.assertEqual(gate._trusted_main_base(), BASE)
        args = invoke.call_args.args[0]
        self.assertEqual(args, ["git", "merge-base", "HEAD",
                                "refs/remotes/origin/main"])
        self.assertTrue(invoke.call_args.kwargs["check"])

    def test_missing_or_malformed_base_is_rejected(self):
        for invalid in ("", "1234", "not-a-sha", "g"*40, "a"*39):
            with self.subTest(base=invalid), patch.object(
                    gate.subprocess, "run",
                    return_value=SimpleNamespace(stdout=invalid+"\n")):
                with self.assertRaises(RuntimeError):
                    gate._trusted_main_base()

    def test_unreachable_tracked_main_fails_closed(self):
        with patch.object(gate.subprocess, "run",
                          side_effect=subprocess.CalledProcessError(128,"git")):
            with self.assertRaises(subprocess.CalledProcessError):
                gate._trusted_main_base()

    def test_changed_protected_blob_is_still_rejected(self):
        def run(args, **kwargs):
            if args[1] == "merge-base":
                return SimpleNamespace(stdout=BASE+"\n")
            if args[1] == "rev-parse":
                return SimpleNamespace(stdout=(CLEAN_BLOB if args[2].startswith(BASE)
                                               else "c"*40)+"\n")
            raise AssertionError("Unreachable after mismatch")
        with patch.object(gate.subprocess, "run", side_effect=run):
            with self.assertRaisesRegex(RuntimeError, "CURRENT_MAIN_WEB_OR_UI_CHANGED"):
                gate.verify_main_integrity()

    def test_unmodified_protected_files_keep_original_worktree_guard(self):
        commands = []
        def run(args, **kwargs):
            commands.append(args)
            if args[1] == "merge-base":
                return SimpleNamespace(stdout=BASE+"\n")
            if args[1] == "rev-parse":
                return SimpleNamespace(stdout=CLEAN_BLOB+"\n")
            if args[1] == "diff":
                return SimpleNamespace(returncode=0)
            raise AssertionError(args)
        with patch.object(gate.subprocess, "run", side_effect=run),patch("builtins.print"):
            gate.verify_main_integrity()
        self.assertEqual(len([x for x in commands if x[1] == "diff"]),len(gate.PROTECTED))
        self.assertTrue(all(x[2] == "--quiet" for x in commands if x[1] == "diff"))


if __name__ == "__main__":
    unittest.main()
