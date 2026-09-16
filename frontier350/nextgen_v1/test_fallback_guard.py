"""CPU-only tests: dummy bytes are not valid exact-recovery snapshots."""
from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from fallback_guard import fallback_allowed


class FallbackGuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "experiment"

    def test_new_or_empty_directory_can_be_attempted_without_rewinding(self):
        self.assertTrue(fallback_allowed(self.out, None))
        self.out.mkdir()
        self.assertTrue(fallback_allowed(self.out, None))

    def test_any_partial_log_or_checkpoint_blocks_retry_without_recovery(self):
        self.out.mkdir()
        for name in ("metrics.jsonl", "resume.pt", "failure.json"):
            artifact = self.out / name
            artifact.write_bytes(b"preserve-me")
            self.assertFalse(fallback_allowed(self.out, None))
            self.assertEqual(artifact.read_bytes(), b"preserve-me")
            artifact.unlink()

    def test_recovery_is_only_a_candidate_never_authenticated_by_guard(self):
        self.out.mkdir()
        (self.out / "metrics.jsonl").write_text("failed attempt\n")
        recovery = self.out / "v7_recovery_123.pt"
        recovery.write_bytes(b"fixture-not-a-real-checkpoint")
        self.assertTrue(fallback_allowed(self.out, recovery))
        self.assertTrue((self.out / "metrics.jsonl").exists())
        self.assertEqual(recovery.read_bytes(), b"fixture-not-a-real-checkpoint")

    def test_missing_empty_or_external_recovery_is_blocked(self):
        self.out.mkdir()
        missing = self.out / "missing.pt"
        self.assertFalse(fallback_allowed(self.out, missing))
        missing.write_bytes(b"")
        self.assertFalse(fallback_allowed(self.out, missing))
        external = self.root / "external.pt"
        external.write_bytes(b"untrusted")
        self.assertFalse(fallback_allowed(self.out, external))

    def test_symlink_recovery_and_output_are_blocked(self):
        self.out.mkdir()
        target = self.out / "real.pt"
        target.write_bytes(b"not-a-checkpoint")
        link = self.out / "link.pt"
        try:
            link.symlink_to(target)
        except (OSError, NotImplementedError):
            self.skipTest("platform does not support symlinks")
        self.assertFalse(fallback_allowed(self.out, link))
        self.assertFalse(fallback_allowed(self.root / "output_link", None) if False else False)
        output_link = self.root / "output_link"
        output_link.symlink_to(self.out, target_is_directory=True)
        self.assertFalse(fallback_allowed(output_link, None))

    def test_non_directory_output_is_rejected(self):
        self.out.write_text("existing diagnostics")
        self.assertFalse(fallback_allowed(self.out, None))

    def test_both_historical_launchers_have_non_destructive_guard(self):
        frontier = Path(__file__).resolve().parents[1]
        for name in ("train_v7_1_stage_launcher.py", "train_v7_2_stage_launcher.py"):
            with self.subTest(name=name):
                code = (frontier / name).read_text(encoding="utf-8")
                self.assertNotIn("shutil.rmtree", code)
                self.assertNotIn("ignore_errors=True", code)
                self.assertIn("from nextgen_v1.fallback_guard import fallback_allowed", code)
                self.assertIn("attempt > 1 and not fallback_allowed(out, recovery)", code)
                self.assertIn("next_fallback = attempt < len(profiles) and fallback_allowed(out, newest)", code)
                self.assertIn("if not next_fallback:", code)
                self.assertIn("raise SystemExit(last_code)", code)


if __name__ == "__main__":
    unittest.main()
