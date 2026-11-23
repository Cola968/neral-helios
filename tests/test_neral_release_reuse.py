"""Regression tests for safe CI release reuse after duplicate-tag reruns."""
import hashlib
import tempfile
import unittest
from pathlib import Path
from scripts.neral_release_reuse import verify_existing_release

SHA = "9c2fd1a18282aa2a14e7bf10ea9798299508e173"
TAG = "neral-preview-" + SHA[:12]


class ExistingReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.local, self.remote = self.root / "local", self.root / "remote"
        self.local.mkdir()
        self.remote.mkdir()
        blob = b"NERAL release fixture, not a real model."
        digest = hashlib.sha256(blob).hexdigest()
        for folder in (self.local, self.remote):
            (folder / "NERAL-1.0-Local-Preview.zip").write_bytes(blob)
            (folder / "SHA256SUMS.txt").write_text(
                digest + "  NERAL-1.0-Local-Preview.zip\n", encoding="utf8")
            (folder / "SOURCE-COMMIT.txt").write_text(SHA + "\n", encoding="utf8")
        self.meta = {"tagName": TAG, "targetCommitish": SHA,
                     "isPrerelease": True, "isDraft": False}

    def test_exact_reuse_succeeds(self):
        self.assertTrue(verify_existing_release(self.local, self.remote, self.meta, SHA))

    def test_different_published_zip_is_rejected(self):
        (self.remote / "NERAL-1.0-Local-Preview.zip").write_bytes(b"changed")
        with self.assertRaises(ValueError):
            verify_existing_release(self.local, self.remote, self.meta, SHA)

    def test_wrong_release_metadata_is_rejected(self):
        for bad in ({**self.meta, "targetCommitish": "main"},
                    {**self.meta, "isDraft": True},
                    {**self.meta, "isPrerelease": False},
                    {**self.meta, "tagName": "different-tag"}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                verify_existing_release(self.local, self.remote, bad, SHA)

    def test_missing_asset_or_invalid_checksum_is_rejected(self):
        (self.remote / "SHA256SUMS.txt").unlink()
        with self.assertRaises(ValueError):
            verify_existing_release(self.local, self.remote, self.meta, SHA)
        (self.remote / "SHA256SUMS.txt").write_text("bad\n", encoding="utf8")
        with self.assertRaises(ValueError):
            verify_existing_release(self.local, self.remote, self.meta, SHA)


if __name__ == "__main__":
    unittest.main()
