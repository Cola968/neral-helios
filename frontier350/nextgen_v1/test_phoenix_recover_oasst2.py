"""Offline regression tests; never contacts Hugging Face or user's PC."""
from __future__ import annotations

import hashlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phoenix_recover_oasst2 as recovery


class FakeResponse(io.BytesIO):
    def __init__(self, payload: bytes, url: str = recovery.UPSTREAM_URL):
        super().__init__(payload)
        self.url = url

    def geturl(self):
        return self.url


class PinnedRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / "private"
        self.root.mkdir()
        self.payload = b"synthetic-archive-bytes"
        self.length = patch.object(recovery, "EXPECTED_BYTES", len(self.payload))
        self.digest = patch.object(recovery, "UPSTREAM_SHA256", hashlib.sha256(self.payload).hexdigest())
        self.length.start()
        self.digest.start()
        self.addCleanup(self.length.stop)
        self.addCleanup(self.digest.stop)

    def _download(self, data=None, url=None):
        return recovery.recover(
            self.root,
            open_url=lambda request, timeout: FakeResponse(
                self.payload if data is None else data,
                recovery.UPSTREAM_URL if url is None else url,
            ),
        )

    def _target(self):
        return self.root / "sft50_sources" / recovery.ARCHIVE_NAME

    def _assert_no_partials(self):
        self.assertEqual(list((self.root / "sft50_sources").glob("*.partial-*")), [])

    def test_verified_archive_written_only_after_full_digest(self):
        result = self._download()
        self.assertEqual(result["status"], "recovered_verified")
        self.assertFalse(result["dataset_approved"])
        self.assertFalse(result["training_started"])
        self.assertEqual(self._target().read_bytes(), self.payload)
        self._assert_no_partials()

    def test_idempotent_verified_archive_never_downloads_again(self):
        self._download()
        result = recovery.recover(self.root, open_url=lambda *a, **k: self.fail("no second network access"))
        self.assertEqual(result["status"], "already_verified")
        self.assertEqual(self._target().read_bytes(), self.payload)

    def test_wrong_digest_leaves_no_archive_or_partial(self):
        bad = b"X" + self.payload[1:]
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self._download(data=bad)
        self.assertFalse(self._target().exists())
        self._assert_no_partials()

    def test_truncated_response_leaves_no_archive(self):
        with self.assertRaisesRegex(ValueError, "byte length"):
            self._download(data=self.payload[:-1])
        self.assertFalse(self._target().exists())
        self._assert_no_partials()

    def test_oversized_response_is_bounded_and_cleaned(self):
        with self.assertRaisesRegex(ValueError, "exceeds pinned length"):
            self._download(data=self.payload + b"x")
        self.assertFalse(self._target().exists())
        self._assert_no_partials()

    def test_foreign_redirect_is_rejected_without_output(self):
        with self.assertRaisesRegex(ValueError, "redirect"):
            self._download(url="https://example.org/archive")
        self.assertFalse(self._target().exists())
        self._assert_no_partials()

    def test_unverified_existing_archive_is_never_overwritten(self):
        target = self._target()
        target.parent.mkdir()
        target.write_bytes(b"wrong")
        with self.assertRaisesRegex(ValueError, "refusing overwrite"):
            self._download()
        self.assertEqual(target.read_bytes(), b"wrong")
        self._assert_no_partials()

    def test_checkout_root_refused_before_network(self):
        with patch.dict(os.environ, {"GITHUB_WORKSPACE": str(self.root.parent)}):
            with self.assertRaisesRegex(ValueError, "GitHub checkout"):
                self._download()
        self.assertFalse(self._target().exists())

    def test_nonexistent_private_root_is_refused(self):
        with self.assertRaisesRegex(ValueError, "private assets root"):
            recovery.recover(self.root / "missing", open_url=lambda *a, **k: self.fail("unexpected network"))


if __name__ == "__main__":
    unittest.main()
