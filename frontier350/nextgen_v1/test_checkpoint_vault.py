"""Small-file CPU-only tests; never read or alter actual Neral checkpoints."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from checkpoint_vault import protect, restore, verify


class VaultTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.src = self.root / 'source' / 'safe.pt'
        self.src.parent.mkdir()
        self.src.write_bytes(b'NERAL_FAKE_CHECKPOINT_DATA' * 64)
        self.sha = hashlib.sha256(self.src.read_bytes()).hexdigest()
        self.vault = self.root / 'vault'
        self.label = 'SAFE-1010003584'

    def saved(self):
        return protect(self.src, self.vault, self.sha, self.label)

    def test_protect_verify_and_original_unchanged(self):
        original = self.src.read_bytes()
        report = self.saved()
        self.assertEqual(report['status'], 'protected_new_copy')
        self.assertEqual(report['sha256'], self.sha)
        self.assertEqual(verify(report['backup'], self.sha)['status'], 'verified')
        self.assertEqual(self.src.read_bytes(), original)
        self.assertFalse((self.vault / (self.label + '-' + self.sha) / 'resume.pt.part').exists())

    def test_repeat_never_overwrites(self):
        first = self.saved()
        second = self.saved()
        self.assertEqual(second['status'], 'verified_existing_no_overwrite')
        self.assertEqual(first['backup'], second['backup'])

    def test_source_sha_mismatch_rejected_before_creating_backup(self):
        with self.assertRaisesRegex(ValueError, 'source does not match'):
            protect(self.src, self.vault, 'f' * 64, self.label)
        self.assertEqual(list(self.vault.iterdir()), [])

    def test_corrupted_replica_rejected(self):
        report = self.saved()
        Path(report['backup'], 'resume.pt').write_bytes(b'corruption')
        with self.assertRaisesRegex(ValueError, 'corrupt or truncated'):
            verify(report['backup'], self.sha)
        with self.assertRaises(ValueError):
            self.saved()
        self.assertTrue(self.src.exists())

    def test_missing_manifest_fail_closed_without_deleting_partial(self):
        self.vault.mkdir()
        partial = self.vault / (self.label + '-' + self.sha)
        partial.mkdir()
        (partial / 'resume.pt.part').write_bytes(b'partial')
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            self.saved()
        self.assertEqual((partial / 'resume.pt.part').read_bytes(), b'partial')

    def test_duplicate_manifest_keys_rejected(self):
        report = self.saved()
        manifest = Path(report['backup'], 'manifest.json')
        manifest.write_text('{"schema":1,"schema":1,"complete":true}', encoding='utf8')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            verify(report['backup'], self.sha)

    def test_restore_only_to_unused_destination(self):
        report = self.saved()
        dest = self.root / 'recovered.pt'
        recovered = restore(report['backup'], self.sha, dest)
        self.assertEqual(recovered['status'], 'restored_no_overwrite')
        self.assertEqual(dest.read_bytes(), self.src.read_bytes())
        with self.assertRaises(FileExistsError):
            restore(report['backup'], self.sha, dest)
        self.assertEqual(dest.read_bytes(), self.src.read_bytes())

    def test_restore_never_overwrites_preexisting_user_file(self):
        report = self.saved()
        dest = self.root / 'important_user_file.pt'
        dest.write_bytes(b'do_not_touch')
        with self.assertRaises(FileExistsError):
            restore(report['backup'], self.sha, dest)
        self.assertEqual(dest.read_bytes(), b'do_not_touch')

    def test_same_volume_guard(self):
        with self.assertRaisesRegex(ValueError, 'distinct mounted volumes'):
            protect(self.src, self.vault, self.sha, self.label, different_volume=True)

    def test_source_symlink_rejected(self):
        alias = self.root / 'alias.pt'
        try:
            alias.symlink_to(self.src)
        except (OSError, NotImplementedError):
            self.skipTest('symlinks unavailable')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            protect(alias, self.vault, self.sha, self.label)

    def test_wrong_backup_sha_rejected(self):
        report = self.saved()
        with self.assertRaisesRegex(ValueError, 'independent pin'):
            verify(report['backup'], 'INVALID')
        with self.assertRaisesRegex(ValueError, 'manifest digest'):
            verify(report['backup'], 'f' * 64)


if __name__ == '__main__':
    unittest.main()