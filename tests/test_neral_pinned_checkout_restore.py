"""Exact SHA-pinned Git restoration tests; no model or GPU used."""
import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.neral_pinned_checkout_restore import PinError, restore_selected


def git(repo, *args):
    return subprocess.check_output(['git', '-C', str(repo), *args]).strip()


class PinnedCheckoutRestorationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name)
        git(self.repo, 'init', '-q')
        git(self.repo, 'config', 'user.name', 'Test')
        git(self.repo, 'config', 'user.email', 'x@y.z')
        (self.repo / 'datasets').mkdir()
        self.one = self.repo / 'datasets/a.jsonl'
        self.two = self.repo / 'datasets/b.json'
        self.one.write_bytes(b'{"text":"Berlin"}\n{"text":"Dresden"}\n')
        self.two.write_bytes(b'{"ok":true}\n')
        git(self.repo, 'add', '.')
        git(self.repo, 'commit', '-qm', 'pinned')
        self.expected = {'datasets/a.jsonl': hashlib.sha256(self.one.read_bytes()).hexdigest(),
                         'datasets/b.json': hashlib.sha256(self.two.read_bytes()).hexdigest()}

    def test_restores_crlf_equivalent_blobs(self):
        self.one.write_bytes(self.one.read_bytes().replace(b'\n', b'\r\n'))
        before = git(self.repo, 'rev-parse', 'HEAD')
        self.assertEqual(restore_selected(self.repo, self.expected), 1)
        self.assertEqual(self.one.read_bytes(), b'{"text":"Berlin"}\n{"text":"Dresden"}\n')
        self.assertEqual(git(self.repo, 'rev-parse', 'HEAD'), before)

    def test_does_not_modify_unknown_change(self):
        self.one.write_bytes(b'{"text":"tampered"}\r\n')
        with self.assertRaises(PinError):
            restore_selected(self.repo, self.expected)
        self.assertEqual(self.one.read_bytes(), b'{"text":"tampered"}\r\n')

    def test_git_blob_must_match_expected_hash(self):
        wrong = dict(self.expected)
        wrong['datasets/a.jsonl'] = '0'*64
        self.one.write_bytes(self.one.read_bytes().replace(b'\n', b'\r\n'))
        with self.assertRaises(PinError):
            restore_selected(self.repo, wrong)
        self.assertIn(b'\r\n', self.one.read_bytes())

    def test_multi_file_validation_before_any_write(self):
        self.one.write_bytes(self.one.read_bytes().replace(b'\n', b'\r\n'))
        self.two.write_bytes(b'{"ok":false}\r\n')
        with self.assertRaises(PinError):
            restore_selected(self.repo, self.expected)
        self.assertIn(b'\r\n', self.one.read_bytes())

    def test_does_not_read_or_restore_parent_paths(self):
        with self.assertRaises(PinError):
            restore_selected(self.repo, {'../elsewhere.txt': 'a'*64})
        with self.assertRaises(PinError):
            restore_selected(self.repo, {'C:\\secrets.txt': 'a'*64})

    def test_starter_calls_pin_restorer_before_trainer_pack(self):
        s = (ROOT / 'scripts/Start-NERAL-QLoRA-Action.ps1').read_text('utf-8')
        self.assertIn('neral_pinned_checkout_restore.py', s)
        self.assertLess(s.index('neral_pinned_checkout_restore.py'), s.index('neral_capability_train_pack.py'))
        self.assertIn('Dataset review or 86-row pack rejected', s)


if __name__ == '__main__':
    unittest.main()
