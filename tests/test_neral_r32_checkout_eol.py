"""Windows newline conversion must not silently invalidate reviewed corpus hashes."""
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class CheckoutEolTests(unittest.TestCase):
    def test_checkout_index_restores_exact_git_lf_without_changing_commit(self):
        with tempfile.TemporaryDirectory() as d:
            repo = Path(d)
            subprocess.run(['git', 'init', '-q', str(repo)], check=True, capture_output=True)
            subprocess.run(['git', '-C', str(repo), 'config', 'user.name', 'Test'], check=True)
            subprocess.run(['git', '-C', str(repo), 'config', 'user.email', 't@example.invalid'], check=True)
            file = repo / 'test.jsonl'
            file.write_bytes(b'{"a":1}\n{"a":2}\n')
            subprocess.run(['git', '-C', str(repo), 'add', '.'], check=True)
            subprocess.run(['git', '-C', str(repo), 'commit', '-qm', 'frozen'], check=True)
            sha = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).strip()
            file.write_bytes(b'{"a":1}\r\n{"a":2}\r\n')
            subprocess.run(['git', '-C', str(repo), '-c', 'core.autocrlf=false',
                            '-c', 'core.eol=lf', 'checkout-index', '--force', '--all'], check=True)
            self.assertEqual(file.read_bytes(), b'{"a":1}\n{"a":2}\n')
            self.assertEqual(subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD']).strip(), sha)

    def test_fixed_sha_pin_before_restoration_and_corpus_review_after(self):
        ps = (ROOT / 'scripts/Start-NERAL-QLoRA-Action.ps1').read_text(encoding='utf-8')
        self.assertIn('core.autocrlf=false', ps)
        self.assertIn('core.eol=lf', ps)
        self.assertIn('checkout-index --force --all', ps)
        self.assertLess(ps.index('Trainer source SHA differs'), ps.index('checkout-index --force --all'))
        self.assertLess(ps.index('checkout-index --force --all'), ps.index('neral_capability_train_pack.py'))
        self.assertIn('Dataset review or 86-row pack rejected', ps)
        self.assertIn('PINNED_CHECKOUT_ORIGINAL_GIT_BLOB_BYTES_RESTORED', ps)


if __name__ == '__main__':
    unittest.main()
