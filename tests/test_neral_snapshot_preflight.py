"""CPU-only snapshot validation regression; does not read a 7B model."""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.neral_snapshot_preflight import verify_snapshot

MODEL = 'Qwen/Qwen2.5-Coder-7B-Instruct'


class SnapshotPreflightTests(unittest.TestCase):
    def test_metadata_then_hash_verify(self):
        calls = []
        def validate(path, verify_hashes=False):
            calls.append((str(path), verify_hashes))
            return {'base_model': MODEL, 'revision': 'c'*40, 'sha256': 'a'*64}
        value = verify_snapshot(Path('model'), validate)
        self.assertEqual(value['sha256'], 'a'*64)
        self.assertEqual(calls, [('model', False), ('model', True)])

    def test_wrong_base_fails_before_hashing(self):
        calls = []
        def validate(path, verify_hashes=False):
            calls.append(verify_hashes)
            return {'base_model': 'Other/Model', 'revision': 'c'*40, 'sha256': 'a'*64}
        with self.assertRaises(ValueError):
            verify_snapshot(Path('model'), validate)
        self.assertEqual(calls, [False])

    def test_integrity_failure_not_swallowed(self):
        def validate(path, verify_hashes=False):
            if verify_hashes: raise ValueError('External snapshot integrity failed')
            return {'base_model': MODEL, 'revision': 'c'*40, 'sha256': 'a'*64}
        with self.assertRaisesRegex(ValueError, 'integrity failed'):
            verify_snapshot(Path('model'), validate)

    def test_changed_manifest_identity_rejected(self):
        def validate(path, verify_hashes=False):
            return {'base_model': MODEL, 'revision': 'c'*40,
                    'sha256': ('b' if verify_hashes else 'a')*64}
        with self.assertRaisesRegex(ValueError, 'manifest changed'):
            verify_snapshot(Path('model'), validate)

    def test_windows_entrypoint_uses_proven_python_and_script_not_inline(self):
        ps=(ROOT/'scripts/Start-NERAL-QLoRA-Action.ps1').read_text(encoding='utf-8')
        self.assertIn('outputs\\neral\\.venv\\Scripts\\python.exe',ps)
        self.assertIn("neral_snapshot_preflight.py') --snapshot $model",ps)
        self.assertNotIn('& $python -c $check $model',ps)
        self.assertIn('if ($LASTEXITCODE -ne 0)',ps)

    def test_does_not_weaken_hash_verification_or_one_shot(self):
        ps=(ROOT/'scripts/Start-NERAL-QLoRA-Action.ps1').read_text(encoding='utf-8')
        self.assertIn('neral_one_shot_guard.py',ps)
        source=(ROOT/'scripts/neral_snapshot_preflight.py').read_text(encoding='utf-8')
        self.assertIn('verify_hashes=True',source)
        self.assertIn('verify_hashes=False',source)
        self.assertIn('SNAPSHOT_FAILURE_REASON',source)


if __name__ == '__main__':
    unittest.main()
