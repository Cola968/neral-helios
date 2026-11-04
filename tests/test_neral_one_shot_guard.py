"""Regression tests for persistent one-shot pilot safety.

Never starts a model or imports torch.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
GUARD = ROOT / 'scripts' / 'neral_one_shot_guard.py'
PS = ROOT / 'scripts' / 'Start-NERAL-QLoRA-Action.ps1'


class OneShotGuardTests(unittest.TestCase):
    def call(self, root, run='41901632244', attempt='1'):
        return subprocess.run([sys.executable, str(GUARD), '--root', str(root), '--run-id', run,
                               '--attempt', attempt], text=True, capture_output=True)

    def test_exactly_one_claim_is_allowed_forever(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            first=self.call(root)
            self.assertEqual(first.returncode, 0, first.stderr)
            marker=root/'.neral-r32-three-step-one-shot.lock'
            self.assertTrue(marker.is_file())
            record=json.loads(marker.read_text())
            self.assertEqual(record, {'schema':1,'action':'r32_qlora_three_optimizer_steps',
                                      'run_id':'41901632244','attempt':1})
            again=self.call(root)
            self.assertNotEqual(again.returncode, 0)
            self.assertEqual(marker.read_text(),json.dumps(record,sort_keys=True)+'\n')
            different_run=self.call(root,run='41901632245')
            self.assertNotEqual(different_run.returncode,0)

    def test_rejects_invalid_run_ids_and_attempts_without_creating_lock(self):
        with tempfile.TemporaryDirectory() as d:
            for run, attempt in [('bad','1'),('41901632244','0'),('41901632244','x'),('123456','9999'),
                                 ('0'*25,'1')]:
                self.assertNotEqual(self.call(d,run=run,attempt=attempt).returncode,0)
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_absent_root_does_not_create_directories(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'no-root'
            self.assertNotEqual(self.call(path).returncode,0)
            self.assertFalse(path.exists())

    def test_existing_symlink_is_rejected_without_touching_target(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            target=root/'target.txt';target.write_text('original')
            link=root/'.neral-r32-three-step-one-shot.lock'
            try:link.symlink_to(target)
            except OSError:self.skipTest('symlink privilege unavailable')
            self.assertNotEqual(self.call(root).returncode,0)
            self.assertEqual(target.read_text(),'original')

    def test_powershell_claim_occurs_only_after_preflight_and_before_training(self):
        s=PS.read_text(encoding='utf-8')
        self.assertIn('neral_one_shot_guard.py',s)
        self.assertLess(s.index('if ($Mode -eq \'preflight\')'),s.index('neral_one_shot_guard.py'))
        self.assertLess(s.index('neral_one_shot_guard.py'),s.index("$step = 'training_started'"))
        self.assertIn('One-shot authorization already consumed or unsafe',s)
        self.assertNotIn('Remove-Item $oneShot',s)

if __name__=='__main__':unittest.main()
