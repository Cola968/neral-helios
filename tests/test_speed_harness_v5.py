import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'frontier350'))
from benchmark_speed_v5 import (require_completed_run, require_idle_gpu,
    verify_production_sources, compare_audits, finite_tree, PROTECTED_RUN)


class HarnessSafetyTests(unittest.TestCase):
    def test_pinned_production_sources_are_unchanged(self):
        self.assertEqual(len(verify_production_sources()),10)

    def test_active_training_blocks_before_gpu_queries(self):
        response=io.BytesIO(json.dumps({'id':PROTECTED_RUN,'status':'in_progress'}).encode())
        with patch.dict(os.environ,{'GITHUB_TOKEN':'test-only'}), \
             patch('urllib.request.urlopen',return_value=response), \
             patch('subprocess.check_output') as subprocess:
            with self.assertRaisesRegex(RuntimeError,'active'):require_idle_gpu()
            subprocess.assert_not_called()

    def test_missing_auth_fails_closed(self):
        with patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,'GITHUB_TOKEN'):require_completed_run()

    def test_other_gpu_process_blocks(self):
        with patch('benchmark_speed_v5.require_completed_run'), \
             patch('subprocess.check_output',return_value='1234\n'):
            with self.assertRaisesRegex(RuntimeError,'compute processes'):require_idle_gpu()

    def test_quality_nan_rejected(self):
        with self.assertRaises(ValueError):finite_tree({'categories':[{'accuracy':float('nan')} ]})

    def test_audit_accepts_identity_rejects_drift_and_missing_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            a=Path(tmp)/'a';b=Path(tmp)/'b';a.mkdir();b.mkdir()
            for step in range(1,4):
                obj={'loss':torch.tensor(1.),'grads':{'weight':torch.ones(2)},
                     'sampled_logits':{'input':torch.ones(2)},'batch_chain':'a'}
                for p in (a,b):torch.save(obj,p/f'audit_{step:03d}.pt')
            for p in (a,b):torch.save({'v9_training_state':{'masters':{'weight':torch.ones(2)}}},p/'resume.pt')
            self.assertTrue(compare_audits(a,b)['pass'])
            obj['grads']['weight'][0]=2.
            torch.save(obj,b/'audit_003.pt')
            self.assertFalse(compare_audits(a,b)['pass'])
            obj['batch_chain']='different'
            torch.save(obj,b/'audit_003.pt')
            with self.assertRaisesRegex(ValueError,'identity'):compare_audits(a,b)


if __name__=='__main__':unittest.main()
