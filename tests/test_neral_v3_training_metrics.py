import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v3_training_metrics import collect, summarize_rank, discover_source


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


class TrainingMetricsTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.experiment = self.root/'training'
        self.experiment.mkdir()
        self.source = self.root/'source'
        self.source.mkdir()
        self.report = self.root/'report'
        (self.source/'train.jsonl').write_bytes(b'{"messages": []}\n')
        self.data_sha = hashlib.sha256((self.source/'train.jsonl').read_bytes()).hexdigest()
        (self.source/'validation.jsonl').write_bytes(b'{\"messages\": [1]}\n')
        self.validation_sha = hashlib.sha256((self.source/'validation.jsonl').read_bytes()).hexdigest()
        dump(self.source/'summary.json', {
            'status': 'NERAL_V3_TWO_LINE_ALGEBRA_REVIEW_ONLY',
            'splits': {
                'train': {'count': 320, 'converted_equations': 142, 'data_sha256': self.data_sha},
                'validation': {'count': 64, 'converted_equations': 28, 'data_sha256': self.validation_sha}
            }
        })

    def add_rank(self, rank, *, bad_digest=False, failed=False):
        folder = self.experiment / f'rank{rank}'
        folder.mkdir()
        weights = b'only a test adapter, no actual model' + str(rank).encode()
        (folder/'adapter_model.safetensors').write_bytes(weights)
        digest = hashlib.sha256(weights).hexdigest()
        dump(folder/'neral-adapter.json', {
            'schema': 1, 'product': 'NERAL',
            'dataset_sha256': self.data_sha,
            'base_revision': 'c03e6d358207e414f1eca0bb1891e29f1db0e242',
            'release_validated': False,
            'files': {'adapter_model.safetensors': ('0'*64 if bad_digest else digest)}
        })
        dump(folder/'adapter_config.json', {
            'peft_type': 'LORA', 'r': rank, 'lora_alpha': rank
        })
        logs = [{'step': i, 'loss': 2.0-i/100} for i in range(1, 81)]
        dump(folder/'training-report.json', {
            'status': 'failed' if failed else 'completed',
            'steps': 80, 'optimizer_steps_completed': 40 if failed else 80,
            'lora_rank': rank,
            'neral_trainable_parameters': 40_370_176 if rank==16 else 80_740_352,
            'logs': logs if not failed else logs[:40],
            'optimizer_step_seconds': [3.1]*80 if not failed else [3.1]*40,
            'peak_allocated_bytes': 9000*1024**2,
            'peak_reserved_bytes': 9500*1024**2,
            'device_used_peak_sampled_bytes': 11000*1024**2,
            'elapsed_seconds': 300
        })
        return digest

    def test_reads_exact_loss_curve_peak_and_adapter_hash(self):
        h16 = self.add_rank(16)
        h32 = self.add_rank(32)
        report = collect(self.experiment, self.source)
        self.assertEqual(report['status'], 'BOTH_TRAINED_SCORES_PENDING')
        self.assertEqual(report['ranks']['rank16']['adapter_sha256'], h16)
        self.assertEqual(report['ranks']['rank32']['adapter_sha256'], h32)
        self.assertEqual(len(report['ranks']['rank16']['loss_curve']), 80)
        self.assertEqual(len(report['ranks']['rank32']['step_times_seconds']), 80)
        self.assertEqual(report['ranks']['rank16']['peak_vram_sampled_mib'], 11000)
        self.assertEqual(report['ranks']['rank32']['loss_curve'][0], {'step':1,'loss':1.99})
        self.assertFalse(report['release_validated'])
        self.assertFalse(report['weights_promoted'])
        self.assertEqual(report['source']['train_count'], 320)
        self.assertNotIn('messages', json.dumps(report))

    def test_integrity_violation_fails_closed(self):
        self.add_rank(16, bad_digest=True)
        with self.assertRaisesRegex(ValueError, 'adapter SHA256'):
            collect(self.experiment, self.source)

    def test_records_partial_failed_training_without_fabrication(self):
        self.add_rank(16)
        self.add_rank(32, failed=True)
        report=collect(self.experiment, self.source)
        self.assertEqual(report['status'], 'PARTIAL_OR_FAILED_TRAINING')
        self.assertEqual(report['ranks']['rank32']['status'], 'failed')
        self.assertIsNone(report['ranks']['rank32']['adapter_sha256'])
        self.assertEqual(report['ranks']['rank32']['steps_completed'], 40)
        self.assertFalse(report['evaluation_available'])

    def test_missing_training_is_not_reported_as_success(self):
        report=collect(self.experiment, self.source)
        self.assertEqual(report['status'], 'NO_TRAINING_REPORTS')
        self.assertEqual(report['ranks']['rank16']['status'], 'missing')

    def test_evaluation_whitelist_counts_and_hashes(self):
        self.add_rank(16)
        self.add_rank(32)
        scores={name:{suite:{'correct':n-1,'format_valid':n,'total':n}
                      for suite,n in [('validation64',64),('math80',80),('phrasing36',36),('holdout20',20)]}
                for name in ('base','rank16_v3','rank32_v3')}
        folder=self.experiment/'evaluation'
        folder.mkdir()
        dump(folder/'summary.json', {
            'status':'NERAL_V3_RANK16_RANK32_DEV_COMPARISON',
            'scores':scores,
            'adapter_16_sha256':hashlib.sha256((self.experiment/'rank16'/'adapter_model.safetensors').read_bytes()).hexdigest(),
            'adapter_32_sha256':hashlib.sha256((self.experiment/'rank32'/'adapter_model.safetensors').read_bytes()).hexdigest(),
            'validation_sha256':self.validation_sha,
            'independent_external_benchmark':False,
            'weights_promoted':False
        })
        report=collect(self.experiment,self.source)
        self.assertEqual(report['status'],'TRAINED_AND_DEVELOPMENT_EVALUATED')
        self.assertTrue(report['evaluation_available'])
        self.assertEqual(report['scores']['rank32_v3']['validation64']['correct'],63)
        self.assertFalse(report['independent_external_benchmark'])

    def test_wrong_eval_adapter_digest_is_rejected(self):
        self.add_rank(16);self.add_rank(32)
        folder=self.experiment/'evaluation';folder.mkdir()
        dump(folder/'summary.json', {'status':'NERAL_V3_RANK16_RANK32_DEV_COMPARISON',
            'adapter_16_sha256':'a'*64, 'adapter_32_sha256':'b'*64,
            'validation_sha256':self.validation_sha,'scores':{},'weights_promoted':False,
            'independent_external_benchmark':False})
        with self.assertRaises(ValueError):collect(self.experiment,self.source)

    def test_find_sealed_source_without_trusting_dynamic_workflow_input(self):
        self.add_rank(16)
        from_folder=self.source
        self.source=self.root/'math-v3-two-line-12345678-1'
        from_folder.rename(self.source)
        dump(self.source/'token-report.json', {'status':'NERAL_V3_TWO_LINE_TOKEN_GATE_PASS'})
        self.assertEqual(discover_source(self.experiment,self.root),self.source)
        report=collect(self.experiment,discover_source(self.experiment,self.root))
        self.assertEqual(report['ranks']['rank16']['status'],'completed')

    def test_failure_report_without_optimizer_steps_does_not_invent_completion(self):
        self.add_rank(16,failed=True)
        reportpath=self.experiment/'rank16'/'training-report.json'
        report=json.loads(reportpath.read_text())
        del report['optimizer_steps_completed']
        dump(reportpath,report)
        out=summarize_rank(self.experiment/'rank16',16,self.data_sha)
        self.assertEqual(out['steps_completed'],40)
        self.assertFalse(out['steps_completed_proven'])

    def test_invalid_loss_and_missing_step_times_cannot_be_faked(self):
        self.add_rank(16)
        reportpath=self.experiment/'rank16'/'training-report.json'
        report=json.loads(reportpath.read_text());report['logs'][3]['loss']=float('nan')
        dump(reportpath,report)
        with self.assertRaisesRegex(ValueError, 'finite'):
            summarize_rank(self.experiment/'rank16',16,self.data_sha)

if __name__ == '__main__': unittest.main()