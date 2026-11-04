"""Safety contracts for offline rank32 workflow proposal (no GPU, no model load)."""
import hashlib
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
# Minimal upstream-anchor fixture; exact real GitHub trainer SHA is separately pinned.
FAKE_TRAINER = '''
    r=16, lora_alpha=16, lora_dropout=0,
    config = SFTConfig(
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args(argv)
    report = {"status": "started",
              "steps": args.steps, "max_length": args.max_length, "logs": [],}
    class Metrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            report["logs"].append({"step": state.global_step, **(logs or {})})
    samples = []
    report["metrics"] = trainer.train().metrics
    from neral_adapter_candidate import seal_adapter
'''
sys.path.insert(0, str(ROOT))
from scripts.neral_rank32_copy import patched_trainer, EXPECTED_SOURCE_GIT_BLOB
from scripts.neral_qlora_action_report import ReportError, summarize_run, save_summary


class Rank32WorkflowContracts(TestCase):
    def _data(self, tmp):
        adapter = Path(tmp) / 'adapter_model.safetensors'
        adapter.write_bytes(b'fixture not a true model')
        report = {
            'status': 'completed', 'steps': 3, 'optimizer_steps_completed': 3,
            'neral_trainable_parameters': 80740352,
            'logs': [{'step': i, 'loss': float(4-i)} for i in [1,2,3]],
            'optimizer_step_seconds': [0.5,0.9,1.1],
            'metrics': {'train_loss': 2.0}, 'elapsed_seconds': 3.1,
            'peak_allocated_bytes': 9_000_000_000,
            'peak_reserved_bytes': 9_500_000_000,
            'device_used_peak_sampled_bytes': 10_000_000_000
        }
        return report, adapter

    def test_patch_pinned_and_exact_three_steps_with_timing(self):
        source = FAKE_TRAINER
        patched = patched_trainer(source)
        self.assertIn('r=args.lora_rank', patched)
        self.assertIn('if args.lora_rank == 32 and args.steps != 3:',patched)
        self.assertIn('on_step_begin', patched)
        self.assertIn('on_step_end',patched)
        self.assertIn('optimizer_steps_completed', patched)

    def test_report_contains_only_numeric_metrics_and_adapter_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,adapter=self._data(tmp)
            x=summarize_run(report,adapter)
            self.assertEqual(x['optimizer_steps'],3)
            self.assertEqual(x['adapter_sha256'],hashlib.sha256(adapter.read_bytes()).hexdigest())
            self.assertEqual(len(x['step_metrics']),3)
            self.assertFalse(x['deployed'])
            self.assertFalse(x['quality_improvement_measured'])
            self.assertNotIn('paths',x)
            self.assertNotIn('model_snapshot',x)
            save_summary(Path(tmp)/'report.json',x)
            with self.assertRaises(ReportError):save_summary(Path(tmp)/'report.json',x)

    def test_reject_false_three_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,adapter=self._data(tmp)
            report['optimizer_steps_completed']=2
            with self.assertRaises(ReportError):summarize_run(report,adapter)

    def test_reject_changed_trainable_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,adapter=self._data(tmp)
            report['neral_trainable_parameters']=40370176
            with self.assertRaises(ReportError):summarize_run(report,adapter)

    def test_reject_missing_or_nonfinite_loss(self):
        with tempfile.TemporaryDirectory() as tmp:
            report,adapter=self._data(tmp)
            report['logs'][1]['loss']=float('nan')
            with self.assertRaises(ReportError):summarize_run(report,adapter)
            report,adapter=self._data(tmp)
            report['logs'].pop()
            with self.assertRaises(ReportError):summarize_run(report,adapter)

    def test_workflow_never_runs_unreviewed_push(self):
        content=(ROOT/'.github'/'workflows'/'neral-qwen7b-r32-qlora.yml').read_text('utf-8')
        self.assertIn('workflow_dispatch:',content)
        self.assertNotIn('\n  push:',content)
        self.assertIn("github.ref == 'refs/heads/main'",content)
        self.assertIn('persist-credentials: false',content)
        self.assertNotIn('pull_request:',content)

    def test_artifact_upload_depends_on_allowlist_guard(self):
        content=(ROOT/'.github'/'workflows'/'neral-qwen7b-r32-qlora.yml').read_text('utf-8')
        self.assertIn('id: upload_guard',content)
        self.assertIn("steps.upload_guard.outputs.safe == 'true'",content)
        self.assertNotIn('adapter_model.safetensors',content)
