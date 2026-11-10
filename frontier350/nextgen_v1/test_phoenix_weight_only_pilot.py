"""CPU/Torch-free preflight checks: no real training, weights or human approval."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from helios350_preview import CONTINUITY_SHA, TOKENIZER_SHA
from phoenix_continuity_cpu_probe import EXPECTED_TOKENS
from phoenix_weight_only_pilot import (MAX_PILOT_NEW_TOKENS, parser_args,
                                       plan_lineage, prepare,
                                       require_fresh_stop_boundary,
                                       verify_operator_approval)
from native_resume_preflight350 import corpus_identity, run_contract


def digest(data: bytes) -> str:
    return sha256(data).hexdigest()


class WarmstartPilotTests(unittest.TestCase):
    def test_fresh_stop_only_allows_full_updates_or_terminal_remainder(self):
        require_fresh_stop_boundary(stop=512, target=1024, update=128)
        require_fresh_stop_boundary(stop=513, target=513, update=128)
        for values in ((511, 1024, 128), (1025, 1024, 128),
                       (0, 1024, 128), (512, MAX_PILOT_NEW_TOKENS + 1, 128),
                       (512, 1024, 0), (512, 1024, -128)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                require_fresh_stop_boundary(stop=values[0], target=values[1],
                                            update=values[2])

    def test_original_historical_progress_never_becomes_new_training(self):
        lineage = plan_lineage(approval_sha='b' * 64,
                               output=Path('/tmp/new'), source=Path('/tmp/old.pt'),
                               run={'seed': 3}, target=1024, holdout_sha='c' * 64)
        self.assertEqual(lineage['historical_processed_tokens'], EXPECTED_TOKENS)
        self.assertEqual(lineage['new_run_initial_tokens'], 0)
        self.assertEqual(lineage['new_run_initial_step'], 0)
        self.assertEqual(lineage['source_sha256'], CONTINUITY_SHA)
        self.assertEqual(lineage['kind'], 'explicit_weight_only_new_optimizer_v1')

    def _fixtures(self, root: Path):
        source = root / 'original' / 'source.pt'
        source.parent.mkdir()
        source.write_bytes(b'synthetic, not a real torch checkpoint')
        data, valid = root / 'train', root / 'valid'
        data.mkdir()
        valid.mkdir()
        (data / 'a.bin').write_bytes(bytes(range(22)))
        (valid / 'b.bin').write_bytes(bytes(range(22, 44)))
        assets = root / 'assets' / 'frontier350_assets'
        assets.mkdir(parents=True)
        token = assets / 'tokenizer.json'
        token.write_bytes(b'synthetic test tokenizer; digest mocked only in test')
        out = root / 'fresh' / 'new-run'
        out.parent.mkdir()
        review = root / 'review.txt'
        review.write_text('synthetic: human review is NOT actually performed', encoding='utf8')
        holdout = root / 'holdout.jsonl'
        holdout.write_text('synthetic: not a real frozen assessment', encoding='utf8')
        approval = root / 'approval.json'
        args = parser_args([
            '--execute', '--trusted-local-checkpoint',
            '--source-checkpoint', str(source), '--assets-root', str(root / 'assets'),
            '--train-data', str(data), '--valid-data', str(valid),
            '--out', str(out), '--approval', str(approval),
            '--approval-sha256', '0' * 64, '--source-review', str(review),
            '--frozen-holdout', str(holdout), '--target-new-tokens', '16',
            '--stop-after-new-tokens', '8', '--seq', '4', '--microbatch', '1',
            '--grad-accum', '2', '--seed', '7'])
        contract = run_contract(seq=4, microbatch=1, grad_accum=2, seed=7,
                                lr=2e-5, schedule='cosine', weight_decay=0.1,
                                warmup_frac=0.02, gradient_checkpointing=True)
        contract.update({'initialization': 'weight_only_new_optimizer_v1',
                         'source_sha256': CONTINUITY_SHA, 'compile_mode': 'none',
                         'master_parameter_dtype': 'float32',
                         'adamw_moment_dtype': 'float32',
                         'compute_autocast_dtype': 'bfloat16'})
        expected = {'source_sha256': CONTINUITY_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
                    'train_identity': corpus_identity([data / 'a.bin'], seq=4,
                                                      role='train', seed=8,
                                                      tokenizer_sha=TOKENIZER_SHA),
                    'validation_identity': corpus_identity([valid / 'b.bin'], seq=4,
                                                           role='valid', seed=1006,
                                                           batch=1, batches=4,
                                                           tokenizer_sha=TOKENIZER_SHA),
                    'output': str(out), 'target_new_tokens': 16,
                    'stop_after_new_tokens': 8, 'run_contract': contract}
        approved = dict(expected, schema=1, approved_for_bounded_pilot=True,
                        human_source_review_complete=True,
                        human_holdout_review_complete=True,
                        review_sha256=digest(review.read_bytes()),
                        holdout_sha256=digest(holdout.read_bytes()))
        approval.write_text(json.dumps(approved), encoding='utf8')
        args.approval_sha256 = digest(approval.read_bytes())
        return args, approved, expected, source, token, approval, review, holdout

    def test_explicit_manifest_exactly_binds_source_data_output_and_holdout(self):
        with tempfile.TemporaryDirectory() as temp:
            args, approved, expected, _, _, manifest, review, holdout = self._fixtures(Path(temp))
            self.assertEqual(verify_operator_approval(
                manifest=manifest, manifest_sha=args.approval_sha256,
                expected=expected, review=review, holdout=holdout), approved)
            with self.assertRaisesRegex(ValueError, 'approved output'):
                verify_operator_approval(manifest=manifest,
                                         manifest_sha=args.approval_sha256,
                                         expected=dict(expected, output='/wrong'),
                                         review=review, holdout=holdout)
            holdout.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'assessment SHA mismatch'):
                verify_operator_approval(manifest=manifest,
                                         manifest_sha=args.approval_sha256,
                                         expected=expected, review=review, holdout=holdout)

    def test_missing_human_review_or_explicit_approval_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            args, approved, expected, _, _, manifest, review, holdout = self._fixtures(Path(temp))
            for field in ('approved_for_bounded_pilot', 'human_source_review_complete',
                          'human_holdout_review_complete'):
                damaged = dict(approved, **{field: False})
                manifest.write_text(json.dumps(damaged), encoding='utf8')
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'human'):
                    verify_operator_approval(manifest=manifest,
                                             manifest_sha=digest(manifest.read_bytes()),
                                             expected=expected, review=review,
                                             holdout=holdout)

    def test_no_run_without_explicit_execute_and_trust(self):
        with tempfile.TemporaryDirectory() as temp:
            args, *_ = self._fixtures(Path(temp))
            for field in ('execute', 'trusted_local_checkpoint'):
                old = getattr(args, field)
                setattr(args, field, False)
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'explicit'):
                    prepare(args)
                setattr(args, field, old)

    def test_source_and_tokenizer_preflight_before_gpu_or_output(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, source, token, *_ = self._fixtures(Path(temp))
            with patch('phoenix_weight_only_pilot.require_pinned_checkpoint',
                       return_value=source), patch('phoenix_weight_only_pilot.trusted_asset',
                                                   return_value=token), patch(
                                                       'phoenix_weight_only_pilot.shutil.disk_usage') as disk:
                disk.return_value.free = 300 * (1024 ** 3)
                plan = prepare(args)
            self.assertEqual((plan['stop'], plan['target'], plan['update']), (8, 16, 8))
            self.assertFalse(args.out.exists())
            self.assertEqual(plan['lineage']['new_run_initial_tokens'], 0)
            self.assertEqual(plan['contract']['compile_mode'], 'none')
            self.assertEqual(plan['contract']['master_parameter_dtype'], 'float32')
            self.assertEqual(plan['contract']['compute_autocast_dtype'], 'bfloat16')

    def test_manifest_tampering_and_preexisting_output_refused_without_writes(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, source, token, manifest, *_ = self._fixtures(Path(temp))
            with patch('phoenix_weight_only_pilot.require_pinned_checkpoint',
                       return_value=source), patch('phoenix_weight_only_pilot.trusted_asset',
                                                   return_value=token):
                manifest.write_bytes(manifest.read_bytes() + b' ')
                with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                    prepare(args)
                args.out.mkdir()
                with self.assertRaises(FileExistsError):
                    prepare(args)

    def test_oversized_GPU_settings_block_before_data_and_gpu(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, source, token, *_ = self._fixtures(Path(temp))
            with patch('phoenix_weight_only_pilot.require_pinned_checkpoint',
                       return_value=source), patch('phoenix_weight_only_pilot.trusted_asset',
                                                   return_value=token):
                for field, size in (('seq', 513), ('microbatch', 3), ('grad_accum', 17)):
                    with self.subTest(field=field):
                        original = getattr(args, field)
                        setattr(args, field, size)
                        with self.assertRaisesRegex(ValueError, 'oversized'):
                            prepare(args)
                        setattr(args, field, original)
                self.assertFalse(args.out.exists())

    def test_approved_execution_mode_cannot_change_during_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            args, _, _, source, token, *_ = self._fixtures(Path(temp))
            with patch('phoenix_weight_only_pilot.require_pinned_checkpoint',
                       return_value=source), patch('phoenix_weight_only_pilot.trusted_asset',
                                                   return_value=token):
                args.compile_mode = 'default'
                with self.assertRaisesRegex(ValueError, 'run parameters changed'):
                    prepare(args)
                self.assertFalse(args.out.exists())

    def test_approval_rejects_legacy_bf16_master_contract(self):
        with tempfile.TemporaryDirectory() as temp:
            args, approved, _, source, token, manifest, *_ = self._fixtures(Path(temp))
            approved['run_contract']['master_parameter_dtype'] = 'bfloat16'
            manifest.write_text(json.dumps(approved), encoding='utf8')
            args.approval_sha256 = digest(manifest.read_bytes())
            with patch('phoenix_weight_only_pilot.require_pinned_checkpoint',
                       return_value=source), patch('phoenix_weight_only_pilot.trusted_asset',
                                                   return_value=token):
                with self.assertRaisesRegex(ValueError, 'run parameters changed'):
                    prepare(args)


if __name__ == '__main__':
    unittest.main()
