"""Opt-in, BOUNDED, distinct weight-only 350M pilot. NEVER an exact historical resume.

This module is not invoked by any Action. It cannot run without a SHA-pinned
operator-approved manifest, independently frozen local holdout and data review.
A new AdamW optimizer, zero NEW-token counter, and separate create-only output
are mandatory. Subsequent pilot resumes require a genuinely complete checkpoint.
It does not promise model improvement, RTX 4070 feasibility, or FP32 masters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import time

from helios350_preview import CONTINUITY_SHA, TOKENIZER_SHA, trusted_asset
from phoenix_continuity_cpu_probe import EXPECTED_TOKENS, require_pinned_checkpoint
from phoenix_warmstart_readiness import assess_weight_only_source, require_new_destination
from checkpoint_journal import CheckpointJournal, safe_save_every
from storage_budget import GIB, MIN_FREE_BYTES, check_storage_budget

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from native_resume_preflight350 import (corpus_identity, inspect_checkpoint,
                                         read_verified_resume, run_contract)
from recovery_identity import file_sha
from window_sampler350 import SAMPLER_VERSION

MAX_PILOT_NEW_TOKENS = 262_144
MIN_INITIAL_RESERVE = 12 * GIB
_SHA = re.compile(r'[0-9a-f]{64}\Z')


def require_fresh_stop_boundary(*, stop: int, target: int, update: int) -> None:
    """A pause inside a full AdamW update cannot be exactly continued."""
    if (any(type(v) is not int for v in (stop, target, update)) or
            not 0 < stop <= target <= MAX_PILOT_NEW_TOKENS or update <= 0):
        raise ValueError('invalid or unbounded fresh token pilot')
    if stop < target and stop % update:
        raise ValueError('partial intermediate optimizer update would poison exact resume')


def verify_operator_approval(*, manifest: Path, manifest_sha: str, expected: dict,
                             review: Path, holdout: Path) -> dict:
    """Check explicit local human authorization, source review and sealed holdout.

    Hash binding prevents accidental substitution; it is NOT a digital signature
    or an independent assurance of licensing, correctness or review quality.
    """
    if not isinstance(manifest_sha, str) or not _SHA.fullmatch(manifest_sha):
        raise ValueError('approval manifest needs independently recorded full SHA-256')
    for file in (manifest, review, holdout):
        if (not Path(file).is_absolute() or Path(file).is_symlink() or
                not Path(file).is_file() or Path(file).stat().st_size == 0):
            raise ValueError('approval, reviewed data and frozen holdout must be existing local regular files')
    if file_sha(manifest) != manifest_sha:
        raise ValueError('approval manifest SHA mismatch')
    approval = json.loads(Path(manifest).read_text(encoding='utf-8'))
    required = set(expected) | {'schema', 'approved_for_bounded_pilot',
                                'human_source_review_complete', 'human_holdout_review_complete',
                                'review_sha256', 'holdout_sha256'}
    if not isinstance(approval, dict) or set(approval) != required:
        raise ValueError('approval fields are incomplete or unexpected')
    if (type(approval['schema']) is not int or approval['schema'] != 1 or
            approval['approved_for_bounded_pilot'] is not True or
            approval['human_source_review_complete'] is not True or
            approval['human_holdout_review_complete'] is not True):
        raise ValueError('human data/run and holdout approvals are not complete')
    if any(approval[k] != v for k, v in expected.items()):
        raise ValueError('approved output, sources, budget or run parameters changed')
    if (approval['review_sha256'] != file_sha(review) or
            approval['holdout_sha256'] != file_sha(holdout) or
            approval['review_sha256'] == approval['holdout_sha256']):
        raise ValueError('source review or held-out assessment SHA mismatch')
    return approval


def plan_lineage(*, approval_sha: str, output: Path, source: Path,
                 run: dict, target: int, holdout_sha: str) -> dict:
    return {'kind': 'explicit_weight_only_new_optimizer_v1',
            'source_sha256': CONTINUITY_SHA,
            'historical_processed_tokens': EXPECTED_TOKENS,
            'new_run_initial_tokens': 0, 'new_run_initial_step': 0,
            'approval_sha256': approval_sha, 'holdout_sha256': holdout_sha,
            'source_path': str(source), 'output_path': str(output),
            'new_target_tokens': target, 'run_contract': run}


def parser_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--execute', action='store_true')
    ap.add_argument('--trusted-local-checkpoint', action='store_true')
    ap.add_argument('--source-checkpoint', type=Path, required=True)
    ap.add_argument('--assets-root', type=Path, required=True)
    ap.add_argument('--train-data', type=Path, required=True)
    ap.add_argument('--valid-data', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--approval', type=Path, required=True)
    ap.add_argument('--approval-sha256', required=True)
    ap.add_argument('--source-review', type=Path, required=True)
    ap.add_argument('--frozen-holdout', type=Path, required=True)
    ap.add_argument('--target-new-tokens', type=int, required=True)
    ap.add_argument('--stop-after-new-tokens', type=int, default=0)
    ap.add_argument('--seq', type=int, default=128)
    ap.add_argument('--microbatch', type=int, default=1)
    ap.add_argument('--grad-accum', type=int, default=4)
    ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--weight-decay', type=float, default=0.1)
    ap.add_argument('--warmup-frac', type=float, default=0.02)
    ap.add_argument('--seed', type=int, default=35001)
    ap.add_argument('--save-every', type=int, default=10)
    ap.add_argument('--eval-every', type=int, default=10)
    ap.add_argument('--eval-batches', type=int, default=4)
    ap.add_argument('--resume', type=Path, default=None)
    ap.add_argument('--resume-sha256', default='')
    ap.add_argument('--compile-mode', choices=('none', 'default'), default='none')
    return ap.parse_args(argv)


def prepare(args) -> dict:
    """All data/approval/source/provenance validation, before Torch/CUDA/output."""
    if not (args.execute and args.trusted_local_checkpoint):
        raise ValueError('explicit --execute and trust required; no training started')
    source = require_pinned_checkpoint(args.source_checkpoint)
    out = Path(args.out)
    if not out.is_absolute():
        raise ValueError('absolute distinct output directory required')
    if args.resume is None:
        if args.resume_sha256:
            raise ValueError('resume SHA without resume file')
        out = require_new_destination(out, source)
    else:
        if out.is_symlink() or not out.is_dir() or not args.resume.is_absolute():
            raise ValueError('existing real output required for explicit pilot resume')
        out = out.resolve(strict=True)
        if (args.resume.is_symlink() or not args.resume.is_file() or
                args.resume.resolve(strict=True) != out / 'resume.pt'):
            raise ValueError('resume must be owned resume.pt, no symlink')
        if not _SHA.fullmatch(args.resume_sha256):
            raise ValueError('explicit exact new-pilot resume SHA required')
    tok_path = trusted_asset(args.assets_root, args.assets_root / 'frontier350_assets' /
                             'tokenizer.json', TOKENIZER_SHA)
    if not tok_path.is_file():
        raise ValueError('missing own pinned tokenizer')
    if (any(type(x) is not int or x <= 0 for x in
            (args.seq, args.microbatch, args.grad_accum, args.eval_every,
             args.eval_batches, args.save_every)) or
            args.seq > 8192 or type(args.seed) is not int):
        raise ValueError('invalid training dimensions')
    if (not all(math.isfinite(x) for x in (args.lr, args.weight_decay, args.warmup_frac)) or
            not 0 < args.lr <= 0.0001 or not 0 <= args.weight_decay <= 0.5 or
            not 0 <= args.warmup_frac <= 1):
        raise ValueError('invalid or high-risk new optimizer settings')
    update = args.seq * args.microbatch * args.grad_accum
    target = args.target_new_tokens
    stop = args.stop_after_new_tokens or target
    require_fresh_stop_boundary(stop=stop, target=target, update=update)
    train_files = sorted(args.train_data.glob('*.bin'))
    valid_files = sorted(args.valid_data.glob('*.bin'))
    train_id = corpus_identity(train_files, seq=args.seq, role='train',
                               seed=args.seed + 1, tokenizer_sha=TOKENIZER_SHA)
    val_id = corpus_identity(valid_files, seq=args.seq, role='valid',
                             seed=args.seed + 999, batch=args.microbatch,
                             batches=args.eval_batches, tokenizer_sha=TOKENIZER_SHA)
    if {file_sha(p) for p in train_files} & {file_sha(p) for p in valid_files}:
        raise ValueError('identical source and validation bytes forbidden')
    contract = run_contract(seq=args.seq, microbatch=args.microbatch,
                            grad_accum=args.grad_accum, seed=args.seed, lr=args.lr,
                            schedule='cosine', weight_decay=args.weight_decay,
                            warmup_frac=args.warmup_frac, gradient_checkpointing=True)
    # Distinct native checkpoint contract makes accidental resume through the
    # ordinary from-scratch entrypoint reject this lineage.
    contract.update({'initialization': 'weight_only_new_optimizer_v1',
                     'source_sha256': CONTINUITY_SHA})
    expected = {'source_sha256': CONTINUITY_SHA, 'tokenizer_sha256': TOKENIZER_SHA,
                'train_identity': train_id, 'validation_identity': val_id,
                'output': str(out), 'target_new_tokens': target,
                'stop_after_new_tokens': stop, 'run_contract': contract}
    approval = verify_operator_approval(manifest=args.approval,
                                        manifest_sha=args.approval_sha256,
                                        expected=expected, review=args.source_review,
                                        holdout=args.frozen_holdout)
    lineage = plan_lineage(approval_sha=args.approval_sha256, source=source,
                           output=out, run=contract, target=target,
                           holdout_sha=approval['holdout_sha256'])
    directory = out if args.resume else out.parent
    free = shutil.disk_usage(directory).free
    if free < MIN_FREE_BYTES + MIN_INITIAL_RESERVE:
        raise RuntimeError('less than 200 GiB plus 12 GiB checkpoint reserve')
    if args.resume:
        check_storage_budget(out, args.resume)
    return {'source': source, 'out': out, 'train_files': train_files,
            'valid_files': valid_files, 'train_id': train_id, 'val_id': val_id,
            'contract': contract, 'lineage': lineage, 'stop': stop, 'target': target,
            'update': update}


def main(argv=None) -> None:
    args = parser_args(argv)
    plan = prepare(args)  # SHA, real assets, approval, disk checks before GPU.
    import phoenix_cuda_diagnostic as gpu_guard
    gpu_guard.require_stably_idle_gpu()
    import numpy as np
    import torch
    from model350 import FRONTIER_350, Neral350M
    from train350 import (append_jsonl, atomic_save, checkpoint_state,
                          cosine_lr, mask_budget_targets, update_token_counts)
    from train350_compiled import PersistentMMapPool, evaluate
    gpu_guard.require_cuda_ready(torch)
    if args.seq > FRONTIER_350.max_seq_len:
        raise ValueError('model context exceeded')
    ck = None
    source = None
    if args.resume:
        ck = read_verified_resume(checkpoint=args.resume,
                                  expected_sha=args.resume_sha256, trusted=True,
                                  out=plan['out'], torch=torch,
                                  config=FRONTIER_350.to_dict(), target_tokens=plan['target'],
                                  tokenizer_sha=TOKENIZER_SHA,
                                  train_identity=plan['train_id'],
                                  validation_identity=plan['val_id'],
                                  expected_run=plan['contract'], run_until=plan['stop'])
        if ck.get('warmstart_lineage') != plan['lineage']:
            raise ValueError('new-run checkpoint origin/holdout/authorization changed')
    else:
        try:
            source = torch.load(plan['source'], map_location='cpu', mmap=True,
                                weights_only=True)
        except (ValueError, RuntimeError, TypeError, __import__('pickle').UnpicklingError):
            source = torch.load(plan['source'], map_location='cpu', mmap=True,
                                weights_only=False)  # ONLY trusted, SHA-pinned own file.
        with torch.device('meta'):
            reference = Neral350M()
        layout = {k: (tuple(v.shape), str(v.dtype))
                  for k, v in reference.state_dict().items()}
        assess_weight_only_source(source, expected_config=FRONTIER_350.to_dict(),
                                  expected_layout=layout, tokenizer_sha=TOKENIZER_SHA)
        if not torch.equal(source['model']['tok_emb.weight'],
                           source['model']['lm_head.weight']):
            raise ValueError('historical tied output and embedding weights disagree')
    # The output is created exclusively after all operator, model and GPU gates.
    if not args.resume:
        plan['out'].mkdir(exist_ok=False)
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(True)
    if model.parameter_report()['total'] != 349_825_152:
        raise RuntimeError('locked model parameter count changed')
    if source is not None:
        model.load_state_dict(source['model'], strict=True)
        del source
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  betas=(0.9, 0.95), eps=1e-8,
                                  weight_decay=args.weight_decay, fused=True)
    rng = random.Random(args.seed + 1)
    step = processed = 0
    best = float('inf')
    if ck is not None:
        model.load_state_dict(ck['model'], strict=True)
        optimizer.load_state_dict(ck['optimizer'])
        step, processed = ck['step'], ck['processed_tokens']
        best = float(ck.get('best_validation', best))
        torch.set_rng_state(ck['torch_rng'])
        torch.cuda.set_rng_state_all(ck['cuda_rng'])
        np.random.set_state(ck['numpy_rng'])
        random.setstate(ck['python_rng'])
        rng.setstate(ck['train_sampler_rng'])
        del ck
    train = PersistentMMapPool(args.train_data, FRONTIER_350.vocab_size)
    valid = PersistentMMapPool(args.valid_data, FRONTIER_350.vocab_size)
    compiled = model if args.compile_mode == 'none' else torch.compile(
        model, backend='inductor', mode='default', dynamic=False)
    updates = math.ceil(plan['target'] / plan['update'])
    warmup = max(1, int(updates * args.warmup_frac))
    cadence = safe_save_every(seq=args.seq, microbatch=args.microbatch,
                              accumulation=args.grad_accum, requested=args.save_every)
    metadata = {'train_identity': plan['train_id'],
                'validation_identity': plan['val_id'], 'sampler_version': SAMPLER_VERSION,
                'run_contract': plan['contract']}
    metrics = plan['out'] / 'metrics.jsonl'
    append_jsonl(metrics, {'event': 'separate_weight_only_pilot',
                           'historical_tokens': EXPECTED_TOKENS,
                           'new_processed_tokens': processed, 'new_step': step,
                           'lineage': plan['lineage'], 'approved_manifest_sha256': args.approval_sha256,
                           'model_quality_gain_verified': False})
    journal = None
    if args.resume:
        journal = CheckpointJournal(plan['out'] / 'resume.pt',
                                    initial_sha256=args.resume_sha256,
                                    initial_tokens=processed, writer=atomic_save)
    start = time.monotonic()
    model.train()
    while processed < plan['stop']:
        lr = cosine_lr(step, warmup, updates, args.lr)
        for group in optimizer.param_groups:
            group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        counts = update_token_counts(plan['stop'] - processed,
                                     args.seq * args.microbatch, args.grad_accum)
        total = sum(counts)
        loss_total = torch.zeros((), device=device, dtype=torch.float32)
        for count in counts:
            x, y = train.batch(args.microbatch, args.seq, device, rng)
            y = mask_budget_targets(y, count)
            _, loss = compiled(x, y)
            if loss is None:
                raise RuntimeError('missing pretraining loss')
            (loss * (count / total)).backward()
            loss_total.add_(loss.detach().float() * (count / total))
        grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                              error_if_nonfinite=True)
        if not math.isfinite(float(loss_total)) or not math.isfinite(float(grad)):
            raise FloatingPointError('nonfinite loss or gradient, update refused')
        gpu_guard.require_safe_runtime_temperature()
        optimizer.step()
        processed += total
        step += 1
        gpu_guard.require_safe_runtime_temperature()
        if step % args.eval_every == 0 or processed >= plan['stop']:
            val = evaluate(model, valid, seq=args.seq, batch=args.microbatch,
                           batches=args.eval_batches, device=device,
                           seed=args.seed + 999)
            best = min(best, val)
            append_jsonl(metrics, {'event': 'validation', 'new_step': step,
                                   'new_tokens': processed, 'loss': val,
                                   'validation_identity': plan['val_id']})
        if step % cadence == 0 or processed >= plan['stop']:
            state = checkpoint_state(model, optimizer, step, processed, best,
                                     plan['target'], TOKENIZER_SHA,
                                     train_sampler_rng=rng,
                                     validation_identity=plan['val_id'],
                                     resume_metadata=metadata)
            state['warmstart_lineage'] = plan['lineage']
            if journal is None:
                atomic_save(state, plan['out'] / 'resume.pt')
                journal = CheckpointJournal(plan['out'] / 'resume.pt',
                                            initial_sha256=file_sha(plan['out'] / 'resume.pt'),
                                            initial_tokens=processed, writer=atomic_save)
            else:
                check_storage_budget(plan['out'], plan['out'] / 'resume.pt')
                journal.commit(state, plan['out'] / 'resume.pt')
        append_jsonl(metrics, {'event': 'step', 'new_step': step,
                               'new_tokens': processed, 'loss': float(loss_total),
                               'lr': lr, 'grad_norm': float(grad),
                               'elapsed_s': time.monotonic() - start})
    if journal is None:
        raise RuntimeError('new run terminated before a resumable checkpoint')
    final_receipt = journal.verify()
    summary = {'event': 'weight_only_pilot_finished',
               'initial_historical_tokens': EXPECTED_TOKENS,
               'new_processed_tokens': processed, 'new_optimizer_steps': step,
               'resume_sha256': final_receipt['sha256'],
               'output': str(plan['out']), 'quality_gain_verified': False,
               'source_unchanged': file_sha(plan['source']) == CONTINUITY_SHA}
    (plan['out'] / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
