from __future__ import annotations

"""Resumable native 350M trainer with SHA-pinned exact-resume CPU preflight.

Fresh runs and exact continuation are distinct. Never overwrite occupied output,
reset Adam moments implicitly, mix changed datasets or silently reuse validation
loss from another sampler. Frozen evaluation holdouts must never enter these shards.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from model350 import Neral350M, FRONTIER_350
from native_resume_preflight350 import corpus_identity, read_verified_resume, run_contract
from recovery_identity import file_sha
from window_sampler350 import SAMPLER_VERSION, WindowSampler

CHECKPOINT_SCHEMA = 2


def atomic_save(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf8') as f:
        f.write(json.dumps(obj, ensure_ascii=False) + '\n')
        f.flush()


def list_shards(root: Path):
    shards = sorted(root.glob('*.bin'))
    if not shards:
        raise FileNotFoundError(f'no *.bin shards in {root}')
    return shards


def sample_batch(shards, batch, seq, device, rng, vocab_size, sampler=None):
    sampler = sampler or WindowSampler([p.stat().st_size // 2 for p in shards], seq)
    rows, targets = [], []
    for _ in range(batch):
        shard, start = sampler.draw(rng)
        p = shards[shard]
        a = np.memmap(p, dtype=np.uint16, mode='r')
        chunk = np.asarray(a[start:start + seq + 1], dtype=np.int64)
        if chunk.shape[0] != seq + 1 or chunk.max(initial=0) >= vocab_size:
            raise ValueError(f'corrupt or out-of-vocabulary window in {p}')
        rows.append(torch.from_numpy(chunk[:-1].copy()))
        targets.append(torch.from_numpy(chunk[1:].copy()))
    return (torch.stack(rows).to(device, non_blocking=True),
            torch.stack(targets).to(device, non_blocking=True))


def cosine_lr(step, warmup, total, peak, min_ratio=0.08):
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    u = min(1.0, (step - warmup) / max(1, total - warmup))
    return peak * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * u)))


def evaluate(model, shards, seq, batch, batches, device, seed):
    rng = random.Random(seed)
    sampler = WindowSampler([p.stat().st_size // 2 for p in shards], seq)
    model.eval()
    vals = []
    with torch.no_grad():
        for _ in range(batches):
            x, y = sample_batch(shards, batch, seq, device, rng, model.cfg.vocab_size, sampler)
            _, loss = model(x, y)
            value = float(loss)
            if not math.isfinite(value):
                raise FloatingPointError('non-finite validation loss')
            vals.append(value)
    model.train()
    return sum(vals) / len(vals)


def checkpoint_state(model, opt, step, processed, best, target_tokens, tokenizer_sha256,
                     train_sampler_rng: random.Random | None = None,
                     validation_identity: str = '', resume_metadata: dict | None = None):
    state = {
        'checkpoint_schema': CHECKPOINT_SCHEMA,
        'model': model.state_dict(), 'optimizer': opt.state_dict(),
        'step': step, 'processed_tokens': processed, 'best_validation': best,
        'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all(),
        'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
        'config': FRONTIER_350.to_dict(), 'target_tokens': target_tokens,
        'tokenizer_sha256': tokenizer_sha256, 'validation_identity': validation_identity,
    }
    if train_sampler_rng is not None:
        state['train_sampler_rng'] = train_sampler_rng.getstate()
    if resume_metadata is not None:
        if not isinstance(resume_metadata, dict) or set(resume_metadata) != {
                'train_identity', 'validation_identity', 'sampler_version', 'run_contract'}:
            raise ValueError('invalid exact-resume metadata')
        if resume_metadata['validation_identity'] != validation_identity:
            raise ValueError('validation identity conflict in checkpoint')
        state.update(resume_metadata)
    return state


def update_token_counts(remaining: int, micro_tokens: int, accumulation: int) -> list[int]:
    if min(remaining, micro_tokens, accumulation) <= 0:
        raise ValueError('positive token-update dimensions required')
    budget = min(remaining, micro_tokens * accumulation)
    return [min(micro_tokens, budget - i) for i in range(0, budget, micro_tokens)]


def mask_budget_targets(targets: torch.Tensor, count: int) -> torch.Tensor:
    if not 0 < count <= targets.numel():
        raise ValueError('invalid supervised token count')
    if count == targets.numel():
        return targets
    result = targets.contiguous().clone()
    result.view(-1)[count:] = -100
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-data', required=True)
    ap.add_argument('--valid-data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--target-tokens', type=int, default=10_000_000_000)
    ap.add_argument('--stop-after-tokens', type=int, default=0)
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--microbatch', type=int, default=4)
    ap.add_argument('--grad-accum', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=0.1)
    ap.add_argument('--warmup-frac', type=float, default=0.01)
    ap.add_argument('--eval-every', type=int, default=100)
    ap.add_argument('--save-every', type=int, default=100)
    ap.add_argument('--log-every', type=int, default=5)
    ap.add_argument('--eval-batches', type=int, default=16)
    ap.add_argument('--seed', type=int, default=35001)
    ap.add_argument('--resume', default='')
    ap.add_argument('--resume-sha256', default='')
    ap.add_argument('--trusted-local-checkpoint', action='store_true')
    ap.add_argument('--gradient-checkpointing', action='store_true')
    ap.add_argument('--tokenizer-sha256', required=True)
    args = ap.parse_args()

    if args.target_tokens <= 0:
        raise ValueError('positive target token budget required')
    run_until = args.stop_after_tokens or args.target_tokens
    if run_until <= 0 or run_until > args.target_tokens:
        raise ValueError('invalid cumulative stop-after-tokens')
    if min(args.seq, args.microbatch, args.grad_accum, args.log_every,
           args.eval_every, args.save_every, args.eval_batches) <= 0:
        raise ValueError('invalid batch/check dimensions')
    if (not math.isfinite(args.lr) or args.lr <= 0 or not math.isfinite(args.weight_decay)
            or args.weight_decay < 0 or not 0 <= args.warmup_frac <= 1):
        raise ValueError('invalid optimizer settings')
    if args.seq > FRONTIER_350.max_seq_len:
        raise ValueError('sequence exceeds architecture context')

    out = Path(args.out).absolute()
    if out.exists() and any(out.iterdir()) and (not args.resume or Path(args.resume).resolve() != (out / 'resume.pt').resolve()):
        raise RuntimeError('existing output requires exact same-folder resume')
    train = list_shards(Path(args.train_data))
    valid = list_shards(Path(args.valid_data))
    train_identity = corpus_identity(train, seq=args.seq, role='train', seed=args.seed + 1,
                                    tokenizer_sha=args.tokenizer_sha256)
    validation_identity = corpus_identity(valid, seq=args.seq, role='valid',
                                          seed=args.seed + 999, batch=args.microbatch,
                                          batches=args.eval_batches,
                                          tokenizer_sha=args.tokenizer_sha256)
    if {file_sha(p) for p in train} & {file_sha(p) for p in valid}:
        raise ValueError('identical train and validation shard bytes forbidden')
    contract = run_contract(seq=args.seq, microbatch=args.microbatch,
                            grad_accum=args.grad_accum, seed=args.seed, lr=args.lr,
                            schedule='cosine', weight_decay=args.weight_decay,
                            warmup_frac=args.warmup_frac,
                            gradient_checkpointing=args.gradient_checkpointing)
    checkpoint = None
    if args.resume:
        checkpoint = read_verified_resume(
            checkpoint=Path(args.resume), expected_sha=args.resume_sha256,
            trusted=args.trusted_local_checkpoint, out=out, torch=torch,
            config=FRONTIER_350.to_dict(), target_tokens=args.target_tokens,
            tokenizer_sha=args.tokenizer_sha256, train_identity=train_identity,
            validation_identity=validation_identity, expected_run=contract,
            run_until=run_until)
    elif args.resume_sha256 or args.trusted_local_checkpoint:
        raise ValueError('resume verification flags require --resume')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 GPU required')
    device = torch.device('cuda')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    out.mkdir(parents=True, exist_ok=True)
    metrics = out / 'metrics.jsonl'
    rng = random.Random(args.seed + 1)
    train_sampler = WindowSampler([p.stat().st_size // 2 for p in train], args.seq)
    model = Neral350M().to(device=device, dtype=torch.bfloat16)
    model.set_gradient_checkpointing(args.gradient_checkpointing)
    report = model.parameter_report()
    if not 345_000_000 <= report['total'] <= 360_000_000:
        raise RuntimeError(f'Frontier parameter budget violated: {report}')
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95),
                            eps=1e-8, weight_decay=args.weight_decay, fused=True)
    tokens_per_update = args.seq * args.microbatch * args.grad_accum
    total_updates = math.ceil(args.target_tokens / tokens_per_update)
    warmup = max(1, int(total_updates * args.warmup_frac))
    step, processed, best = 0, 0, float('inf')
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=True)
        opt.load_state_dict(checkpoint['optimizer'])
        step, processed = checkpoint['step'], checkpoint['processed_tokens']
        best = float(checkpoint.get('best_validation', best))
        torch.set_rng_state(checkpoint['torch_rng'])
        random.setstate(checkpoint['python_rng'])
        np.random.set_state(checkpoint['numpy_rng'])
        torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
        rng.setstate(checkpoint['train_sampler_rng'])
        del checkpoint
    validation_seed = args.seed + 999
    startup = {
        'event': 'startup', 'parameters': report, 'config': FRONTIER_350.to_dict(),
        'checkpoint_schema': CHECKPOINT_SCHEMA, 'target_tokens': args.target_tokens,
        'run_until_tokens': run_until, 'processed_tokens': processed,
        'tokens_per_update': tokens_per_update, 'total_updates': total_updates,
        'warmup_updates': warmup, 'gradient_checkpointing': args.gradient_checkpointing,
        'tokenizer_sha256': args.tokenizer_sha256,
        'train_sampler_rng_source': 'checkpoint' if args.resume else 'fresh_seed',
        'sampler_version': SAMPLER_VERSION, 'train_identity': train_identity,
        'validation_seed': validation_seed, 'validation_identity': validation_identity,
        'source_checkpoint_sha256': args.resume_sha256 or None,
        'gpu': torch.cuda.get_device_name(0),
    }
    print(json.dumps(startup), flush=True)
    append_jsonl(metrics, startup)
    metadata = {'train_identity': train_identity, 'validation_identity': validation_identity,
                'sampler_version': SAMPLER_VERSION, 'run_contract': contract}
    def save_state():
        return checkpoint_state(model, opt, step, processed, best, args.target_tokens,
                                args.tokenizer_sha256, train_sampler_rng=rng,
                                validation_identity=validation_identity,
                                resume_metadata=metadata)

    model.train()
    start_time = last_time = time.time()
    last_tokens = invocation_start_tokens = processed
    last_eval_step = -1
    while processed < run_until:
        lr = cosine_lr(step, warmup, total_updates, args.lr)
        for group in opt.param_groups:
            group['lr'] = lr
        opt.zero_grad(set_to_none=True)
        counts = update_token_counts(run_until - processed,
                                     args.seq * args.microbatch, args.grad_accum)
        update_tokens = sum(counts)
        loss_acc = torch.zeros((), device=device, dtype=torch.float32)
        for count in counts:
            x, y = sample_batch(train, args.microbatch, args.seq, device, rng,
                                model.cfg.vocab_size, train_sampler)
            y = mask_budget_targets(y, count)
            _, loss = model(x, y)
            if loss is None:
                raise RuntimeError(f'no loss at step {step}')
            (loss * (count / update_tokens)).backward()
            loss_acc.add_(loss.detach().float() * (count / update_tokens))
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0,
                                                   error_if_nonfinite=True)
        loss_value = float(loss_acc)
        grad_value = float(grad_norm)
        if not math.isfinite(loss_value):
            raise FloatingPointError(f'non-finite loss at step {step}')
        opt.step()
        processed += update_tokens
        step += 1
        if step % args.log_every == 0 or processed >= run_until:
            torch.cuda.synchronize()
            now = time.time()
            row = {'event': 'train', 'step': step, 'processed_tokens': processed,
                   'target_tokens': args.target_tokens, 'run_until_tokens': run_until,
                   'progress_total': processed / args.target_tokens,
                   'progress_invocation': (processed - invocation_start_tokens) /
                                          max(1, run_until - invocation_start_tokens),
                   'loss': loss_value, 'lr': lr, 'grad_norm': grad_value,
                   'tokens_per_second': (processed - last_tokens) / max(1e-9, now - last_time),
                   'elapsed_s': now - start_time,
                   'gpu_mem_allocated_gb': torch.cuda.memory_allocated() / 2**30,
                   'gpu_mem_reserved_gb': torch.cuda.memory_reserved() / 2**30,
                   'gpu_peak_allocated_gb': torch.cuda.max_memory_allocated() / 2**30}
            print(json.dumps(row), flush=True)
            append_jsonl(metrics, row)
            last_time, last_tokens = now, processed
        if step % args.eval_every == 0 or processed >= run_until:
            val = evaluate(model, valid, args.seq, args.microbatch,
                           args.eval_batches, device, validation_seed)
            ev = {'event': 'validation', 'step': step, 'processed_tokens': processed,
                  'validation_loss': val, 'best_validation_loss': min(best, val),
                  'validation_seed': validation_seed,
                  'validation_identity': validation_identity}
            print(json.dumps(ev), flush=True)
            append_jsonl(metrics, ev)
            last_eval_step = step
            if val < best:
                best = val
                atomic_save({'model': model.state_dict(), 'step': step,
                             'processed_tokens': processed, 'validation_loss': val,
                             'config': FRONTIER_350.to_dict(),
                             'tokenizer_sha256': args.tokenizer_sha256},
                            out / 'best_validation.pt')
        if step % args.save_every == 0 or processed >= run_until:
            atomic_save(save_state(), out / 'resume.pt')
    if last_eval_step != step:
        val = evaluate(model, valid, args.seq, args.microbatch,
                       args.eval_batches, device, validation_seed)
        best = min(best, val)
        ev = {'event': 'validation', 'step': step, 'processed_tokens': processed,
              'validation_loss': val, 'best_validation_loss': best,
              'validation_seed': validation_seed,
              'validation_identity': validation_identity}
        print(json.dumps(ev), flush=True)
        append_jsonl(metrics, ev)
    atomic_save(save_state(), out / 'resume.pt')
    summary = {'complete': processed >= args.target_tokens,
               'paused': processed < args.target_tokens, 'parameters': report,
               'processed_tokens': processed, 'target_tokens': args.target_tokens,
               'run_until_tokens': run_until, 'steps': step,
               'best_validation_loss': best, 'elapsed_s': time.time() - start_time,
               'config': FRONTIER_350.to_dict(),
               'tokenizer_sha256': args.tokenizer_sha256,
               'sampler_version': SAMPLER_VERSION, 'train_identity': train_identity,
               'validation_identity': validation_identity,
               'checkpoint_schema': CHECKPOINT_SCHEMA,
               'train_sampler_rng_source': 'checkpoint' if args.resume else 'fresh_seed',
               'validation_seed': validation_seed}
    (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
