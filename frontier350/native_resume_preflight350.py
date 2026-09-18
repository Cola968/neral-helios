"""Fail-closed CPU preflight for BOTH native 350M trainer entrypoints.

No model construction or CUDA allocation. Caller must invoke before torch.cuda.*,
creating an output directory or constructing an optimizer. The full SHA pins
trusted local pickle bytes. A model-only checkpoint is NEVER an exact resume.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from recovery_identity import file_sha, object_sha
from resume_integrity350 import require_full_adam_resume
from window_sampler350 import SAMPLER_VERSION

_SHA = re.compile(r'[0-9a-f]{64}\Z')


def corpus_identity(paths, *, seq: int, role: str, seed: int, batch: int = 0,
                    batches: int = 0, tokenizer_sha: str) -> str:
    if role not in ('train', 'valid') or not _SHA.fullmatch(tokenizer_sha):
        raise ValueError('corpus role and exact tokenizer SHA required')
    if not paths or type(seq) is not int or seq <= 0:
        raise ValueError('corpus identity requires valid sequence and shards')
    shards = []
    for path in sorted(map(Path, paths)):
        if not path.is_file() or path.is_symlink():
            raise ValueError('missing or symlinked corpus shard')
        if path.stat().st_size % 2 or path.stat().st_size // 2 <= seq:
            raise ValueError('corrupt, too short, or misaligned uint16 shard')
        shards.append((path.name, path.stat().st_size // 2, file_sha(path)))
    if len({name for name, _, _ in shards}) != len(shards):
        raise ValueError('duplicate corpus shard names')
    return object_sha({'role': role, 'shards': shards, 'seq': seq,
                       'seed': seed, 'batch': batch, 'batches': batches,
                       'tokenizer_sha256': tokenizer_sha,
                       'sampler_version': SAMPLER_VERSION})


def run_contract(*, seq: int, microbatch: int, grad_accum: int, seed: int,
                 lr: float, schedule: str, weight_decay: float,
                 warmup_frac: float, gradient_checkpointing: bool) -> dict:
    return {'seq': seq, 'microbatch': microbatch, 'grad_accum': grad_accum,
            'seed': seed, 'lr': lr, 'schedule': schedule,
            'weight_decay': weight_decay, 'warmup_frac': warmup_frac,
            'gradient_checkpointing': bool(gradient_checkpointing)}


def require_exact_update_boundary(*, processed: int, step: int, target_tokens: int,
                                  run_until: int, run: dict) -> int:
    """Reject a purported exact resume across a shortened optimizer update.

    The native loop clips the last optimizer update to run_until. An intermediate
    stop inside an update would introduce an *extra* AdamW step versus an
    uninterrupted run. Such a checkpoint may be a deliberate completed run,
    but must never be called an exact continuation of the original trajectory.
    The terminal target itself may have a remainder; no resume follows it.
    """
    if any(type(run.get(k)) is not int or run[k] <= 0 for k in
           ('seq', 'microbatch', 'grad_accum')):
        raise ValueError('exact resume requires positive integer update dimensions')
    tokens_per_update = run['seq'] * run['microbatch'] * run['grad_accum']
    if (type(processed) is not int or type(step) is not int or step <= 0 or
            type(target_tokens) is not int or target_tokens <= 0 or
            type(run_until) is not int or not processed < run_until <= target_tokens):
        raise ValueError('invalid or nonadvancing exact-resume token boundary')
    if processed != step * tokens_per_update:
        raise ValueError('checkpoint ended in a partial optimizer update; cannot exactly resume')
    if run_until < target_tokens and run_until % tokens_per_update:
        raise ValueError('intermediate stop would create a partial optimizer update; exact resume forbidden')
    return tokens_per_update


def verify_weight_moment_mapping(ck: dict, *, expected_run: dict) -> None:
    """Map tied native parameters to Adam slots; reject state_dict option overrides.

    Optimizer.load_state_dict restores parameter-group options such as betas and
    weight decay, silently overriding the newly constructed optimizer. The
    metadata run_contract alone cannot protect against a divergent group.
    """
    model, optimizer = ck['model'], ck['optimizer']
    if len(optimizer['param_groups']) != 1 or 'tok_emb.weight' not in model or 'lm_head.weight' not in model:
        raise ValueError('only the native tied-head, single-group AdamW layout is supported')
    if model['tok_emb.weight'].shape != model['lm_head.weight'].shape:
        raise ValueError('tied embedding / head shape mismatch')
    group = optimizer['param_groups'][0]
    if (tuple(group.get('betas', ())) != (0.9, 0.95) or
            group.get('eps') != 1e-8 or
            group.get('weight_decay') != expected_run['weight_decay'] or
            group.get('fused') is not True or
            group.get('amsgrad', False) is not False or
            group.get('maximize', False) is not False):
        raise ValueError('serialized AdamW group hyperparameters differ from native run contract')
    saved_lr = group.get('lr')
    peak_lr = expected_run['lr']
    if (type(saved_lr) not in (int, float) or not math.isfinite(saved_lr) or
            not 0 < saved_lr <= peak_lr * (1 + 1e-9) or
            (expected_run['schedule'] == 'constant' and saved_lr != peak_lr)):
        raise ValueError('serialized AdamW learning rate differs from native schedule bounds')
    ordered = [(name, tensor) for name, tensor in model.items() if name != 'lm_head.weight']
    ids = group['params']
    if len(ordered) != len(ids):
        raise ValueError('AdamW slot count differs from native named parameters')
    for ident, (name, tensor) in zip(ids, ordered):
        if not hasattr(tensor, 'shape') or not hasattr(tensor, 'dtype'):
            raise ValueError('non-tensor model parameter')
        slot = optimizer['state'][ident]
        if (tuple(slot['exp_avg'].shape) != tuple(tensor.shape) or
                tuple(slot['exp_avg_sq'].shape) != tuple(tensor.shape) or
                slot['exp_avg'].dtype != tensor.dtype or
                slot['exp_avg_sq'].dtype != tensor.dtype):
            raise ValueError(f'AdamW moment shape/dtype disagrees with parameter: {name}')


def inspect_checkpoint(ck: dict, *, config: dict, target_tokens: int,
                       tokenizer_sha: str, train_identity: str,
                       validation_identity: str, expected_run: dict,
                       run_until: int) -> dict:
    if ck.get('config') != config or ck.get('checkpoint_schema') != 2:
        raise ValueError('wrong native checkpoint architecture/schema')
    if type(ck.get('target_tokens')) is not int or ck['target_tokens'] != target_tokens:
        raise ValueError('different cumulative token/LR target')
    if ck.get('tokenizer_sha256') != tokenizer_sha or not _SHA.fullmatch(tokenizer_sha):
        raise ValueError('tokenizer SHA missing or changed')
    if (ck.get('train_identity') != train_identity or
            ck.get('validation_identity') != validation_identity or
            ck.get('sampler_version') != SAMPLER_VERSION or
            ck.get('run_contract') != expected_run):
        raise ValueError('dataset, window sampler, validation or optimizer schedule changed; exact resume forbidden')
    result = require_full_adam_resume(ck)
    verify_weight_moment_mapping(ck, expected_run=expected_run)
    require_exact_update_boundary(processed=ck['processed_tokens'], step=ck['step'],
                                  target_tokens=target_tokens, run_until=run_until,
                                  run=expected_run)
    return result


def read_verified_resume(*, checkpoint: Path, expected_sha: str, trusted: bool,
                         out: Path, torch, config: dict, target_tokens: int,
                         tokenizer_sha: str, train_identity: str,
                         validation_identity: str, expected_run: dict,
                         run_until: int):
    """Load only the exact trusted SHA-pinned local checkpoint onto CPU.

    Returns the already validated state for later model/optimizer loading.
    No fallback/recovery/warm start occurs implicitly.
    """
    path = Path(checkpoint)
    out = Path(out)
    if not trusted or not isinstance(expected_sha, str) or not _SHA.fullmatch(expected_sha):
        raise ValueError('exact resume requires trusted local checkpoint and full --resume-sha256')
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError('resume path must be absolute, existing and non-symlinked')
    if path.resolve() != (out.resolve() / 'resume.pt'):
        raise ValueError('resume must equal resume.pt inside its own output directory')
    if file_sha(path) != expected_sha:
        raise ValueError('resume SHA mismatch; refused before pickle deserialization')
    # Trusted own pickle only. Legacy RNG layout can require weights_only=False.
    ck = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    inspect_checkpoint(ck, config=config, target_tokens=target_tokens,
                       tokenizer_sha=tokenizer_sha, train_identity=train_identity,
                       validation_identity=validation_identity,
                       expected_run=expected_run, run_until=run_until)
    return ck
