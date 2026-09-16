"""Read-only higher-precision validation of trusted local Neral checkpoints.

Keeps BF16 *model* weights/forward identical to production inference, but computes
cross-entropy on FP32-cast logits in <=512-token chunks. Computes one token-weighted
mean rather than averaging rounded BF16 per-batch losses. This is NOT full NIS2 or
model promotion. Only load independently trusted local torch/pickle checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from model350 import FRONTIER_350, Neral350M
from train350_compiled import PersistentMMapPool
from recovery_identity import file_sha


def protocol_sha(*, seq: int, batch: int, batches: int, seed: int, shards: list[tuple[str, str]]) -> str:
    specification = {
        'implementation_sha256': file_sha(Path(__file__)),
        'model_sha256': file_sha(Path(__file__).resolve().parents[1] / 'model350.py'),
        'sampler_sha256': file_sha(Path(__file__).resolve().parents[1] / 'train350_compiled.py'),
        'seq': seq, 'batch': batch, 'batches': batches, 'seed': seed,
        'holdout_shards_sha256': shards, 'loss': 'FP32 CE sum across <=512-token chunks / all token count',
        'model_weight_precision': 'BF16', 'model_mode': 'eval',
    }
    return hashlib.sha256(json.dumps(specification, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def run(args: argparse.Namespace) -> dict:
    if not (args.seq == 512 and args.batch == 4 and args.batches == 32 and args.seed == 351337):
        raise ValueError('FP32 diagnostic is locked to the predeclared 65536-token validation sample')
    if args.expected_tokens <= 0 or len(args.checkpoint_sha) != 64 or len(args.tokenizer_sha) != 64:
        raise ValueError('checkpoint/tokenizer independent SHA and positive token contract required')
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('CUDA BF16 GPU required')
    checkpoint = Path(args.checkpoint)
    checkpoint_before = file_sha(checkpoint)
    if checkpoint_before != args.checkpoint_sha:
        raise RuntimeError('checkpoint differs from independently pinned SHA before loading')
    pool = PersistentMMapPool(Path(args.valid_data), FRONTIER_350.vocab_size)
    shard_hashes = [(path.name, file_sha(path)) for path in pool.paths]
    if not shard_hashes:
        raise RuntimeError('empty validation shards')
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck.get('config') != FRONTIER_350.to_dict() or int(ck.get('processed_tokens', -1)) != args.expected_tokens:
        raise RuntimeError('checkpoint architecture or cumulative training token mismatch')
    if ck.get('tokenizer_sha256') != args.tokenizer_sha:
        raise RuntimeError('checkpoint tokenizer contract mismatch')
    model = Neral350M().to(device='cuda', dtype=torch.bfloat16)
    model.load_state_dict(ck['model'], strict=True)
    model.eval()
    if model.parameter_report()['total'] != 349_825_152:
        raise RuntimeError('model parameter contract mismatch')
    rng = random.Random(args.seed)
    device = torch.device('cuda')
    elapsed_start = time.perf_counter()
    per_batch: list[dict] = []
    total_nll = 0.0
    total_tokens = 0
    with torch.inference_mode():
        for index in range(args.batches):
            x, y = pool.batch(args.batch, args.seq, device, rng)
            logits, _ = model(x)
            flat = logits.reshape(-1, logits.shape[-1])
            labels = y.reshape(-1)
            batch_nll = 0.0
            for offset in range(0, labels.numel(), 512):
                end = min(labels.numel(), offset + 512)
                # Explicit .float() makes the scalar CE FP32 rather than BF16.
                nll = F.cross_entropy(flat[offset:end].float(), labels[offset:end], reduction='sum')
                batch_nll += float(nll)
            if not math.isfinite(batch_nll):
                raise FloatingPointError('non-finite FP32 validation loss')
            tokens = int(labels.numel())
            per_batch.append({'batch': index + 1, 'tokens': tokens, 'nll_sum_fp32': batch_nll,
                              'mean_nll_fp32': batch_nll / tokens})
            total_nll += batch_nll
            total_tokens += tokens
            if index % 8 == 7 or index == args.batches - 1:
                print(json.dumps({'event': 'fp32_eval_progress', 'batches_completed': index + 1,
                                  'tokens_completed': total_tokens, 'mean_loss_so_far': total_nll / total_tokens}), flush=True)
    if total_tokens != 65_536 or len(per_batch) != 32:
        raise RuntimeError('validation token accounting mismatch')
    if file_sha(checkpoint) != checkpoint_before:
        raise RuntimeError('checkpoint bytes changed during read-only evaluation')
    if [(p.name, file_sha(p)) for p in pool.paths] != shard_hashes:
        raise RuntimeError('validation shards changed during evaluation')
    loss = total_nll / total_tokens
    report = {'schema': 1, 'suite': 'neral-nextgen-holdout-fp32-v1',
              'architecture': 'frontier350-v1', 'parameters': 349_825_152,
              'checkpoint_sha256': checkpoint_before, 'checkpoint_processed_tokens': args.expected_tokens,
              'tokenizer_sha256': args.tokenizer_sha,
              'protocol_sha256': protocol_sha(seq=args.seq, batch=args.batch, batches=args.batches,
                                              seed=args.seed, shards=shard_hashes),
              'validation_loss_fp32': loss, 'validation_perplexity_fp32': math.exp(loss),
              'validation_total_nll_fp32': total_nll, 'validation_tokens': total_tokens,
              'per_batch': per_batch, 'seed': args.seed, 'seq': args.seq, 'batch': args.batch,
              'batches': args.batches, 'heldout_shard_sha256': shard_hashes,
              'wall_seconds': time.perf_counter() - elapsed_start,
              'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'read_only': True, 'training_started': False, 'promotion_allowed': False,
              'full_nis2_verified': False,
              'precision_note': 'BF16 model logits, explicitly FP32 CE per chunk; deterministic fixed-sample diagnostic only.'}
    with Path(args.out).open('x', encoding='utf8') as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    print(json.dumps({'event': 'fp32_eval_complete', 'out': args.out,
                      'tokens': total_tokens, 'loss': loss, 'perplexity': math.exp(loss),
                      'checkpoint_sha256': checkpoint_before}), flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for item in ('checkpoint', 'checkpoint-sha', 'tokenizer-sha', 'valid-data', 'out'):
        ap.add_argument('--' + item, required=True)
    ap.add_argument('--expected-tokens', type=int, required=True)
    ap.add_argument('--seq', type=int, default=512)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--batches', type=int, default=32)
    ap.add_argument('--seed', type=int, default=351337)
    run(ap.parse_args())


if __name__ == '__main__':
    main()
