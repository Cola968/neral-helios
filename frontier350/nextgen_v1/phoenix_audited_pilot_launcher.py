"""Optional guarded entrypoint for an ALREADY human-approved bounded own pilot.

Runs exhaustive local corpus integrity AND exact holdout-overlap scans before
calling the original opt-in training pilot. No GitHub Action invokes this.
Negative exact-token results cannot certify benchmark independence.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from phoenix_candidate_lineage import SAMPLER, TOKENIZER_SHA
from phoenix_holdout_gate import require_zero_exact_overlap
from phoenix_shard_distribution_audit import inspect_shards, summarize


def corpus_digest(paths: list[Path], rows: list[dict], *, seq: int, role: str,
                  seed: int, batch: int = 0, batches: int = 0) -> str:
    """Reproduce native corpus_identity from bytes ALREADY inspected, no re-read."""
    paths = sorted(map(Path, paths))
    if len(rows) != len(paths) or role not in ('train', 'valid'):
        raise ValueError('shard audit and native file listing disagree')
    shards = [(p.name, row['tokens'], row['sha256']) for p, row in zip(paths, rows)]
    payload = {'role': role, 'shards': shards, 'seq': seq, 'seed': seed,
               'batch': batch, 'batches': batches, 'tokenizer_sha256': TOKENIZER_SHA,
               'sampler_version': SAMPLER}
    return hashlib.sha256(json.dumps(payload, sort_keys=True,
                          ensure_ascii=False).encode('utf8')).hexdigest()


def verify_snapshot(args, plan: dict) -> dict:
    """Ensure scanned full content is exactly what human-approved native preflight hashed."""
    train = inspect_shards(args.train_data, args.seq)
    valid = inspect_shards(args.valid_data, args.seq)
    aggregate = summarize(train, valid, args.seq)
    seed = plan['contract']['seed']
    train_sha = corpus_digest(plan['train_files'], train, seq=args.seq,
                              role='train', seed=seed + 1)
    valid_sha = corpus_digest(plan['valid_files'], valid, seq=args.seq,
                              role='valid', seed=seed + 999,
                              batch=args.microbatch, batches=args.eval_batches)
    if train_sha != plan['train_id'] or valid_sha != plan['val_id']:
        raise ValueError('audited shard bytes differ from previously approved native identities')
    return aggregate


def main(argv=None) -> None:
    # Lazy import: source pilot controls explicit approval, manifest and original SHA.
    pilot = importlib.import_module('phoenix_weight_only_pilot')
    args = pilot.parser_args(argv)
    plan = pilot.prepare(args)  # Original human SHA/approval/disk gates BEFORE Torch/CUDA.
    report = verify_snapshot(args, plan)
    # Mandatory for THIS launcher: refuse if the same approved frozen test
    # prompts appear verbatim in training OR validation token shards.
    overlap = require_zero_exact_overlap(args, plan)
    print(json.dumps({'event': 'approved_corpus_and_exact_overlap_preflight_complete',
                      'train_shards': report['train']['shards'],
                      'validation_shards': report['validation']['shards'],
                      'token_ids_in_vocab': report['all_token_ids_in_vocab'],
                      'exact_overlap_gate_passed': overlap['exact_overlap_gate_passed'],
                      'holdout_independence_verified': False,
                      'trainer_started': False, 'quality_gain_verified': False},
                     sort_keys=True), flush=True)
    # Re-run original preflight immediately before execution. Source files can
    # still mutate afterward: neither double scan nor hashing is a snapshot.
    pilot.main(argv)


if __name__ == '__main__':
    main()
