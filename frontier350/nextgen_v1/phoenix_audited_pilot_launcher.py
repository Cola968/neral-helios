"""Compatibility launcher for the SHA-approved Neral 350M weight-only pilot.

The native phoenix_weight_only_pilot.main now enforces BOTH full-corpus identity
and frozen-holdout exact-overlap checks before GPU import, for direct starts and
resumes. Delegate to it exactly once: the former wrapper repeated both full
corpus scans unnecessarily. No approval, CUDA use or training occurs on import.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from phoenix_candidate_lineage import SAMPLER, TOKENIZER_SHA
from phoenix_shard_distribution_audit import inspect_shards, summarize


def corpus_digest(paths: list[Path], rows: list[dict], *, seq: int, role: str,
                  seed: int, batch: int = 0, batches: int = 0) -> str:
    """Reproduce native corpus_identity from already inspected bytes."""
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
    """Verify fully scanned shard bytes match the approved native identities."""
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
    """Call the guarded native trainer once; never bypass its own preflight."""
    pilot = importlib.import_module('phoenix_weight_only_pilot')
    pilot.main(argv)


if __name__ == '__main__':
    main()
