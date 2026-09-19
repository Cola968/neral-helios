"""Optional guarded entrypoint for an ALREADY human-approved bounded own pilot.

Runs an exhaustive read-only corpus integrity scan before delegating to the
original opt-in pilot's full preflight and actual main. Never called from CI.
Does not produce approval, freeze data, or start anything without approval.
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
    # Lazy import: the source pilot controls opt-in, manifest and original SHA.
    pilot = importlib.import_module('phoenix_weight_only_pilot')
    args = pilot.parser_args(argv)
    plan = pilot.prepare(args)  # Existing human SHA/approval/disk gates, before Torch/CUDA.
    report = verify_snapshot(args, plan)
    print(json.dumps({'event': 'approved_corpus_integrity_preflight_complete',
                      'train_shards': report['train']['shards'],
                      'validation_shards': report['validation']['shards'],
                      'token_ids_in_vocab': report['all_token_ids_in_vocab'],
                      'trainer_started': False, 'quality_gain_verified': False},
                     sort_keys=True), flush=True)
    # Deliberately re-run all original approval/source/data preflights to narrow
    # the time-of-check/use window. Does NOT promise a true immutable snapshot.
    pilot.main(argv)


if __name__ == '__main__':
    main()
