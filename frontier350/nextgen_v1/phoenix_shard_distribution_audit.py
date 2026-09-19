"""Read-only audit of uint16 shard-window sampling distribution.

Quantifies theoretical bias of historic shard-uniform sampling versus the new
window-uniform sampler. Does NOT infer historical draws, model quality, or
training throughput. Never reads model weights or writes shard contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def inspect_shards(root: Path, seq: int, *, hash_bytes: bool = True) -> list[dict]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError('shard root must be an existing non-symlink directory')
    shards = sorted(root.glob('*.bin'))
    if not shards:
        raise ValueError('no .bin shards found')
    result = []
    for path in shards:
        if path.is_symlink() or not path.is_file():
            raise ValueError('symlink or non-file shard refused')
        size = path.stat().st_size
        if size % 2:
            raise ValueError('odd-byte uint16 shard')
        token_count = size // 2
        windows = token_count - seq
        if windows <= 0:
            raise ValueError('shard has no complete seq+1 window')
        digest = None
        if hash_bytes:
            h = hashlib.sha256()
            with path.open('rb') as stream:
                for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    h.update(chunk)
            if path.stat().st_size != size:
                raise RuntimeError('shard changed size during audit')
            digest = h.hexdigest()
        result.append({'tokens': token_count, 'windows': windows, 'sha256': digest})
    return result


def summarize(train: list[dict], valid: list[dict], seq: int) -> dict:
    if type(seq) is not int or seq <= 0:
        raise ValueError('positive integer sequence length required')
    if not train or not valid:
        raise ValueError('both datasets required')
    for shards in (train, valid):
        for row in shards:
            if (type(row['tokens']) is not int or type(row['windows']) is not int or
                    row['tokens'] - seq != row['windows'] or row['windows'] <= 0):
                raise ValueError('invalid shard window counts')
    train_hashes = [row['sha256'] for row in train]
    valid_hashes = [row['sha256'] for row in valid]
    if any(not isinstance(sha, str) or len(sha) != 64 for sha in train_hashes + valid_hashes):
        raise ValueError('full shard hashes mandatory for cross-split audit')
    if len(set(train_hashes)) != len(train_hashes) or len(set(valid_hashes)) != len(valid_hashes):
        raise ValueError('duplicate identical shard bytes within a dataset')
    if set(train_hashes) & set(valid_hashes):
        raise ValueError('identical shard bytes across training and validation')

    def distribution(shards: list[dict]) -> dict:
        total = sum(row['windows'] for row in shards)
        n = len(shards)
        worst_over = max(total / (n * row['windows']) for row in shards)
        worst_under = min(total / (n * row['windows']) for row in shards)
        # Total variation distance between historic shard-uniform and
        # desired window-uniform distributions over individual start indices.
        tv = 0.5 * sum(abs(1 / n - row['windows'] / total) for row in shards)
        return {'shards': n, 'tokens_in_shards': sum(row['tokens'] for row in shards),
                'valid_windows': total, 'smallest_shard_windows': min(row['windows'] for row in shards),
                'largest_shard_windows': max(row['windows'] for row in shards),
                'max_old_sampling_overrepresentation_factor': worst_over,
                'min_old_sampling_factor': worst_under,
                'old_vs_window_uniform_total_variation': tv}

    return {'schema': 1, 'classification': 'READ_ONLY_SAMPLING_DISTRIBUTION_AUDIT',
            'seq': seq, 'train': distribution(train), 'validation': distribution(valid),
            'byte_identical_split_overlap': False,
            'semantic_overlap_checked': False, 'historical_sampling_counts_known': False,
            'quality_gain_proven': False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train-data', required=True, type=Path)
    p.add_argument('--valid-data', required=True, type=Path)
    p.add_argument('--seq', required=True, type=int)
    args = p.parse_args()
    if args.seq <= 0:
        p.error('sequence length must be positive')
    train = inspect_shards(args.train_data, args.seq)
    valid = inspect_shards(args.valid_data, args.seq)
    report = summarize(train, valid, args.seq)
    # Aggregate metrics ONLY: no paths, token contents, shard hashes or answers.
    print(json.dumps(report, sort_keys=True))


if __name__ == '__main__':
    main()
