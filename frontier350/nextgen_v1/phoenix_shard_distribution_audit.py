"""Read-only audit of uint16 shard-window sampling distribution.

Quantifies theoretical bias of historic shard-uniform sampling versus the new
window-uniform sampler. Does NOT infer historical draws, model quality, or
training throughput. Never reads model weights or writes shard contents.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from array import array
from pathlib import Path


# Own Frontier 350 config: 16,384 vocabulary entries, IDs 0..16,383.
# Hard-code the contract here so this read-only audit never imports Torch.
VOCAB_SIZE = 16_384
SHA256 = re.compile(r'[0-9a-f]{64}\Z')
CHUNK_BYTES = 4 * 1024 * 1024  # Even size preserves uint16 alignment.


def _identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _validate_uint16(chunk: bytes) -> None:
    if len(chunk) % 2:
        raise ValueError('odd-byte uint16 shard chunk')
    values = array('H')
    values.frombytes(chunk)
    if sys.byteorder != 'little':
        values.byteswap()
    if values and max(values) >= VOCAB_SIZE:
        raise ValueError('out-of-vocabulary uint16 token ID in shard')


def inspect_shards(root: Path, seq: int, *, hash_bytes: bool = True) -> list[dict]:
    if type(seq) is not int or seq <= 0:
        raise ValueError('positive integer sequence length required')
    if root.is_symlink() or not root.is_dir():
        raise ValueError('shard root must be an existing non-symlink directory')
    shards = sorted(root.glob('*.bin'))
    if not shards:
        raise ValueError('no .bin shards found')
    result = []
    for path in shards:
        if path.is_symlink() or not path.is_file():
            raise ValueError('symlink or non-file shard refused')
        before = path.stat()
        size = before.st_size
        if size % 2:
            raise ValueError('odd-byte uint16 shard')
        token_count = size // 2
        windows = token_count - seq
        if windows <= 0:
            raise ValueError('shard has no complete seq+1 window')
        h = hashlib.sha256() if hash_bytes else None
        bytes_read = 0
        # SHA and full-vocabulary scan are the SAME read, not disjoint passes.
        with path.open('rb') as stream:
            opened = os.fstat(stream.fileno())
            if _identity(opened) != _identity(before):
                raise RuntimeError('shard replaced before read')
            for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
                _validate_uint16(chunk)
                bytes_read += len(chunk)
                if h is not None:
                    h.update(chunk)
            closed = os.fstat(stream.fileno())
        after = path.stat()
        if (bytes_read != size or _identity(closed) != _identity(before) or
                _identity(after) != _identity(before) or path.is_symlink()):
            raise RuntimeError('shard changed during audit; refuse results')
        digest = h.hexdigest() if h is not None else None
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
    if any(not isinstance(sha, str) or not SHA256.fullmatch(sha)
           for sha in train_hashes + valid_hashes):
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
            'all_token_ids_in_vocab': True, 'file_metadata_stable_during_read': True,
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
