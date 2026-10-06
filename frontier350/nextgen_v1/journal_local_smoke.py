"""One-shot nontraining WSL/NTFS check using tiny, self-generated torch checkpoints.

Never reads real weights, token data, tokenizer, GitHub secrets or GPU. Outputs a
numerical JSON receipt only. Leaves its own uniquely named folder untouched.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from train350 import atomic_save
from checkpoint_journal import CheckpointJournal, recover_to_new_file, sha256_file


def synthetic_state(tokens: int, step: int) -> dict:
    return {'checkpoint_schema': 2, 'processed_tokens': tokens, 'step': step,
            'model': {'mock': torch.tensor([1.0, 2.0])},
            'optimizer': {'state': {'mock': {'momentum': torch.tensor([0.1])}}},
            'torch_rng': torch.get_rng_state(), 'cuda_rng': [],
            'tokenizer_sha256': 'synthetic-not-real',
            'train_sampler_rng': ('synthetic', step)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.absolute()
    if not root.parent.is_dir() or root.exists() or args.report.exists():
        raise FileExistsError('require unused smoke folder and report; do not overwrite')
    root.mkdir()
    original = root / 'resume.pt'
    start = 1_010_003_584
    atomic_save(synthetic_state(start, 100), original)
    old_sha = sha256_file(original)
    journal = CheckpointJournal(original, initial_sha256=old_sha, initial_tokens=start,
                                writer=atomic_save)
    after = start + 999_424
    receipt = journal.commit(synthetic_state(after, 161), original)
    if journal.verify() != receipt:
        raise AssertionError('first real filesystem commit failed')
    if os.stat(original).st_ino != os.stat(journal.root / receipt['name']).st_ino:
        raise AssertionError('real drive hard-link identity missing')
    old_generation = journal.root / f'ckpt-{start}-{old_sha}.pt'
    if sha256_file(old_generation) != old_sha:
        raise AssertionError('starting checkpoint was not preserved')

    def fail_after_replace(obj: dict, path: Path) -> None:
        atomic_save(obj, path)
        raise OSError('simulated crash after replace before SHA journal receipt')

    journal.writer = fail_after_replace
    try:
        journal.commit(synthetic_state(after + 1, 162), original)
        raise AssertionError('simulated interruption was not detected')
    except OSError as error:
        if 'simulated crash' not in str(error):
            raise
    if sha256_file(original) == receipt['sha256']:
        raise AssertionError('failure was not injected after replacement')
    recovered = root / 'recovered_new_file.pt'
    restoration = recover_to_new_file(journal=journal.root, destination=recovered)
    if restoration['sha256'] != receipt['sha256']:
        raise AssertionError('recovery did not use latest committed generation')
    loaded = torch.load(recovered, map_location='cpu', weights_only=True)
    if loaded['processed_tokens'] != after or loaded['step'] != 161:
        raise AssertionError('recovered optimizer checkpoint counters are wrong')
    result = {'schema': 1, 'status': 'synthetic_torch_ntfs_crash_recovery_passed',
              'real_gpu_training': False, 'real_model_weights_loaded': False,
              'starting_tokens': start, 'committed_tokens': after,
              'max_window_tokens': 999_424, 'preserved_old_generation': True,
              'committed_sha256': receipt['sha256'],
              'failure_after_replace_detected': True, 'recovered_checkpoint_valid': True,
              'original_modified_for_training': False}
    args.report.write_text(json.dumps(result, sort_keys=True, indent=2), encoding='utf8')
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
