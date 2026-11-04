"""Verify pinned local Qwen HF snapshot without loading neural weights.

Uses only stdlib and model_catalog.py from the immutable trainer checkout.
Never updates or downloads a file and never starts model inference/training.
"""
from __future__ import annotations

import argparse
from pathlib import Path

BASE = 'Qwen/Qwen2.5-Coder-7B-Instruct'


def verify_snapshot(folder: Path, validate):
    """Fail closed: check metadata before streaming every file SHA-256 hash."""
    folder = Path(folder)
    metadata = validate(folder, verify_hashes=False)
    if metadata.get('base_model') != BASE:
        raise ValueError('Unexpected Qwen base model; no training permitted')
    print('SNAPSHOT_METADATA_VERIFIED', flush=True)
    verified = validate(folder, verify_hashes=True)
    if verified.get('base_model') != BASE:
        raise ValueError('Unexpected base model after hash verification')
    if metadata.get('sha256') != verified.get('sha256'):
        raise ValueError('Snapshot manifest changed during hash verification')
    print('SNAPSHOT_ALL_SHA256_VERIFIED', flush=True)
    return verified


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--snapshot', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        from model_catalog import validate_hf_snapshot
        verified = verify_snapshot(args.snapshot, validate_hf_snapshot)
        print('SNAPSHOT_REVISION=' + verified['revision'], flush=True)
        return 0
    except Exception as exc:
        # Private job logs: no paths, tracebacks, prompts or credentials.
        print('SNAPSHOT_FAILURE_TYPE=' + type(exc).__name__, flush=True)
        print('SNAPSHOT_FAILURE_REASON=' + str(exc).replace('\n', ' ').replace('\r', ' ')[:160], flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
