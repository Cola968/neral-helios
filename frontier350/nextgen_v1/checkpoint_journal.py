"""Local same-volume checkpoint journal. No model import and no automatic recovery.

A fully resumable state must be supplied by the caller; this module never treats a
weights-only snapshot as a recovery point. Each committed generation is hard-linked
from an atomically replaced resume.pt, so retaining it is O(1) in checkpoint size.
The link protects against overwrites, NOT a drive failure, deletion, or bad RAM.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Callable

MAX_ROLLBACK_TOKENS = 1_000_000
_NAME = re.compile(r"^ckpt-([0-9]+)-([a-f0-9]{64})\.pt$")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def safe_save_every(*, seq: int, microbatch: int, accumulation: int,
                    max_rollback_tokens: int = MAX_ROLLBACK_TOKENS,
                    requested: int | None = None) -> int:
    """A completed update is indivisible; refuse an update larger than the budget."""
    if (not 0 < max_rollback_tokens <= MAX_ROLLBACK_TOKENS or
            min(seq, microbatch, accumulation) <= 0):
        raise ValueError('invalid checkpoint rollback or optimizer dimensions')
    tokens_per_update = seq * microbatch * accumulation
    interval = max_rollback_tokens // tokens_per_update
    if interval < 1:
        raise ValueError('one optimizer update exceeds the rollback token limit')
    if requested is not None and requested <= 0:
        raise ValueError('checkpoint interval must be positive')
    return min(interval, requested) if requested is not None else interval


def _sync_directory(path: Path) -> None:
    if os.name == 'nt':
        return  # Windows does not permit opening directories with ordinary Python open().
    try:
        handle = os.open(str(path), os.O_RDONLY)
    except OSError:
        return  # Directory fsync is unavailable on some Windows-backed WSL mounts.
    try:
        try:
            os.fsync(handle)
        except OSError:
            pass
    finally:
        os.close(handle)


def _atomic_receipt(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + '.tmp')
    if temporary.exists():
        raise FileExistsError(f'stale receipt temp file: {temporary}')
    try:
        with temporary.open('x', encoding='utf8') as output:
            json.dump(data, output, sort_keys=True, separators=(',', ':'))
            output.write('\n')
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        # Never erase an unknown file; only remove the temp created by this call.
        if temporary.exists():
            temporary.unlink()


def _checked_generation(root: Path, receipt: dict) -> Path:
    name = receipt.get('name')
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError('invalid checkpoint generation name')
    match = _NAME.fullmatch(name)
    if int(match.group(1)) != receipt.get('processed_tokens') or match.group(2) != receipt.get('sha256'):
        raise ValueError('generation name does not match receipt')
    path = root / name
    if not path.is_file() or sha256_file(path) != receipt['sha256']:
        raise ValueError('committed checkpoint is absent or corrupt')
    return path


class CheckpointJournal:
    """Append-only checkpoint generations and SHA-pinned CURRENT receipt.

    ``writer`` must atomically write a complete checkpoint to resume.pt; the
    trainer's original torch-based atomic_save is used unchanged. Caller checks
    optimizer/sampler/RNG/architecture before accepting an input checkpoint.
    """

    def __init__(self, resume: Path, *, initial_sha256: str, initial_tokens: int,
                 writer: Callable[[dict, Path], None],
                 max_rollback_tokens: int = MAX_ROLLBACK_TOKENS, retain: int = 3):
        self.resume = Path(resume).resolve()
        self.root = self.resume.parent / 'checkpoint_journal'
        if self.resume.name != 'resume.pt':
            raise ValueError('journal only supports resume.pt, not weights-only snapshots')
        if not 0 < max_rollback_tokens <= MAX_ROLLBACK_TOKENS or retain < 2:
            raise ValueError('rollback must be <=1M and >=2 generations retained')
        if not re.fullmatch(r'[0-9a-f]{64}', initial_sha256):
            raise ValueError('initial checkpoint SHA-256 required')
        if type(initial_tokens) is not int or initial_tokens < 0:
            raise ValueError('invalid initial token counter')
        if not self.resume.is_file() or sha256_file(self.resume) != initial_sha256:
            raise ValueError('starting resume.pt does not match the pinned SHA-256')
        self.writer = writer
        self.max_gap = max_rollback_tokens
        self.retain = retain
        self.root.mkdir(exist_ok=True)
        latest = self.root / 'latest.json'
        if latest.exists():
            receipt = json.loads(latest.read_text(encoding='utf8'))
            _checked_generation(self.root, receipt)
            if (receipt.get('sha256') != initial_sha256 or
                    receipt.get('processed_tokens') != initial_tokens):
                raise ValueError('resume.pt differs from latest committed journal; recover explicitly')
        else:
            if any(self.root.iterdir()):
                raise RuntimeError('unknown journal contents; refuse bootstrap')
            # SHA identity alone does NOT establish optimizer completeness. The
            # guarded caller checks that separately; generic journal users must
            # not get an unsupported 'complete_resumable_state' label at bootstrap.
            receipt = self._archive(initial_tokens, initial_sha256,
                                    kind='pinned_initial_structure_unverified')
            _atomic_receipt(latest, receipt)
        self.last_tokens = initial_tokens
        self.last_sha = initial_sha256

    def _archive(self, tokens: int, digest: str, *,
                 kind: str = 'complete_resumable_state') -> dict:
        if kind not in ('complete_resumable_state', 'pinned_initial_structure_unverified'):
            raise ValueError('invalid checkpoint receipt classification')
        name = f'ckpt-{tokens}-{digest}.pt'
        dest = self.root / name
        if dest.exists():
            raise FileExistsError(f'checkpoint generation already exists: {dest}')
        # Same-volume hard link: no multi-GB copy or extra GPU synchronization.
        # Never silently fall back to a mutable pointer or an unverified copy.
        os.link(self.resume, dest)
        if sha256_file(dest) != digest:
            raise IOError('new checkpoint generation failed independent SHA verification')
        _sync_directory(self.root)
        return {'schema': 1, 'name': name, 'sha256': digest,
                'processed_tokens': tokens, 'bytes': dest.stat().st_size,
                'kind': kind}

    def commit(self, state: dict, target: Path) -> dict:
        if Path(target).resolve() != self.resume:
            self.writer(state, Path(target))  # For example, best_validation.pt.
            return {'journaled': False}
        if not isinstance(state, dict):
            raise ValueError('checkpoint is not a state dictionary')
        tokens = state.get('processed_tokens')
        if type(tokens) is not int or not self.last_tokens < tokens <= self.last_tokens + self.max_gap:
            raise ValueError('checkpoint must advance by <=1M tokens, with no skip')
        if not isinstance(state.get('model'), dict) or not isinstance(state.get('optimizer'), dict):
            raise ValueError('checkpoint lacks model or optimizer state')
        # Previously {'optimizer': {'state': {}}} passed and received a misleading
        # 'complete_resumable_state' receipt. Reject BEFORE any checkpoint write.
        moments = state['optimizer'].get('state')
        if not isinstance(moments, dict) or not moments:
            raise ValueError('checkpoint lacks nonempty AdamW optimizer moments; refuse journal receipt')
        if not all(key in state for key in ('step', 'torch_rng', 'cuda_rng', 'tokenizer_sha256')):
            raise ValueError('checkpoint lacks step, RNG or tokenizer identity')
        if 'train_sampler_rng' not in state and 'v9_training_state' not in state:
            raise ValueError('checkpoint lacks resumable sampler state')
        self.writer(state, self.resume)
        with self.resume.open('rb') as completed:
            os.fsync(completed.fileno())
        digest = sha256_file(self.resume)
        receipt = self._archive(tokens, digest)
        _atomic_receipt(self.root / 'latest.json', receipt)
        self.last_tokens, self.last_sha = tokens, digest
        # Non-destructive retention: leave old generations until separately
        # audited disk-pressure management. Never delete a known-good state here.
        return receipt

    def verify(self) -> dict:
        latest = json.loads((self.root / 'latest.json').read_text(encoding='utf8'))
        _checked_generation(self.root, latest)
        if sha256_file(self.resume) != latest['sha256']:
            raise ValueError('resume.pt changed after commit; recover from journal explicitly')
        return latest


def recover_to_new_file(*, journal: Path, destination: Path) -> dict:
    """Explicit non-overwriting recovery; never mutates a running trainer directory."""
    root = Path(journal).resolve()
    receipt = json.loads((root / 'latest.json').read_text(encoding='utf8'))
    source = _checked_generation(root, receipt)
    dest = Path(destination)
    if dest.exists():
        raise FileExistsError('restore destination must not exist')
    if not dest.parent.is_dir():
        raise FileNotFoundError('create a separate recovery directory first')
    # Hard linking is O(1) on the same drive. Cross-drive recovery is explicit
    # operator work, rather than silently copying several gigabytes.
    os.link(source, dest)
    if sha256_file(dest) != receipt['sha256']:
        raise IOError('recovered file did not pass SHA verification')
    return {'recovered': str(dest), **receipt}
