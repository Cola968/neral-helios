"""Restore exact reviewed Git bytes in a SHA-pinned scratch checkout only.

Windows may convert LF -> CRLF on checkout. All reviewed SHA-256 hashes
are checked against immutable Git objects before any write. Only exact
CRLF-equivalent text may be restored; unexpected changes fail closed.
No model loads, network, training or external user-file modifications.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess

PIN = 'a082738f9145722dec443141e5d7801a3ead93af'
LOCK = 'datasets/neral_phase22_corpus.lock.json'
REVIEWS = (
    'datasets/neral_qlora_pilot.review.json',
    'datasets/neral_project_sft_v1.review.json',
    'datasets/neral_capabilities_sft_v1.review.json',
)
MAX_FILE = 4 * 1024 * 1024
SHA = re.compile(r'[a-f0-9]{64}\Z')


class PinError(ValueError):
    pass


def git(root: Path, *args: str) -> bytes:
    try:
        return subprocess.check_output(['git', '-C', str(root), *args],
                                       stderr=subprocess.PIPE)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PinError('Pinned Git object unavailable') from exc


def _relative(name: str) -> PurePosixPath:
    if not isinstance(name, str) or '\\' in name or ':' in name:
        raise PinError('Invalid pinned relative filename')
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or any(part in ('', '.', '..') for part in p.parts):
        raise PinError('Invalid pinned relative filename')
    return p


def _digest(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _object(root: Path, relative: str) -> bytes:
    data = git(root, 'show', f'HEAD:{relative}')
    if len(data) > MAX_FILE:
        raise PinError('Pinned data exceeds configured file budget')
    return data


def restore_selected(root: Path, expected: dict[str, str]) -> int:
    """Validate every blob/worktree pair before restoring any CRLF-only drift."""
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or root.resolve() != root.absolute():
        raise PinError('Not a regular scratch checkout')
    if not isinstance(expected, dict) or len(expected) > 128:
        raise PinError('Invalid pinned file set')
    planned = []
    for relative, digest in sorted(expected.items()):
        p = _relative(relative)
        if not isinstance(digest, str) or not SHA.fullmatch(digest):
            raise PinError('Invalid expected SHA-256')
        target = root.joinpath(*p.parts)
        if target.is_symlink() or not target.is_file() or target.stat().st_size > MAX_FILE:
            raise PinError('Pinned checkout file missing, symlinked or oversized')
        original = _object(root, relative)
        if _digest(original) != digest:
            raise PinError('Reviewed lock does not match pinned Git blob: ' + relative)
        local = target.read_bytes()
        if local == original:
            continue
        if local.replace(b'\r\n', b'\n') != original:
            raise PinError('Unexpected checkout mutation beyond Windows EOL: ' + relative)
        planned.append((target, original))
    for target, original in planned:
        target.write_bytes(original)
        if target.read_bytes() != original:
            raise PinError('Unable to restore exact Git blob bytes')
    return len(planned)


def restore_pinned(root: Path) -> dict:
    root = Path(root)
    head = git(root, 'rev-parse', 'HEAD').decode('ascii').strip()
    if head != PIN:
        raise PinError('Pinned trainer HEAD differs from authorized commit')
    lock = json.loads(_object(root, LOCK))
    if not isinstance(lock, dict) or not isinstance(lock.get('frozen'), dict) or not isinstance(lock.get('files'), dict):
        raise PinError('Invalid pinned corpus lock metadata')
    expected = {LOCK: _digest(_object(root, LOCK))}
    for name, digest in {**lock['frozen'], **lock['files']}.items():
        _relative(name)
        expected['datasets/' + name] = digest
    for review in REVIEWS:
        reviewed = json.loads(_object(root, review))
        expected[review] = _digest(_object(root, review))
        if not isinstance(reviewed, dict) or not isinstance(reviewed.get('sources'), dict):
            raise PinError('Review sources are missing')
        for path, digest in reviewed['sources'].items():
            _relative(path)
            if path in expected and expected[path] != digest:
                raise PinError('Conflicting reviewed source hash')
            expected[path] = digest
    changed = restore_selected(root, expected)
    return {'status': 'PINNED_GIT_DATA_VERIFIED', 'files_checked': len(expected),
            'lf_files_restored': changed, 'training_executed': False}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--repo-root', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(restore_pinned(args.repo_root), sort_keys=True))
        return 0
    except (PinError, ValueError, OSError, KeyError, UnicodeError) as exc:
        print(f'PINNED_CORPUS_REFUSED={type(exc).__name__}: {str(exc)[:180]}')
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
