"""Read-only validation shard preflight; never consumes training or benchmark data.

A recorded manifest digest is evidence, not an independently frozen pin until it
is committed separately. Only valid_files are admitted; extras and symlinks fail.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

SHA = re.compile(r'[0-9a-f]{64}\Z')


def digest(path: Path) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError('non-symlink regular file required')
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inspect(root: Path, expected_tokenizer_sha: str) -> dict:
    root = Path(root)
    if not SHA.fullmatch(expected_tokenizer_sha):
        raise ValueError('invalid independently pinned tokenizer digest')
    if root.is_symlink() or not root.is_dir():
        raise ValueError('missing/symlink corpus root')
    manifest_path = root / 'manifest.json'
    manifest_sha = digest(manifest_path)
    def distinct(entries):
        result = {}
        for name, value in entries:
            if name in result:
                raise ValueError('duplicate manifest key')
            result[name] = value
        return result
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'), object_pairs_hook=distinct,
                          parse_constant=lambda x: (_ for _ in ()).throw(ValueError('invalid JSON constant')))
    if manifest.get('benchmark_content_included') is not False:
        raise ValueError('holdout separation is not attested')
    if not isinstance(manifest.get('tokenizer'), dict) or manifest['tokenizer'].get('tokenizer_sha256') != expected_tokenizer_sha:
        raise ValueError('tokenizer mismatch')
    valid_dir = root / 'valid'
    if valid_dir.is_symlink() or not valid_dir.is_dir():
        raise ValueError('validation directory missing or symlinked')
    entries = manifest.get('valid_files')
    if not isinstance(entries, list) or not entries:
        raise ValueError('empty valid_files manifest')
    seen = set()
    total = 0
    receipt = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError('invalid validation entry')
        name = entry.get('path')
        tokens = entry.get('tokens')
        sha = entry.get('sha256')
        if (not isinstance(name, str) or name in ('', '.', '..') or Path(name).name != name
                or '/' in name or '\\' in name or not name.endswith('.bin') or name in seen):
            raise ValueError('invalid, duplicate or non-basename validation shard')
        if type(tokens) is not int or tokens < 513 or not isinstance(sha, str) or not SHA.fullmatch(sha):
            raise ValueError('invalid validation shard size/digest contract')
        path = valid_dir / name
        if path.is_symlink() or not path.is_file() or path.stat().st_size != tokens * 2 or digest(path) != sha:
            raise ValueError(f'validation shard size or digest mismatch: {name}')
        seen.add(name)
        total += tokens
        receipt.append({'name': name, 'tokens': tokens, 'sha256': sha})
    if seen != {path.name for path in valid_dir.iterdir()}:
        raise ValueError('unlisted or missing validation files')
    if total != manifest.get('validation_tokens') or total < 65_537:
        raise ValueError('holdout token accounting or required sample size mismatch')
    return {'schema': 1, 'status': 'read_only_holdout_verified', 'manifest_sha256': manifest_sha,
            'tokenizer_sha256': expected_tokenizer_sha, 'validation_tokens': total,
            'validation_shards': receipt, 'holdout_only': True, 'model_training_started': False,
            'full_nis2_verified': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus-root', required=True)
    parser.add_argument('--tokenizer-sha', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    report = inspect(Path(args.corpus_root), args.tokenizer_sha)
    with Path(args.out).open('x', encoding='utf8') as output:
        json.dump(report, output, indent=2, sort_keys=True)
        output.write('\n')
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
