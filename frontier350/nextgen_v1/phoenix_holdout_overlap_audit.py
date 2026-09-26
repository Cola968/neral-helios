"""Private, read-only exact token-subsequence leakage diagnostic for Neral.

Uses a bounded Aho-Corasick automaton to scan uint16 training/validation shards
once per split. A negative result NEVER certifies holdout independence: BPE
context changes, paraphrases and non-shard sources can evade exact matching.
Does not import Torch, create a holdout, approve training or upload anything.
"""
from __future__ import annotations

import argparse
from array import array
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import sys

from phoenix_blind_ab_audit import TAG, load, validate_holdout
from phoenix_shard_distribution_audit import (
    CHUNK_BYTES, VOCAB_SIZE, _identity, _validate_uint16, summarize,
)

MAX_STATES = 300_000
MAX_PROMPT_CHARS = 1024


def build_patterns(cases, tokenizer) -> list[tuple[tuple[int, ...], str]]:
    """Raw question and full preview template; tokenizer context may still differ."""
    patterns = []
    for case in cases:
        prompt = case['prompt']
        if len(prompt) > MAX_PROMPT_CHARS:
            raise ValueError('holdout prompt exceeds model preview limit')
        for variant, text in (('raw', prompt),
                              ('chat', 'User: ' + prompt + '\nAssistant:')):
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if (not isinstance(ids, list) or not 1 <= len(ids) <= 384 or
                    any(type(v) is not int or not 0 <= v < VOCAB_SIZE for v in ids)):
                raise ValueError('invalid/oversized holdout token sequence')
            patterns.append((tuple(ids), case['id'] + ':' + variant))
    return patterns


class ExactMatcher:
    def __init__(self, patterns: list[tuple[tuple[int, ...], str]]):
        if not patterns or len(patterns) > 1000:
            raise ValueError('expected 1..1000 token patterns')
        self.next: list[dict[int, int]] = [{}]
        self.fail: list[int] = [0]
        self.output: list[set[str]] = [set()]
        self.emit: list[int] = [0]  # Nearest terminal failure suffix; avoid copied output sets.
        for ids, label in patterns:
            if (not ids or not isinstance(label, str) or not label or
                    any(type(x) is not int or not 0 <= x < VOCAB_SIZE for x in ids)):
                raise ValueError('invalid token pattern')
            state = 0
            for token in ids:
                child = self.next[state].get(token)
                if child is None:
                    if len(self.next) >= MAX_STATES:
                        raise ValueError('holdout matcher memory limit exceeded')
                    child = len(self.next)
                    self.next[state][token] = child
                    self.next.append({})
                    self.fail.append(0)
                    self.output.append(set())
                    self.emit.append(0)
                state = child
            self.output[state].add(label)
        queue = deque(self.next[0].values())
        while queue:
            state = queue.popleft()
            for token, child in self.next[state].items():
                parent = self.fail[state]
                while parent and token not in self.next[parent]:
                    parent = self.fail[parent]
                self.fail[child] = self.next[parent].get(token, 0)
                suffix = self.fail[child]
                self.emit[child] = suffix if self.output[suffix] else self.emit[suffix]
                queue.append(child)

    def feed(self, values, state: int, found: set[str]) -> int:
        for token in values:
            while state and token not in self.next[state]:
                state = self.fail[state]
            state = self.next[state].get(token, 0)
            if self.output[state]:
                found.update(self.output[state])
            suffix = self.emit[state]
            while suffix:
                found.update(self.output[suffix])
                suffix = self.emit[suffix]
        return state


def scan_split(root: Path, seq: int, matcher: ExactMatcher) -> tuple[list[dict], set[str]]:
    """Reset matcher on each shard; preserve matcher across all read chunks."""
    if type(seq) is not int or seq <= 0 or root.is_symlink() or not root.is_dir():
        raise ValueError('invalid, symlinked or absent shard root')
    files = sorted(root.glob('*.bin'))
    if not files:
        raise ValueError('no uint16 shards')
    rows, found = [], set()
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError('symlink or non-file shard refused')
        before = path.stat()
        if before.st_size % 2 or before.st_size // 2 <= seq:
            raise ValueError('invalid shard size or incomplete causal window')
        digest = hashlib.sha256()
        count = 0
        state = 0
        with path.open('rb') as source:
            if _identity(os.fstat(source.fileno())) != _identity(before):
                raise RuntimeError('shard changed before overlap scan')
            for chunk in iter(lambda: source.read(CHUNK_BYTES), b''):
                _validate_uint16(chunk)
                values = array('H')
                values.frombytes(chunk)
                if sys.byteorder != 'little':
                    values.byteswap()
                state = matcher.feed(values, state, found)
                digest.update(chunk)
                count += len(chunk)
            closed = os.fstat(source.fileno())
        if (count != before.st_size or _identity(closed) != _identity(before) or
                _identity(path.stat()) != _identity(before) or path.is_symlink()):
            raise RuntimeError('shard changed during overlap scan')
        tokens = count // 2
        rows.append({'tokens': tokens, 'windows': tokens - seq,
                     'sha256': digest.hexdigest()})
    return rows, found


def audit(cases, tokenizer, train: Path, valid: Path, seq: int) -> dict:
    patterns = build_patterns(cases, tokenizer)
    matcher = ExactMatcher(patterns)
    train_rows, train_hits = scan_split(train, seq, matcher)
    valid_rows, valid_hits = scan_split(valid, seq, matcher)
    summary = summarize(train_rows, valid_rows, seq)
    def matched_ids(hits):
        return sorted({label.rsplit(':', 1)[0] for label in hits})
    return {'schema': 1,
            'classification': 'EXACT_TOKEN_SUBSEQUENCE_DIAGNOSTIC_NOT_HOLDOUT_CLEARANCE',
            'case_count': len(cases), 'pattern_count': len(patterns),
            'train_case_ids_with_exact_hits': matched_ids(train_hits),
            'validation_case_ids_with_exact_hits': matched_ids(valid_hits),
            'train_matching_variants': len(train_hits),
            'validation_matching_variants': len(valid_hits),
            'train_shards_scanned': summary['train']['shards'],
            'validation_shards_scanned': summary['validation']['shards'],
            'full_corpus_token_ids_valid': True,
            'semantic_or_contextual_overlap_checked': False,
            'holdout_independence_verified': False,
            'training_approved': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--holdout', type=Path, required=True)
    parser.add_argument('--assets-root', type=Path, required=True)
    parser.add_argument('--train-data', type=Path, required=True)
    parser.add_argument('--valid-data', type=Path, required=True)
    parser.add_argument('--private-dir', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--seq', type=int, required=True)
    args = parser.parse_args()
    if not args.execute or not TAG.fullmatch(args.tag):
        parser.error('explicit --execute and simple private output tag required')
    from helios350_preview import TOKENIZER_SHA, trusted_asset
    root = args.private_dir
    if (root.is_symlink() or not root.is_dir() or root.resolve().name.lower() != 'phoenix'
            or root.resolve().parent.name.lower() != 'neralprivateassets'):
        parser.error('existing private NeralPrivateAssets/phoenix directory required')
    target = root / ('phoenix_overlap_' + args.tag + '.json')
    if target.exists() or target.is_symlink():
        raise FileExistsError('private overlap report must not be overwritten')
    holdout, holdout_sha = load(args.holdout)
    cases = validate_holdout(holdout)
    tokenizer_file = trusted_asset(args.assets_root,
        args.assets_root / 'frontier350_assets' / 'tokenizer.json', TOKENIZER_SHA)
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    if tokenizer.get_vocab_size() != VOCAB_SIZE:
        raise ValueError('wrong own tokenizer vocabulary')
    result = audit(cases, tokenizer, args.train_data, args.valid_data, args.seq)
    result['holdout_sha256'] = holdout_sha
    result['tokenizer_sha256'] = TOKENIZER_SHA
    with target.open('x', encoding='utf-8') as output:
        json.dump(result, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write('\n')
    # Aggregate only in terminal. Detailed matching IDs remain private.
    print(json.dumps({'event': 'private_exact_overlap_audit_complete',
                      'cases': len(cases),
                      'train_cases_with_hits': len(result['train_case_ids_with_exact_hits']),
                      'validation_cases_with_hits': len(result['validation_case_ids_with_exact_hits']),
                      'holdout_independence_verified': False,
                      'training_approved': False}, sort_keys=True))


if __name__ == '__main__':
    main()
