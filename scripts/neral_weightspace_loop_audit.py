"""Offline, privacy-preserving Qwen2.5 baseline loop triage for NERAL.

Accepts only output token IDs, digests, and termination metadata. It cannot
identify a defective neuron or justify a weight edit; a high repetition score
is only a manually reviewable hypothesis. No network/GPU/models/files written.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

EOS_QWEN25 = 151645
BASE_MODEL = 'Qwen/Qwen2.5-Coder-7B-Instruct'
MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 2000
MAX_GENERATION_TOKENS = 512
EXPECTED = {
    'task_id', 'base_model_id', 'base_revision', 'prompt_sha256',
    'tokenizer_sha256', 'generation_config_sha256', 'adapter_sha256',
    'max_new_tokens', 'finish_reason', 'output_token_ids',
}
TERMINATIONS = {'eos', 'length', 'cancelled', 'failed'}
SHA256 = re.compile(r'[a-f0-9]{64}\Z')
COMMIT = re.compile(r'[a-f0-9]{40}\Z')
TASK = re.compile(r'[a-z0-9][a-z0-9_-]{2,63}\Z')


class AuditError(ValueError):
    """Unverifiable baseline record; never silently score it as complete."""


def _unique_object(items):
    result = {}
    for key, value in items:
        if key in result:
            raise AuditError('duplicate JSON object key')
        result[key] = value
    return result


def read_audit(path: str | Path) -> list[dict]:
    """Load a bounded JSONL from local storage; no prompt/text content allowed."""
    path = Path(path)
    if not path.is_file() or path.is_symlink() or path.stat().st_size > MAX_BYTES:
        raise AuditError('missing, symlinked, or oversized local audit file')
    try:
        content = path.read_text(encoding='utf-8')
    except (OSError, UnicodeError) as exc:
        raise AuditError('cannot read local audit file') from exc
    if not content or not content.endswith('\n'):
        raise AuditError('JSONL must be nonempty and end with a newline')
    lines = content.splitlines()
    if len(lines) > MAX_ROWS:
        raise AuditError('too many audit records')
    rows = []
    for line in lines:
        if len(line) > 8192:
            raise AuditError('oversized audit record')
        try:
            record = json.loads(line, object_pairs_hook=_unique_object,
                                parse_constant=lambda _: (_ for _ in ()).throw(AuditError('nonfinite JSON')))
        except (ValueError, TypeError) as exc:
            raise AuditError('invalid JSONL record') from exc
        rows.append(record)
    return rows


def repetition_fraction(tokens: list[int], *, n: int = 4) -> float:
    """Repeated n-grams / all n-grams; repetition alone is NOT a reasoning loop."""
    if type(n) is not int or not 2 <= n <= 32:
        raise AuditError('invalid n-gram window')
    if not isinstance(tokens, list) or not all(type(x) is int and 0 <= x < 200000 for x in tokens):
        raise AuditError('invalid output token IDs')
    if len(tokens) < n:
        return 0.0
    total = len(tokens) - n + 1
    unique = {tuple(tokens[k:k + n]) for k in range(total)}
    return (total - len(unique)) / total


def _validated(row: dict) -> dict:
    if not isinstance(row, dict) or set(row) != EXPECTED:
        raise AuditError('unexpected fields: only token IDs, hashes and completion metadata allowed')
    if not isinstance(row['task_id'], str) or not TASK.fullmatch(row['task_id']):
        raise AuditError('invalid task ID')
    if row['base_model_id'] != BASE_MODEL or not isinstance(row['base_revision'], str) or not COMMIT.fullmatch(row['base_revision']):
        raise AuditError('invalid model or revision')
    for key in ('prompt_sha256', 'tokenizer_sha256', 'generation_config_sha256'):
        val = row[key]
        if not isinstance(val, str) or not SHA256.fullmatch(val):
            raise AuditError('missing provenance hash')
    if row['adapter_sha256'] is not None:
        raise AuditError('baseline must have no adapter')
    budget = row['max_new_tokens']
    if type(budget) is not int or not 1 <= budget <= MAX_GENERATION_TOKENS:
        raise AuditError('invalid model generation budget')
    reason = row['finish_reason']
    if type(reason) is not str or reason not in TERMINATIONS:
        raise AuditError('invalid completion status')
    tokens = row['output_token_ids']
    if not isinstance(tokens, list) or len(tokens) > budget:
        raise AuditError('invalid output token count')
    repetition_fraction(tokens)
    if reason == 'eos' and (not tokens or tokens[-1] != EOS_QWEN25 or EOS_QWEN25 in tokens[:-1]):
        raise AuditError('EOS completion lacks a terminal EOS token')
    if reason == 'length' and (len(tokens) != budget or EOS_QWEN25 in tokens):
        raise AuditError('length completion must exactly exhaust its budget without EOS')
    if reason in {'cancelled', 'failed'} and EOS_QWEN25 in tokens:
        raise AuditError('interrupted result contains EOS')
    return row


def assess_loop_signal(records: list[dict]) -> dict:
    """Baseline-only batch triage. A flag is an invitation to review, never edit."""
    if not isinstance(records, list) or not records or len(records) > MAX_ROWS:
        raise AuditError('expected 1..2000 baseline cases')
    seen_ids, seen_prompts = set(), set()
    identity = None
    counts = {'eos': 0, 'length': 0, 'cancelled': 0, 'failed': 0}
    suspects = 0
    rep_sum = 0.0
    for record in records:
        r = _validated(record)
        task_id = r['task_id']
        prompt_hash = r['prompt_sha256']
        if task_id in seen_ids or prompt_hash in seen_prompts:
            raise AuditError('duplicate task ID or prompt hash')
        seen_ids.add(task_id)
        seen_prompts.add(prompt_hash)
        config = (r['base_model_id'], r['base_revision'], r['tokenizer_sha256'],
                  r['generation_config_sha256'], r['max_new_tokens'])
        if identity is None:
            identity = config
        elif identity != config:
            raise AuditError('mixed model revisions or generation settings')
        reason = r['finish_reason']
        counts[reason] += 1
        fraction = repetition_fraction(r['output_token_ids'])
        if reason in {'eos', 'length'}:
            rep_sum += fraction
            if reason == 'length' and len(r['output_token_ids']) >= 32 and fraction >= .35:
                suspects += 1
    eligible = counts['eos'] + counts['length']
    interrupted = counts['cancelled'] + counts['failed']
    # Small or cancelled samples do not establish a model-wide failure pattern.
    decision = ('insufficient_cases' if eligible < 20 else
                'manual_loop_review_required' if suspects >= 3 and suspects / eligible >= .15 else
                'no_repeated_loop_signal')
    return {
        'schema': 1, 'model_id': BASE_MODEL, 'base_revision': identity[1],
        'n_cases': len(records), 'n_eligible': eligible, 'eos_finished': counts['eos'],
        'length_limited': counts['length'], 'interrupted': interrupted,
        'review_candidates': suspects,
        'length_limit_fraction': counts['length'] / eligible if eligible else None,
        'review_candidate_fraction': suspects / eligible if eligible else None,
        'mean_repeated_4gram_fraction': rep_sum / eligible if eligible else None,
        'decision': decision, 'recommendation': 'review_only_no_weight_edit',
        'limitations': 'Token repetition is not proof of a reasoning-loop defect, and no task correctness is scored.',
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('baseline_jsonl', help='Local JSONL of token IDs + hashes only; never raw prompts')
    args = parser.parse_args(argv)
    try:
        report = assess_loop_signal(read_audit(args.baseline_jsonl))
    except AuditError as exc:
        parser.error(str(exc))
    print(json.dumps(report, sort_keys=True, allow_nan=False, ensure_ascii=False))


if __name__ == '__main__':
    main()
