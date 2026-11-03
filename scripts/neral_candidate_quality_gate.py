"""Read-only, fail-closed NERAL adapter quality gate for independently reviewed HOLDOUTs.

Outputs REJECT or MANUAL_REVIEW_REQUIRED, never deploys or changes model weights.
Neither imported records nor review assertions are independently authenticated.
Designed for a future *independent final* holdout, not the known 100 DEV cases.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

MODEL = 'Qwen/Qwen2.5-Coder-7B-Instruct'
SHA256 = re.compile(r'[a-f0-9]{64}\Z')
SHA40 = re.compile(r'[a-f0-9]{40}\Z')
IDENT = re.compile(r'[a-z0-9][a-z0-9_-]{2,63}\Z')
CATEGORIES = {'coding', 'math', 'reasoning', 'instruction', 'reliability'}
RUN_KEYS = {'schema', 'role', 'base_model_id', 'base_revision', 'base_snapshot_sha256',
            'tokenizer_sha256', 'prompt_template_sha256', 'decoding_sha256',
            'tools_sha256', 'benchmark_sha256', 'adapter_sha256', 'cases'}
ROW_KEYS = {'id', 'prompt_sha256', 'rubric_sha256', 'answer_sha256',
            'status', 'complete', 'unsupported_claims', 'reviewer'}
COMMON = RUN_KEYS - {'role', 'adapter_sha256', 'cases'}
MAX_JSON_BYTES = 8 * 1024 * 1024
MAX_ADAPTER_BYTES = 512 * 1024 * 1024


class GateError(ValueError):
    """Invalid provenance, missing review, unequal comparison, or unsafe file."""


def _require_hash(value, name):
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise GateError(f'invalid {name}')


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise GateError('duplicate JSON key')
        value[key] = item
    return value


def _invalid_constant(_):
    raise GateError('non-finite JSON constant')


def read_json(path: Path):
    """Reject linked/oversized inputs and JSON ambiguities, including NaN."""
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise GateError('missing, linked, or oversized JSON input')
    try:
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=_invalid_constant)
    except (UnicodeError, ValueError, OSError, RecursionError) as exc:
        raise GateError('unreadable or invalid JSON input') from exc


def sha256_file(path: Path) -> str:
    """Check actual local adapter file bytes; never loads or rewrites the model."""
    path = Path(path)
    if (path.is_symlink() or path.suffix != '.safetensors' or not path.is_file()
            or not 0 < path.stat().st_size <= MAX_ADAPTER_BYTES):
        raise GateError('missing, linked, incorrectly named or oversized adapter')
    dig = hashlib.sha256()
    with path.open('rb') as stream:
        while True:
            data = stream.read(1024 * 1024)
            if not data:
                break
            dig.update(data)
    return dig.hexdigest()


def _manifest(manifest):
    if not isinstance(manifest, dict) or set(manifest) != {
        'schema', 'benchmark_version', 'split', 'independent_review_assertion', 'tasks'
    } or manifest['schema'] != 1:
        raise GateError('unsupported benchmark manifest')
    if (manifest['split'] != 'final' or manifest['independent_review_assertion'] is not True
            or not isinstance(manifest['benchmark_version'], str)
            or not 3 <= len(manifest['benchmark_version']) <= 100):
        raise GateError('final independently reviewed benchmark required; DEV is not a final holdout')
    tasks = manifest['tasks']
    if not isinstance(tasks, list) or not 100 <= len(tasks) <= 500:
        raise GateError('expected 100..500 independent final cases')
    seen_ids, seen_prompts, seen_rubrics, categories = set(), set(), set(), Counter()
    for task in tasks:
        if not isinstance(task, dict) or set(task) != {'id', 'category', 'prompt_sha256', 'rubric_sha256'}:
            raise GateError('invalid benchmark case')
        key, cat = task['id'], task['category']
        if not isinstance(key, str) or IDENT.fullmatch(key) is None or cat not in CATEGORIES:
            raise GateError('invalid benchmark case id/category')
        _require_hash(task['prompt_sha256'], 'prompt digest')
        _require_hash(task['rubric_sha256'], 'rubric digest')
        if (key in seen_ids or task['prompt_sha256'] in seen_prompts
                or task['rubric_sha256'] in seen_rubrics):
            raise GateError('duplicate task identity/commitment')
        seen_ids.add(key);seen_prompts.add(task['prompt_sha256']);seen_rubrics.add(task['rubric_sha256'])
        categories[cat] += 1
    if any(categories[x] < 10 for x in CATEGORIES):
        raise GateError('missing category coverage in final benchmark')
    sha = hashlib.sha256(json.dumps(manifest, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode('utf-8')).hexdigest()
    return {task['id']: task for task in tasks}, sha


def _run(run, *, role, tasks, benchmark_sha):
    if not isinstance(run, dict) or set(run) != RUN_KEYS or run['schema'] != 1 or run['role'] != role:
        raise GateError('wrong run schema or role')
    if run['base_model_id'] != MODEL or not isinstance(run['base_revision'], str) or not SHA40.fullmatch(run['base_revision']):
        raise GateError('wrong model family or revision')
    for key in COMMON - {'schema', 'base_model_id', 'base_revision'}:
        _require_hash(run[key], key)
    if run['benchmark_sha256'] != benchmark_sha:
        raise GateError('scored runs are not tied to this exact holdout')
    if role == 'baseline':
        if run['adapter_sha256'] is not None:
            raise GateError('baseline must be the unmodified base model')
    else:
        _require_hash(run['adapter_sha256'], 'candidate adapter digest')
    rows = run['cases']
    if not isinstance(rows, list) or len(rows) != len(tasks):
        raise GateError('missing/extra scored cases')
    indexed = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != ROW_KEYS or not isinstance(row['id'], str):
            raise GateError('incorrect scored row shape')
        case = tasks.get(row['id'])
        if case is None or row['id'] in indexed:
            raise GateError('duplicate/unknown scored case')
        if row['prompt_sha256'] != case['prompt_sha256'] or row['rubric_sha256'] != case['rubric_sha256']:
            raise GateError('answer review is not bound to final task and rubric')
        _require_hash(row['answer_sha256'], 'answer digest')
        if (row['status'] not in {'full','partial','fail'} or type(row['complete']) is not bool
                or type(row['unsupported_claims']) is not int or not 0 <= row['unsupported_claims'] <= 100):
            raise GateError('invalid scoring/completion status')
        if row['status'] == 'full' and (not row['complete'] or row['unsupported_claims']):
            raise GateError('incomplete or unsupported answer cannot receive full credit')
        reviewer = row['reviewer']
        if (not isinstance(reviewer,str) or not 3 <= len(reviewer) <= 120
                or reviewer.lower().strip() in {'model-itself','self','autogenerated','model','llm'}):
            raise GateError('independent named reviewer assertion required')
        indexed[row['id']] = row
    if len(indexed) != len(tasks):
        raise GateError('incomplete case inventory')
    return indexed


def _exact_one_sided_p(wins, losses):
    """Exact paired McNemar sign p for a candidate improving on a baseline."""
    n = wins + losses
    if n == 0:
        return 1.0
    # Exact binomial upper tail for H0 P(win)=0.5; n <= 500 keeps ints bounded.
    return sum(math.comb(n, k) for k in range(wins, n + 1)) / (2**n)


def compare_scored_runs(manifest, baseline, candidate, adapter_digest):
    """Review-only recommendation; never returns automatic promotion approval."""
    _require_hash(adapter_digest, 'on-disk adapter digest')
    tasks, benchmark_sha = _manifest(manifest)
    a = _run(baseline, role='baseline', tasks=tasks, benchmark_sha=benchmark_sha)
    b = _run(candidate, role='candidate', tasks=tasks, benchmark_sha=benchmark_sha)
    if candidate['adapter_sha256'] != adapter_digest:
        raise GateError('candidate adapter differs from local sealed bytes')
    for key in COMMON:
        if baseline[key] != candidate[key]:
            raise GateError(f'baseline and candidate disagree on {key}')
    baseline_full = sum(r['status'] == 'full' for r in a.values())
    candidate_full = sum(r['status'] == 'full' for r in b.values())
    newly = [k for k in tasks if a[k]['status'] != 'full' and b[k]['status'] == 'full']
    regressions = [k for k in tasks if a[k]['status'] == 'full' and b[k]['status'] != 'full']
    before_unsupported = sum(r['unsupported_claims'] for r in a.values())
    after_unsupported = sum(r['unsupported_claims'] for r in b.values())
    before_incomplete = sum(not r['complete'] for r in a.values())
    after_incomplete = sum(not r['complete'] for r in b.values())
    by_category = {}
    for category in sorted(CATEGORIES):
        selected = [k for k,t in tasks.items() if t['category'] == category]
        before = sum(a[k]['status'] == 'full' for k in selected)
        after = sum(b[k]['status'] == 'full' for k in selected)
        by_category[category] = {'cases':len(selected),'baseline_full':before,'candidate_full':after,'delta':after-before}
    p_value = _exact_one_sided_p(len(newly),len(regressions))
    min_gain = max(5, math.ceil(len(tasks) * 0.05))
    accepted_for_review = (
        candidate_full - baseline_full >= min_gain
        and not regressions and after_unsupported <= before_unsupported
        and after_incomplete <= before_incomplete
        and all(x['delta'] >= 0 for x in by_category.values())
        and p_value <= 0.05
    )
    return {
        'schema':1,'decision':'MANUAL_REVIEW_REQUIRED' if accepted_for_review else 'REJECT',
        'base_model_id':MODEL,'base_revision':candidate['base_revision'],
        'benchmark_sha256':benchmark_sha, 'adapter_sha256':adapter_digest,
        'case_count':len(tasks),'baseline_full':baseline_full,'candidate_full':candidate_full,
        'newly_solved':len(newly),'regressions':len(regressions),
        'absolute_gain':candidate_full-baseline_full,'minimum_gain_required':min_gain,
        'one_sided_exact_p':p_value,'unsupported_baseline':before_unsupported,
        'unsupported_candidate':after_unsupported,'incomplete_baseline':before_incomplete,
        'incomplete_candidate':after_incomplete,'categories':by_category,
        'weights_modified':False,'deployment_authorized':False,
        'independent_reviews_authenticated':False,
        'limitations':[
            'Run outputs, test independence and human review are user-supplied assertions, not independently authenticated.',
            'The exact paired test does not guarantee task independence or detect mislabeled answers.',
            'This CLI never loads/edits weights, auto-activates an adapter, merges code or authorizes release.'
        ],
    }


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--final-manifest',required=True,type=Path)
    p.add_argument('--baseline',required=True,type=Path)
    p.add_argument('--candidate',required=True,type=Path)
    p.add_argument('--adapter-file',required=True,type=Path)
    args=p.parse_args(argv)
    try:
        result=compare_scored_runs(read_json(args.final_manifest),read_json(args.baseline),
                                   read_json(args.candidate),sha256_file(args.adapter_file))
    except (GateError, OSError, ValueError, TypeError) as exc:
        p.exit(2, f'Candidate gate REFUSED: {type(exc).__name__}: {exc}\n')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    return 0 if result['decision']=='MANUAL_REVIEW_REQUIRED' else 1


if __name__=='__main__':
    raise SystemExit(main())
