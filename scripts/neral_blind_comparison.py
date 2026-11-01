"""Prepare blind review of already collected responses. Never runs a model.

Input rows: {id, text}; metadata JSON files must match the frozen run contract.
Writes separate reviewer.jsonl and private-key.json in a NEW output directory.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import secrets

ROOT = Path(__file__).resolve().parents[1]
HOLDOUT = ROOT / 'datasets/neral_qlora_holdout.jsonl'
PAYLOAD_FIELDS = {'id', 'category', 'prompt', 'rubric', 'A', 'B'}
SCORE_FIELDS = {'score_A_0_to_2', 'score_B_0_to_2', 'regression_notes'}


def review_commitment(rows):
    payload = [{key: row[key] for key in PAYLOAD_FIELDS} for row in rows]
    payload.sort(key=lambda row: row['id'])
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_responses(path, expected):
    result = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        if set(row) != {'id', 'text'} or row['id'] in result or row['id'] not in expected:
            raise ValueError('Unexpected, duplicate or malformed response.')
        if not isinstance(row['text'], str) or not row['text'].strip() or len(row['text']) > 32768:
            raise ValueError('Invalid answer.')
        result[row['id']] = row['text']
    if set(result) != expected:
        raise ValueError('Missing responses; do not drop failed tasks.')
    return result


def prepare(base_file, adapter_file, base_meta, adapter_meta, out):
    cases = [json.loads(line) for line in HOLDOUT.read_text(encoding='utf-8').splitlines()]
    digest = hashlib.sha256(HOLDOUT.read_bytes()).hexdigest()
    metadata = [json.loads(Path(p).read_text(encoding='utf-8')) for p in (base_meta, adapter_meta)]
    common = {'base_model', 'base_revision', 'base_snapshot_integrity', 'holdout_sha256', 'system_prompt',
              'chat_template_sha256', 'max_new_tokens', 'do_sample', 'quantization', 'packages'}
    for key in common:
        if key not in metadata[0] or metadata[0][key] != metadata[1].get(key):
            raise ValueError(f'Unequal or missing comparison setting: {key}')
    if (metadata[0]['base_model'] != 'Qwen/Qwen2.5-Coder-7B-Instruct'
            or not re.fullmatch('[a-f0-9]{40}', str(metadata[0]['base_revision']))
            or not re.fullmatch('[a-f0-9]{64}', str(metadata[0]['base_snapshot_integrity']))
            or not re.fullmatch('[a-f0-9]{64}', str(metadata[0]['chat_template_sha256']))
            or not metadata[0]['system_prompt'] or not metadata[0]['packages']
            or metadata[0]['max_new_tokens'] != 256 or not metadata[0]['quantization']
            or metadata[0]['holdout_sha256'] != digest or metadata[0]['do_sample'] is not False
            or metadata[0].get('adapter_sha256') is not None
            or not isinstance(metadata[1].get('adapter_sha256'), str)
            or not re.fullmatch('[a-f0-9]{64}', metadata[1]['adapter_sha256'])):
        raise ValueError('Invalid base/adapter provenance or holdout.')
    answers = [read_responses(p, {c['id'] for c in cases}) for p in (base_file, adapter_file)]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rng = random.Random(secrets.randbits(128))
    review, key = [], {}
    for case in cases:
        order = [0, 1]
        rng.shuffle(order)
        review.append({**case, 'A': answers[order[0]][case['id']], 'B': answers[order[1]][case['id']],
                       'score_A_0_to_2': None, 'score_B_0_to_2': None, 'regression_notes': ''})
        key[case['id']] = {'A': 'base' if order[0] == 0 else 'adapter', 'B': 'base' if order[1] == 0 else 'adapter'}
    (out / 'reviewer.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in review), encoding='utf-8')
    (out / 'private-key.json').write_text(json.dumps({'mapping': key, 'metadata': metadata,
        'review_commitment_sha256': review_commitment(review),
        'input_sha256': [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (base_file, adapter_file)]}, indent=2), encoding='utf-8')


def score_review(review_file, private_key):
    """Unblind COMPLETE human scores. Identity-only gains cannot establish capability."""
    rows = [json.loads(line) for line in Path(review_file).read_text(encoding='utf-8').splitlines()]
    key = json.loads(Path(private_key).read_text(encoding='utf-8'))
    expected = set(key['mapping'])
    if len(rows) != len(expected) or {r.get('id') for r in rows} != expected:
        raise ValueError('Incomplete or duplicate scored cases.')
    if any(set(row) != PAYLOAD_FIELDS | SCORE_FIELDS for row in rows):
        raise ValueError('Scored review fields changed.')
    if review_commitment(rows) != key['review_commitment_sha256']:
        raise ValueError('Prompts, rubrics or responses changed after blinding.')
    totals = {'base': 0, 'adapter': 0}
    capability = {'base': 0, 'adapter': 0}
    categories = {}
    regressions = []
    per_case = []
    for row in rows:
        values = {}
        mapping = key['mapping'][row['id']]
        if set(mapping) != {'A', 'B'} or set(mapping.values()) != {'base', 'adapter'}:
            raise ValueError('Invalid private mapping.')
        for label in ('A', 'B'):
            value = row[f'score_{label}_0_to_2']
            if type(value) is not int or value not in (0, 1, 2):
                raise ValueError('Every score must be an integer 0, 1 or 2; no skipped cases.')
            values[mapping[label]] = value
        if not isinstance(row['regression_notes'], str):
            raise ValueError('Review notes must be text.')
        category = categories.setdefault(row['category'], {'base': 0, 'adapter': 0, 'cases': 0})
        category['cases'] += 1
        for model, value in values.items():
            totals[model] += value
            category[model] += value
            if row['category'] != 'identity':
                capability[model] += value
        delta = values['adapter']-values['base']
        if delta < 0:
            regressions.append(row['id'])
        per_case.append({'id': row['id'], 'category': row['category'], **values,
                         'delta': delta, 'notes': row['regression_notes']})
    return {'cases': len(rows), 'totals': totals, 'capability_excluding_identity': capability,
            'categories': categories, 'regressions': regressions, 'per_case': per_case,
            'capability_gain_on_this_pilot': capability['adapter'] > capability['base'] and not regressions,
            'release_authorized': False,
            'scope': 'Human-scored pilot only; no statistical/general quality claim or automatic activation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'adapter', 'base-meta', 'adapter-meta', 'out', 'score-review', 'private-key'):
        parser.add_argument('--'+name)
    args = parser.parse_args()
    if args.score_review:
        if not args.private_key or any((args.base, args.adapter, args.base_meta, args.adapter_meta, args.out)):
            parser.error('Scoring requires --score-review and --private-key only.')
        print(json.dumps(score_review(args.score_review, args.private_key), indent=2, ensure_ascii=False))
    else:
        if not all((args.base, args.adapter, args.base_meta, args.adapter_meta, args.out)) or args.private_key:
            parser.error('Blinding requires --base, --adapter, --base-meta, --adapter-meta and --out.')
        prepare(args.base, args.adapter, args.base_meta, args.adapter_meta, args.out)
