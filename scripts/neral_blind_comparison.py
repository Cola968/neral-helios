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
        'input_sha256': [hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in (base_file, adapter_file)]}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'adapter', 'base-meta', 'adapter-meta', 'out'):
        parser.add_argument('--'+name, required=True)
    args = parser.parse_args()
    prepare(args.base, args.adapter, args.base_meta, args.adapter_meta, args.out)
