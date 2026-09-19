"""Create an offline, private, per-case blinded answer-review packet.

This tool does NOT create an independent holdout, grade answers, load weights,
train a model, or certify improvement. An independent human must author and
freeze the holdout and subsequently score the masked answers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import secrets

ORIGINAL_SHA = 'f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7'
SHA = re.compile(r'[0-9a-f]{64}\Z')
TAG = re.compile(r'[A-Za-z0-9_-]{1,40}\Z')
FINISH = {'eos', 'max_tokens', 'role_boundary', 'repetition_guard'}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load(path: Path) -> tuple[dict, str]:
    if path.is_symlink() or not path.is_file():
        raise ValueError('input must be a regular, non-symlinked file')
    raw = path.read_bytes()
    if len(raw) > 4_000_000:
        raise ValueError('refuse oversized private input')
    value = json.loads(raw.decode('utf-8'))
    if not isinstance(value, dict):
        raise ValueError('input JSON must be an object')
    return value, digest(raw)


def validate_holdout(data: dict) -> list[dict]:
    if set(data) != {'schema', 'cases'} or type(data['schema']) is not int or data['schema'] != 1:
        raise ValueError('unexpected holdout schema')
    cases = data['cases']
    if not isinstance(cases, list) or not 8 <= len(cases) <= 500:
        raise ValueError('holdout requires 8..500 independently authored cases')
    seen_ids, seen_prompts, languages = set(), set(), set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {'id', 'language', 'domain', 'prompt', 'rubric'}:
            raise ValueError('invalid holdout case schema')
        ident, language = case['id'], case['language']
        if not isinstance(ident, str) or not TAG.fullmatch(ident) or ident in seen_ids:
            raise ValueError('duplicate or malformed case ID')
        if language not in ('de', 'en'):
            raise ValueError('German and English cases only')
        for field in ('domain', 'prompt', 'rubric'):
            if not isinstance(case[field], str) or not 1 <= len(case[field].strip()) <= 4000:
                raise ValueError('empty or oversized holdout field')
        normalized = ' '.join(case['prompt'].casefold().split())
        if normalized in seen_prompts:
            raise ValueError('exact normalized duplicate holdout prompt')
        seen_ids.add(ident)
        seen_prompts.add(normalized)
        languages.add(language)
    if languages != {'de', 'en'}:
        raise ValueError('holdout needs both DE and EN')
    return cases


def validate_generations(report: dict, cases: list[dict], holdout_sha: str) -> None:
    required = {'schema', 'checkpoint_sha256', 'tokenizer_sha256', 'holdout_sha256',
                'max_new_tokens', 'decode', 'cases'}
    if set(report) != required or type(report['schema']) is not int or report['schema'] != 1:
        raise ValueError('unexpected generation-report schema')
    if any(not isinstance(report[key], str) or not SHA.fullmatch(report[key])
           for key in ('checkpoint_sha256', 'tokenizer_sha256', 'holdout_sha256')):
        raise ValueError('invalid SHA in generation report')
    if report['holdout_sha256'] != holdout_sha:
        raise ValueError('generation report references different holdout bytes')
    cap = report['max_new_tokens']
    if type(cap) is not int or not 1 <= cap <= 512:
        raise ValueError('invalid generation cap')
    if not isinstance(report['decode'], dict) or not report['decode']:
        raise ValueError('explicit decoding parameters required')
    rows = report['cases']
    if not isinstance(rows, list) or len(rows) != len(cases):
        raise ValueError('missing generation cases')
    for row, case in zip(rows, cases):
        if not isinstance(row, dict) or set(row) != {'id', 'prompt_sha256',
                'input_ids_sha256', 'answer', 'generated_tokens', 'finish_reason'}:
            raise ValueError('invalid generation case schema')
        if row['id'] != case['id'] or row['prompt_sha256'] != digest(case['prompt'].encode('utf-8')):
            raise ValueError('case order/prompt mismatch')
        if not isinstance(row['input_ids_sha256'], str) or not SHA.fullmatch(row['input_ids_sha256']):
            raise ValueError('missing tokenized-prompt identity')
        if not isinstance(row['answer'], str) or len(row['answer']) > 20_000:
            raise ValueError('invalid free answer')
        if type(row['generated_tokens']) is not int or not 0 <= row['generated_tokens'] <= cap:
            raise ValueError('invalid generated token count')
        if row['finish_reason'] not in FINISH:
            raise ValueError('invalid stop reason')


def prepare(holdout: dict, holdout_sha: str, base: dict, candidate: dict) -> tuple[dict, dict]:
    cases = validate_holdout(holdout)
    for report in (base, candidate):
        validate_generations(report, cases, holdout_sha)
    if base['checkpoint_sha256'] != ORIGINAL_SHA:
        raise ValueError('baseline must be the pinned unchanged original checkpoint')
    if candidate['checkpoint_sha256'] == ORIGINAL_SHA:
        raise ValueError('candidate must have a different checkpoint hash')
    for key in ('tokenizer_sha256', 'max_new_tokens', 'decode'):
        if base[key] != candidate[key]:
            raise ValueError('baseline/candidate evaluation setting mismatch: ' + key)
    if any(a['input_ids_sha256'] != b['input_ids_sha256']
           for a, b in zip(base['cases'], candidate['cases'])):
        raise ValueError('baseline/candidate received different tokenized input')
    review, mapping = [], []
    for case, left, right in zip(cases, base['cases'], candidate['cases']):
        swapped = bool(secrets.randbits(1))
        a, b = (right, left) if swapped else (left, right)
        review.append({'id': case['id'], 'language': case['language'],
                       'domain': case['domain'], 'prompt': case['prompt'],
                       'rubric': case['rubric'], 'A': a['answer'], 'B': b['answer']})
        mapping.append({'id': case['id'], 'A': 'candidate' if swapped else 'baseline',
                        'B': 'baseline' if swapped else 'candidate'})
    return ({'schema': 1, 'classification': 'UNSCORED_BLIND_REVIEW_NOT_VERIFIED_INDEPENDENT',
             'holdout_sha256': holdout_sha, 'items': review},
            {'schema': 1, 'holdout_sha256': holdout_sha,
             'baseline_checkpoint_sha256': base['checkpoint_sha256'],
             'candidate_checkpoint_sha256': candidate['checkpoint_sha256'],
             'case_mapping': mapping})


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--holdout', required=True, type=Path)
    p.add_argument('--baseline', required=True, type=Path)
    p.add_argument('--candidate', required=True, type=Path)
    p.add_argument('--private-dir', required=True, type=Path)
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    if not args.execute or not TAG.fullmatch(args.tag):
        p.error('explicit --execute and simple tag required')
    root = args.private_dir
    if root.is_symlink() or not root.is_dir() or root.resolve().name.lower() != 'phoenix' or root.resolve().parent.name.lower() != 'neralprivateassets':
        p.error('existing private NeralPrivateAssets/phoenix directory required')
    holdout, holdout_sha = load(args.holdout)
    base, _ = load(args.baseline)
    candidate, _ = load(args.candidate)
    review, key = prepare(holdout, holdout_sha, base, candidate)
    blinded = root / ('phoenix_blind_' + args.tag + '.json')
    secret = root / ('phoenix_blind_' + args.tag + '.key.json')
    if any(path.exists() or path.is_symlink() for path in (blinded, secret)):
        raise FileExistsError('no private packet/key overwrite permitted')
    # Both outputs must remain private. Do not upload packet, grading or key to CI.
    for path, data in ((secret, key), (blinded, review)):
        with path.open('x', encoding='utf-8') as out:
            json.dump(data, out, ensure_ascii=False, indent=2)
            out.write('\n')
    print(json.dumps({'event': 'private_blind_packet_created', 'case_count': len(review['items']),
                      'holdout_sha256': holdout_sha,
                      'human_scoring_complete': False, 'independence_verified': False,
                      'quality_improvement_proven': False}, sort_keys=True))


if __name__ == '__main__':
    main()
