"""Unblind a FROZEN human grading file into a private paired aggregate.

Does not judge free answers, certify grader independence or claim quality gain.
Requires a private review packet, its secret key and a separately frozen,
case-complete grading file. Never publish any private source or answer text.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from phoenix_blind_ab_audit import ORIGINAL_SHA, SHA, TAG, load


def summarize(packet: dict, packet_sha: str, grades: dict, grades_sha: str,
              key: dict) -> dict:
    if (not isinstance(packet, dict) or set(packet) !=
            {'schema', 'classification', 'holdout_sha256', 'items'} or
            type(packet['schema']) is not int or packet['schema'] != 1 or
            packet['classification'] != 'UNSCORED_BLIND_REVIEW_NOT_VERIFIED_INDEPENDENT' or
            not isinstance(packet['holdout_sha256'], str) or
            not SHA.fullmatch(packet['holdout_sha256'])):
        raise ValueError('invalid original private review packet')
    items = packet['items']
    if not isinstance(items, list) or not 8 <= len(items) <= 500:
        raise ValueError('invalid review case count')
    if (not isinstance(grades, dict) or set(grades) !=
            {'schema', 'packet_sha256', 'items'} or
            type(grades['schema']) is not int or grades['schema'] != 1 or
            grades['packet_sha256'] != packet_sha):
        raise ValueError('grading file is not pinned to exact original review packet')
    if (not isinstance(key, dict) or set(key) !=
            {'schema', 'holdout_sha256', 'packet_sha256', 'baseline_checkpoint_sha256',
             'candidate_checkpoint_sha256', 'case_mapping'} or
            type(key['schema']) is not int or key['schema'] != 1 or
            key['packet_sha256'] != packet_sha or
            key['holdout_sha256'] != packet['holdout_sha256'] or
            key['baseline_checkpoint_sha256'] != ORIGINAL_SHA or
            not isinstance(key['candidate_checkpoint_sha256'], str) or
            not SHA.fullmatch(key['candidate_checkpoint_sha256']) or
            key['candidate_checkpoint_sha256'] == ORIGINAL_SHA):
        raise ValueError('key provenance is incomplete or wrong')
    if (not isinstance(grades['items'], list) or
            not isinstance(key['case_mapping'], list) or
            len(grades['items']) != len(items) or
            len(key['case_mapping']) != len(items)):
        raise ValueError('grading/key rows do not exactly match review packet')
    totals = {'baseline_correct': 0, 'candidate_correct': 0,
              'both_correct': 0, 'both_wrong': 0,
              'baseline_only': 0, 'candidate_only': 0}
    breakdown: dict[str, dict[str, dict[str, int]]] = {'language': {}, 'domain': {}}
    seen = set()
    for original, grade, mapping in zip(items, grades['items'], key['case_mapping']):
        if (not isinstance(original, dict) or set(original) !=
                {'id', 'language', 'domain', 'prompt', 'rubric', 'A', 'B'} or
                not isinstance(original['id'], str) or
                not TAG.fullmatch(original['id']) or original['id'] in seen or
                original['language'] not in ('de', 'en') or
                not isinstance(original['domain'], str) or not original['domain'].strip()):
            raise ValueError('invalid review row')
        seen.add(original['id'])
        if (not isinstance(grade, dict) or set(grade) !=
                {'id', 'A_correct', 'B_correct', 'notes'} or
                grade['id'] != original['id'] or
                type(grade['A_correct']) is not bool or type(grade['B_correct']) is not bool or
                not isinstance(grade['notes'], str) or not 1 <= len(grade['notes'].strip()) <= 2000):
            raise ValueError('human grade incomplete, out of order or invalid')
        if (not isinstance(mapping, dict) or set(mapping) !=
                {'id', 'A', 'B', 'A_answer_sha256', 'B_answer_sha256'} or
                mapping['id'] != original['id'] or
                {mapping['A'], mapping['B']} != {'baseline', 'candidate'} or
                any(not isinstance(mapping[label + '_answer_sha256'], str) or
                    not SHA.fullmatch(mapping[label + '_answer_sha256']) or
                    mapping[label + '_answer_sha256'] != hashlib.sha256(
                        original[label].encode('utf-8')).hexdigest()
                    for label in ('A', 'B'))):
            raise ValueError('secret mapping inconsistent with review packet')
        is_base_a = mapping['A'] == 'baseline'
        base = grade['A_correct'] if is_base_a else grade['B_correct']
        candidate = grade['B_correct'] if is_base_a else grade['A_correct']
        counts = {'baseline_correct': int(base), 'candidate_correct': int(candidate),
                  'both_correct': int(base and candidate),
                  'both_wrong': int(not base and not candidate),
                  'baseline_only': int(base and not candidate),
                  'candidate_only': int(candidate and not base)}
        for field, value in counts.items():
            totals[field] += value
        for dimension in breakdown:
            label = original[dimension]
            group = breakdown[dimension].setdefault(label, {'n': 0, **dict.fromkeys(totals, 0)})
            group['n'] += 1
            for field, value in counts.items():
                group[field] += value
    if set(breakdown['language']) != {'de', 'en'}:
        raise ValueError('review packet has no bilingual coverage')
    return {'schema': 1,
            'classification': 'HUMAN_REPORTED_SCORES_NOT_INDEPENDENTLY_VERIFIED',
            'packet_sha256': packet_sha, 'grading_sha256': grades_sha,
            'holdout_sha256': packet['holdout_sha256'],
            'baseline_checkpoint_sha256': ORIGINAL_SHA,
            'candidate_checkpoint_sha256': key['candidate_checkpoint_sha256'],
            'n': len(items), **totals,
            'paired_net_correct_delta': totals['candidate_only'] - totals['baseline_only'],
            'groups': breakdown,
            'independent_holdout_verified': False,
            'grader_independence_verified': False,
            'quality_improvement_proven': False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--packet', required=True, type=Path)
    p.add_argument('--grades', required=True, type=Path)
    p.add_argument('--secret-key', required=True, type=Path)
    p.add_argument('--private-dir', required=True, type=Path)
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    if not args.execute or not TAG.fullmatch(args.tag):
        p.error('explicit --execute and simple tag required')
    root = args.private_dir
    if (root.is_symlink() or not root.is_dir() or
            root.resolve().name.lower() != 'phoenix' or
            root.resolve().parent.name.lower() != 'neralprivateassets'):
        p.error('existing NeralPrivateAssets/phoenix directory required')
    packet, packet_sha = load(args.packet)
    grades, grades_sha = load(args.grades)
    key, _ = load(args.secret_key)
    report = summarize(packet, packet_sha, grades, grades_sha, key)
    target = root / ('phoenix_scored_' + args.tag + '.json')
    if target.is_symlink() or target.exists():
        raise FileExistsError('never overwrite an existing grading report')
    with target.open('x', encoding='utf-8') as out:
        json.dump(report, out, ensure_ascii=False, indent=2, sort_keys=True)
        out.write('\n')
    print(json.dumps({'event': 'private_human_report_aggregated', 'n': report['n'],
                      'baseline_correct': report['baseline_correct'],
                      'candidate_correct': report['candidate_correct'],
                      'quality_improvement_proven': False}, sort_keys=True))


if __name__ == '__main__':
    main()
