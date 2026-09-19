"""Offline exploratory paired-score consistency and exact discordance audit.

Consumes ONLY private aggregate human-reported scores. No model inference,
independence certification, winner selection, or automatic training gate.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from math import comb
from pathlib import Path

from phoenix_blind_ab_audit import ORIGINAL_SHA, SHA, TAG, load

FIELDS = ('baseline_correct', 'candidate_correct', 'both_correct', 'both_wrong',
          'baseline_only', 'candidate_only')
FLAGS = ('independent_holdout_verified', 'grader_independence_verified',
         'quality_improvement_proven')


def check_counts(row: dict, *, n: int) -> None:
    if type(n) is not int or not 1 <= n <= 500:
        raise ValueError('invalid sample size')
    if any(type(row.get(k)) is not int or not 0 <= row[k] <= n for k in FIELDS):
        raise ValueError('invalid paired counts')
    both, wrong, base, cand = (row[k] for k in
                               ('both_correct', 'both_wrong', 'baseline_only', 'candidate_only'))
    if (both + wrong + base + cand != n or
            row['baseline_correct'] != both + base or
            row['candidate_correct'] != both + cand):
        raise ValueError('paired partition/counts inconsistent')


def exact_two_sided_discordance(base_only: int, candidate_only: int) -> float:
    """Exact two-sided paired sign/McNemar test, without asymptotic approximation."""
    if (type(base_only) is not int or type(candidate_only) is not int or
            min(base_only, candidate_only) < 0 or base_only + candidate_only > 500):
        raise ValueError('invalid discordances')
    n = base_only + candidate_only
    if n == 0:
        return 1.0
    numerator = 2 * sum(comb(n, i) for i in range(min(base_only, candidate_only) + 1))
    return float(Fraction(min(2 ** n, numerator), 2 ** n))


def analyze(report: dict) -> dict:
    expected = {'schema', 'classification', 'packet_sha256', 'grading_sha256',
                'holdout_sha256', 'baseline_checkpoint_sha256',
                'candidate_checkpoint_sha256', 'n', *FIELDS, 'paired_net_correct_delta',
                'groups', *FLAGS}
    if not isinstance(report, dict) or set(report) != expected or type(report['schema']) is not int or report['schema'] != 1:
        raise ValueError('unexpected scored report schema')
    if report['classification'] != 'HUMAN_REPORTED_SCORES_NOT_INDEPENDENTLY_VERIFIED':
        raise ValueError('scored report classification mismatch')
    if any(not isinstance(report[k], str) or not SHA.fullmatch(report[k])
           for k in ('packet_sha256', 'grading_sha256', 'holdout_sha256',
                     'baseline_checkpoint_sha256', 'candidate_checkpoint_sha256')):
        raise ValueError('invalid provenance hash')
    if report['baseline_checkpoint_sha256'] != ORIGINAL_SHA:
        raise ValueError('baseline checkpoint is not the pinned original')
    if report['baseline_checkpoint_sha256'] == report['candidate_checkpoint_sha256']:
        raise ValueError('candidate and baseline checkpoint hashes identical')
    if any(report[f] is not False for f in FLAGS):
        raise ValueError('cannot elevate source-declared verification flags')
    n = report['n']
    if type(n) is not int or not 8 <= n <= 500:
        raise ValueError('expected 8..500 reviewed examples')
    check_counts(report, n=n)
    if type(report['paired_net_correct_delta']) is not int or report['paired_net_correct_delta'] != (report['candidate_only'] - report['baseline_only']):
        raise ValueError('paired net correctness mismatch')
    groups = report['groups']
    if not isinstance(groups, dict) or set(groups) != {'language', 'domain'}:
        raise ValueError('missing stratification dimensions')
    for dimension, values in groups.items():
        if not isinstance(values, dict) or not values:
            raise ValueError('empty stratification')
        if dimension == 'language' and set(values) != {'de', 'en'}:
            raise ValueError('bilingual groups missing')
        sums = {'n': 0, **dict.fromkeys(FIELDS, 0)}
        for label, row in values.items():
            if (not isinstance(label, str) or not label.strip() or len(label) > 4000 or
                    not isinstance(row, dict) or set(row) != {'n', *FIELDS} or
                    type(row['n']) is not int):
                raise ValueError('invalid stratification row')
            check_counts(row, n=row['n'])
            for field in sums:
                sums[field] += row[field]
        if any(sums[k] != report[k] for k in FIELDS) or sums['n'] != n:
            raise ValueError('group counts disagree with overall results')
    discordant = report['baseline_only'] + report['candidate_only']
    return {'schema': 1,
            'classification': 'EXPLORATORY_HUMAN_REPORTED_PAIRED_EVIDENCE_ONLY',
            'scored_report_sha256': None,
            'n': n, 'baseline_correct': report['baseline_correct'],
            'candidate_correct': report['candidate_correct'],
            'baseline_only': report['baseline_only'],
            'candidate_only': report['candidate_only'],
            'discordant_pairs': discordant,
            'paired_net_correct_delta': report['paired_net_correct_delta'],
            'paired_net_accuracy_delta': report['paired_net_correct_delta'] / n,
            'exact_two_sided_discordance_p_exploratory': exact_two_sided_discordance(
                report['baseline_only'], report['candidate_only']),
            'language_sample_sizes': {label: row['n'] for label, row in groups['language'].items()},
            'independent_holdout_verified': False,
            'grader_independence_verified': False,
            'quality_improvement_proven': False,
            'training_approval_granted': False}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--scored-report', type=Path, required=True)
    p.add_argument('--private-dir', type=Path, required=True)
    p.add_argument('--tag', required=True)
    args = p.parse_args()
    if not args.execute or not TAG.fullmatch(args.tag):
        p.error('explicit --execute and simple tag required')
    root = args.private_dir
    if (root.is_symlink() or not root.is_dir() or root.resolve().name.lower() != 'phoenix'
            or root.resolve().parent.name.lower() != 'neralprivateassets'):
        p.error('existing private NeralPrivateAssets/phoenix directory required')
    source, source_sha = load(args.scored_report)
    evidence = analyze(source)
    evidence['scored_report_sha256'] = source_sha
    dest = root / ('phoenix_paired_' + args.tag + '.json')
    if dest.exists() or dest.is_symlink():
        raise FileExistsError('refuse to overwrite private paired evidence')
    with dest.open('x', encoding='utf-8') as file:
        json.dump(evidence, file, indent=2, sort_keys=True)
        file.write('\n')
    print(json.dumps({'event': 'exploratory_private_paired_evidence_created',
                      'n': evidence['n'], 'quality_improvement_proven': False,
                      'training_approval_granted': False}, sort_keys=True))


if __name__ == '__main__':
    main()
