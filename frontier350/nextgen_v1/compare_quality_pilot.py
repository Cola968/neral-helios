"""Compare a bounded Neral 350M candidate against its immutable SAFE anchor.

NIS2 chat-core is a diagnostic, NOT the full NIS2 score or a promotion gate.
No checkpoint is loaded, rewritten, selected, or promoted here.
"""
import argparse
import json
import math
from pathlib import Path


def read(path):
    obj = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(obj, dict):
        raise ValueError('expected JSON object')
    return obj


def finite(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('non-finite benchmark value')
    return result


def compare(base_dev, cand_dev, base_nis, cand_nis, train, *, anchor_sha, candidate_sha, start_tokens, extra_tokens):
    if len(anchor_sha) != 64 or len(candidate_sha) != 64 or anchor_sha == candidate_sha:
        raise ValueError('distinct pinned checkpoint hashes required')
    stop = start_tokens + extra_tokens
    if start_tokens <= 0 or extra_tokens <= 0 or extra_tokens % 24576:
        raise ValueError('invalid bounded token contract')
    if base_dev.get('checkpoint_sha256') != anchor_sha or cand_dev.get('checkpoint_sha256') != candidate_sha:
        raise ValueError('development reports do not match pinned checkpoint hashes')
    if base_dev.get('checkpoint_processed_tokens') != start_tokens or cand_dev.get('checkpoint_processed_tokens') != stop:
        raise ValueError('development checkpoint token mismatch')
    for key in ('suite', 'protocol_sha256', 'tokenizer_sha256', 'architecture', 'cases'):
        if base_dev.get(key) != cand_dev.get(key):
            raise ValueError('development protocol mismatch: '+key)
    if base_dev.get('cases', 0) <= 0:
        raise ValueError('empty development suite')
    base_cats = base_dev.get('categories', {})
    cand_cats = cand_dev.get('categories', {})
    if not base_cats or set(base_cats) != set(cand_cats):
        raise ValueError('development categories changed')
    dev_categories = {}
    for cat in sorted(base_cats):
        if base_cats[cat].get('total') != cand_cats[cat].get('total'):
            raise ValueError('development denominators changed')
        a = finite(base_cats[cat]['accuracy_pct'])
        b = finite(cand_cats[cat]['accuracy_pct'])
        dev_categories[cat] = {'safe_pct': a, 'candidate_pct': b, 'delta_pp': b-a, 'no_regression': b>=a}
    dev_base = finite(base_dev['overall_accuracy_pct'])
    dev_cand = finite(cand_dev['overall_accuracy_pct'])
    for key in ('benchmark', 'suite_sha256', 'architecture', 'parameters', 'case_counts'):
        if base_nis.get(key) != cand_nis.get(key):
            raise ValueError('NIS2 chat-core protocol mismatch: '+key)
    if base_nis.get('checkpoint_processed_tokens') != start_tokens or cand_nis.get('checkpoint_processed_tokens') != stop:
        raise ValueError('NIS2 checkpoint token mismatch')
    if base_nis.get('case_counts', {}).get('total') != 140:
        raise ValueError('not the pinned 140-case NIS2 diagnostic')
    if base_nis.get('full_nis2', {}).get('score') is not None or cand_nis.get('full_nis2', {}).get('score') is not None:
        raise ValueError('do not misrepresent chat-core as canonical NIS2')
    if not train.get('complete') or train.get('new_tokens') != extra_tokens or train.get('processed_tokens') != stop:
        raise ValueError('incomplete or mismatched training')
    if train.get('checkpoint_sha256') != candidate_sha or train.get('updates') != extra_tokens//24576:
        raise ValueError('training checkpoint identity/update mismatch')
    if train.get('frozen_weights_unchanged') is not True or train.get('promoted') is not False:
        raise ValueError('frozen-weight/no-promotion contract failed')
    keys = ('pair_overall_pct', 'instruction_following_pct', 'generation_health_degenerative_pct', 'generation_health_repeated_trigram_pct')
    diagnostics = {}
    for key in keys:
        a = finite(base_nis['scores'][key]); b = finite(cand_nis['scores'][key])
        diagnostics[key] = {'safe': a, 'candidate': b, 'delta_candidate_minus_safe': b-a}
    dev_pass = dev_cand >= dev_base and all(r['no_regression'] for r in dev_categories.values())
    return {'schema': 1, 'status': 'comparison_complete', 'anchor_sha256': anchor_sha,
            'candidate_sha256': candidate_sha, 'training_tokens': extra_tokens,
            'candidate_tokens': stop, 'development': {'safe_overall_pct': dev_base,
            'candidate_overall_pct': dev_cand, 'delta_pp': dev_cand-dev_base,
            'no_regression_in_this_dev_suite': dev_pass, 'categories': dev_categories},
            'nis2_chat_core_diagnostic_only': {'suite_sha256': base_nis['suite_sha256'],
            'case_counts': base_nis['case_counts'], 'metrics': diagnostics,
            'canonical_full_nis2_verified': False, 'used_for_checkpoint_selection': False},
            'validation_loss_verified': False, 'quality_improvement_proven': False,
            'end_to_end_44k_verified': False, 'promotion_allowed': False,
            'note': 'Research comparison only; full held-out validation, all NIS2 lineages and promotion gate not run.'}


def main():
    ap = argparse.ArgumentParser()
    for name in ('base-dev', 'candidate-dev', 'base-nis', 'candidate-nis', 'train-summary', 'anchor-sha', 'candidate-sha', 'out'):
        ap.add_argument('--'+name, required=True)
    ap.add_argument('--start-tokens', type=int, required=True)
    ap.add_argument('--extra-tokens', type=int, required=True)
    a = ap.parse_args()
    result = compare(read(a.base_dev), read(a.candidate_dev), read(a.base_nis), read(a.candidate_nis),
                     read(a.train_summary), anchor_sha=a.anchor_sha, candidate_sha=a.candidate_sha,
                     start_tokens=a.start_tokens, extra_tokens=a.extra_tokens)
    Path(a.out).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
