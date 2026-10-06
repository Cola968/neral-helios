"""Compare TWO read-only fixed-validation reports. Never promotes or changes weights.

The caller must independently verify both checkpoint bytes and the holdout shard
hashes. One evaluation protocol SHA and its parameters must match exactly.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'invalid {name}')
    x = float(value)
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f'nonfinite or nonpositive {name}')
    return x


def assess(base: dict, candidate: dict, *, anchor_sha: str, candidate_sha: str,
           anchor_tokens: int, candidate_tokens: int) -> dict:
    if anchor_sha == candidate_sha or any(len(s) != 64 or any(ch not in '0123456789abcdef' for ch in s)
                                           for s in (anchor_sha, candidate_sha)):
        raise ValueError('two different pinned lowercase checkpoint digests required')
    for label, report, sha, tokens in (('SAFE', base, anchor_sha, anchor_tokens),
                                       ('candidate', candidate, candidate_sha, candidate_tokens)):
        if report.get('schema') != 1 or report.get('architecture') != 'frontier350-v1':
            raise ValueError(f'{label}: invalid validation report schema or architecture')
        if report.get('checkpoint_sha256') != sha or report.get('checkpoint_processed_tokens') != tokens:
            raise ValueError(f'{label}: checkpoint digest or token identity differs')
        finite_positive(report.get('validation_loss'), f'{label} validation_loss')
        finite_positive(report.get('validation_perplexity'), f'{label} validation_perplexity')
        if not isinstance(report.get('protocol_sha256'), str) or len(report['protocol_sha256']) != 64:
            raise ValueError(f'{label}: missing fixed evaluation protocol digest')
    if candidate_tokens <= anchor_tokens:
        raise ValueError('candidate did not advance beyond SAFE')
    for key in ('protocol_sha256', 'tokenizer_sha256', 'seq', 'batch', 'batches', 'seed', 'architecture'):
        if base.get(key) != candidate.get(key):
            raise ValueError(f'nonidentical paired validation protocol: {key}')
    safe_loss = finite_positive(base['validation_loss'], 'SAFE loss')
    cand_loss = finite_positive(candidate['validation_loss'], 'candidate loss')
    return {
        'schema': 1,
        'status': 'paired_validation_complete',
        'safe_checkpoint_sha256': anchor_sha,
        'candidate_checkpoint_sha256': candidate_sha,
        'protocol_sha256': base['protocol_sha256'],
        'validation_tokens_evaluated_per_model': base['seq'] * base['batch'] * base['batches'],
        'safe_validation_loss': safe_loss,
        'candidate_validation_loss': cand_loss,
        'loss_delta_candidate_minus_safe': cand_loss - safe_loss,
        'safe_validation_perplexity': finite_positive(base['validation_perplexity'], 'SAFE perplexity'),
        'candidate_validation_perplexity': finite_positive(candidate['validation_perplexity'], 'candidate perplexity'),
        'loss_not_worse_on_this_fixed_sample': cand_loss <= safe_loss,
        'promotion_allowed': False,
        'full_nis2_verified': False,
        'note': 'One fixed holdout sample is diagnostic only; no checkpoint selection or release. '
                'Earlier chat-core regression remains a separate blocker.',
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('base', 'candidate', 'anchor-sha', 'candidate-sha', 'out'):
        ap.add_argument('--' + name, required=True)
    ap.add_argument('--anchor-tokens', type=int, required=True)
    ap.add_argument('--candidate-tokens', type=int, required=True)
    args = ap.parse_args()
    base = json.loads(Path(args.base).read_text(encoding='utf-8-sig'))
    candidate = json.loads(Path(args.candidate).read_text(encoding='utf-8-sig'))
    report = assess(base, candidate, anchor_sha=args.anchor_sha,
                    candidate_sha=args.candidate_sha, anchor_tokens=args.anchor_tokens,
                    candidate_tokens=args.candidate_tokens)
    target = Path(args.out)
    with target.open('x', encoding='utf-8') as out:  # no overwrite, even on reruns
        json.dump(report, out, indent=2, ensure_ascii=False)
        out.write('\n')
    print(json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
