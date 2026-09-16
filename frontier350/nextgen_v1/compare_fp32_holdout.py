"""Pure-Python audit of paired FP32 validation reports; never promotes a model."""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def finite(x, label):
    if type(x) not in (int, float) or not math.isfinite(x):
        raise ValueError(f'invalid {label}')
    return float(x)


def validate(report, *, expected_sha, expected_tokens):
    if report.get('schema') != 1 or report.get('suite') != 'neral-nextgen-holdout-fp32-v1':
        raise ValueError('wrong FP32 report suite')
    if report.get('checkpoint_sha256') != expected_sha or report.get('checkpoint_processed_tokens') != expected_tokens:
        raise ValueError('checkpoint hash or token mismatch')
    if report.get('architecture') != 'frontier350-v1' or report.get('parameters') != 349825152:
        raise ValueError('model architecture mismatch')
    for field, value in (('read_only', True), ('training_started', False),
                         ('promotion_allowed', False), ('full_nis2_verified', False)):
        if report.get(field) is not value:
            raise ValueError(f'invalid safety flag: {field}')
    if report.get('validation_tokens') != 65536 or report.get('seed') != 351337:
        raise ValueError('invalid fixed validation sample')
    if not isinstance(report.get('per_batch'), list) or len(report['per_batch']) != 32:
        raise ValueError('incomplete FP32 batch loss list')
    total = 0.0
    for i, row in enumerate(report['per_batch']):
        if row.get('batch') != i+1 or row.get('tokens') != 2048:
            raise ValueError('batch index/token count mismatch')
        value = finite(row.get('nll_sum_fp32'), 'batch loss')
        mean = finite(row.get('mean_nll_fp32'), 'batch mean')
        if value <= 0 or mean <= 0 or not math.isclose(value / 2048, mean, abs_tol=1e-9):
            raise ValueError('batch loss accounting mismatch')
        total += value
    loss = finite(report.get('validation_loss_fp32'), 'overall loss')
    ppl = finite(report.get('validation_perplexity_fp32'), 'perplexity')
    if not math.isclose(total / 65536, loss, abs_tol=1e-8) or not math.isclose(math.exp(loss), ppl, rel_tol=1e-8):
        raise ValueError('aggregate loss or perplexity mismatch')
    return loss


def compare(base, mid, final, *, safe_sha, mid_sha, final_sha, safe_tokens=1010003584):
    pins = [safe_sha, mid_sha, final_sha]
    if len(set(pins)) != 3 or any(type(s) is not str or len(s) != 64 or any(ch not in '0123456789abcdef' for ch in s) for s in pins):
        raise ValueError('distinct externally pinned checkpoint SHA-256 required')
    tokens = [safe_tokens, safe_tokens+245760, safe_tokens+491520]
    reports = [base, mid, final]
    losses = [validate(r, expected_sha=s, expected_tokens=t) for r,s,t in zip(reports, pins, tokens)]
    for field in ('protocol_sha256','tokenizer_sha256','heldout_shard_sha256','seq','batch','batches','seed','validation_tokens'):
        if any(r.get(field) != base.get(field) for r in reports[1:]):
            raise ValueError('unmatched validation protocol: '+field)
    if type(base.get('protocol_sha256')) is not str or len(base['protocol_sha256']) != 64:
        raise ValueError('missing protocol SHA')
    batches = [[r['mean_nll_fp32'] for r in report['per_batch']] for report in reports]
    candidates = {}
    for name, i in (('midpoint',1), ('final',2)):
        deltas = [float(x-y) for x,y in zip(batches[i], batches[0])]
        candidates[name] = {'checkpoint_sha256':pins[i], 'tokens':tokens[i],
                            'validation_loss_fp32':losses[i],
                            'loss_delta_vs_safe':losses[i]-losses[0],
                            'batch_delta_mean':statistics.fmean(deltas),
                            'batch_delta_min':min(deltas), 'batch_delta_max':max(deltas),
                            'batches_with_lower_loss':sum(v<0 for v in deltas),
                            'batches_with_higher_loss':sum(v>0 for v in deltas),
                            'promotion_allowed': False}
    return {'schema':1,'status':'matched_fp32_development_diagnostic',
            'safe_checkpoint_sha256':safe_sha,'safe_validation_loss_fp32':losses[0],
            'protocol_sha256':base['protocol_sha256'], 'validation_tokens_per_model':65536,
            'candidates':candidates,'promotion_allowed':False,'full_nis2_verified':False,
            'note':'No winner or release selection. Correlated fixed windows and BF16 model weights limit inference; check generation, independent data and replication.'}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for flag in ('base','mid','final','safe-sha','mid-sha','final-sha','out'):
        ap.add_argument('--'+flag,required=True)
    args=ap.parse_args()
    reports=[json.loads(Path(p).read_text(encoding='utf-8-sig')) for p in (args.base,args.mid,args.final)]
    outcome=compare(*reports,safe_sha=args.safe_sha,mid_sha=args.mid_sha,final_sha=args.final_sha)
    with Path(args.out).open('x',encoding='utf-8') as stream:
        json.dump(outcome,stream,indent=2,ensure_ascii=False)
        stream.write('\n')
    print(json.dumps(outcome,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
