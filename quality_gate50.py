"""Evaluate whether completed 50M general-language pretraining is ready to move to SFT.

This gate does not claim release readiness. It deliberately ignores SOLVE/Helios tools and
does not require chat-format EOS from a base model that has not yet been instruction tuned.
"""
from __future__ import annotations
import argparse,json
from pathlib import Path


def load(p):return json.loads(Path(p).read_text(encoding='utf8'))

def mean_bpb(report,seq='1024'):
    vals=[]
    for lang in ('de','en'):
        row=report.get('true_byte_bpb',{}).get('languages',{}).get(lang,{})
        v=row.get('sequence_lengths',{}).get(seq,{}).get('bits_per_byte')
        if v is not None:vals.append(float(v))
    return sum(vals)/len(vals) if vals else None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--before-v3',required=True);ap.add_argument('--after-v3',required=True);ap.add_argument('--after-v4',required=True);ap.add_argument('--after-free',required=True);ap.add_argument('--training-summary',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
    b3=load(a.before_v3);a3=load(a.after_v3);a4=load(a.after_v4);free=load(a.after_free);training=load(a.training_summary)
    before_bpb=mean_bpb(b3);after_bpb=mean_bpb(a3)
    base=a3.get('free_generation',{}).get('base_continuation',{}).get('summary',{})
    fs=free.get('summary',{})
    pair=float(a4.get('pair_suite',{}).get('accuracy',0));context=float(a4.get('context_recall',{}).get('accuracy',0))
    repetition=float(base.get('mean_repeated_trigram_fraction',1));utf8=float(fs.get('valid_utf8_fraction',0))
    gates={
      'stage_completed':bool(training.get('complete')),
      'heldout_byte_bpb_improved':before_bpb is not None and after_bpb is not None and after_bpb < before_bpb,
      'v4_pair_accuracy_above_chance_margin':pair >= .55,
      'context_recall_above_chance':context > .50,
      'base_generation_repetition_below_30pct':repetition < .30,
      'utf8_at_least_99pct':utf8 >= .99,
    }
    ready=all(gates.values())
    report={
      'pretraining_stage_complete':bool(training.get('complete')),
      'ready_for_instruction_tuning':ready,
      'release_ready':False,
      'gates':gates,
      'measurements':{
        'before_mean_bpb_1024':before_bpb,'after_mean_bpb_1024':after_bpb,
        'v4_pair_accuracy_pct':100*pair,'context_recall_pct':100*context,
        'base_repeated_trigram_pct':100*repetition,'utf8_valid_pct':100*utf8,
      },
      'interpretation':'Passing means the base model is sufficiently stable to justify a separate instruction-tuning experiment. It is not a chatbot/release gate. Failing means gather/train better general-language data before SFT rather than hiding weakness with decoding penalties.',
    }
    Path(a.out).write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
