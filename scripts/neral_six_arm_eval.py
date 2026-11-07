"""Six-arm *future* model-weight evaluation protocol with strict prompt parity.

Old dev72 comparisons gave Qwen three frozen examples while the two LoRA
models received none, confounding weight quality and prompting strategy.
This read-only, CPU-only scorer sets up a 3(models) x 2(0 or 3 shots)
factorial trial. It does NOT re-score historic runs or create GPU tasks.
It is an exploratory experiment protocol, not a release approval.
"""
from __future__ import annotations

from fractions import Fraction
from hashlib import sha256
import json

from scripts.neral_v6_fresh_dev_eval import (
    prepare_fewshot_messages, parse_answer, validate_variant_output,
    suite_fingerprint,
)
from scripts.neral_release_gate import exact_mcnemar_p

SIX_ARMS=(
    "qwen_zero","v5_zero","v61_zero",
    "qwen_three","v5_three","v61_three",
)
MODELS=("qwen","v5","v61")
SIZES={"zero":0,"three":3}


def _split_arm(arm:str)->tuple[str,str]:
    if arm not in SIX_ARMS:
        raise ValueError("Only six prespecified models x shot arms")
    label,shot=arm.rsplit("_",1)
    return label,shot


def messages_for_arm(case:dict,arm:str)->list[dict]:
    model,shot=_split_arm(arm)
    # Model identity does not affect prompts. The old helper encodes a
    # pinned same 3 examples, independent of model and checkpoint.
    msgs=prepare_fewshot_messages(
        case, "qwen_fewshot" if shot=="three" else "neral_v5")
    if len(msgs)!=2+2*SIZES[shot] or msgs[-1]["content"]!=case["prompt"]:
        raise ValueError("Broken locked few-shot comparison contract")
    return msgs


def match_shot_pair(candidate:str,reference:str)->bool:
    _,c=_split_arm(candidate)
    _,r=_split_arm(reference)
    if c!=r:
        raise ValueError("Never compare a 3-shot policy to a zero-shot one as a weight gain")
    return True


def evaluation_contract()->dict:
    return {
        "schema":1,"status":"NERAL_MODEL_PROMPT_PARITY_6ARM_CONTRACT",
        "arms":list(SIX_ARMS),"shot_conditions":[0,3],
        "per_arm_cases":72,
        "paired_comparisons_within_same_shot_only":True,
        "training_performed":False,"release_validated":False,
        "weight_superiority_proven":False,
        "adapter_promoted":False,
        "limitations":"A design contract only: synthetic/dev benchmarks remain non-independent; results must use a new sealed externally-reviewed suite.",
    }


def _compare(correctness:dict,a:str,b:str)->dict:
    match_shot_pair(a,b)
    aa,bb=correctness[a],correctness[b]
    better=sum(x and not y for x,y in zip(aa,bb))
    worse=sum(not x and y for x,y in zip(aa,bb))
    return {
        "improvements":better,"regressions":worse,
        "gap_pp":round(100*(better-worse)/len(aa),4),
        "exact_mcnemar_p_exploratory":exact_mcnemar_p(better,worse),
    }


def evaluate_six_arms(cases:list[dict],responses:dict)->dict:
    if (not isinstance(cases,list) or len(cases)!=72 or
            len({x.get("id") for x in cases})!=72 or
            not isinstance(responses,dict) or set(responses)!=set(SIX_ARMS)):
        raise ValueError("Six exactly aligned arms and 72 tasks mandatory")
    correctness={}
    scores={}
    for arm in SIX_ARMS:
        validate_variant_output(cases,responses[arm])
        results=[
            bool(row["complete"] and
                 parse_answer(case,row["text"])==Fraction(int(case["answer"])))
            for case,row in zip(cases,responses[arm])
        ]
        correctness[arm]=results
        scores[arm]={
            "correct":sum(results),
            "total":72,
            "ungraded_completed":sum(
                row["complete"] and parse_answer(case,row["text"]) is None
                for case,row in zip(cases,responses[arm])
            ),
            "incomplete":sum(not row["complete"] for row in responses[arm]),
            "by_domain":{
                domain:sum(results[i] for i,case in enumerate(cases)
                           if case["domain"]==domain)
                for domain in ("math","reasoning","code_trace")
            },
        }

    for case in cases:
        for shots in ("zero","three"):
            first=messages_for_arm(case,"qwen_"+shots)
            if any(messages_for_arm(case,m+"_"+shots)!=first for m in MODELS[1:]):
                raise ValueError("Model-specific prompt drift prohibited")
    paired={}
    for shot in ("zero","three"):
        paired[shot]={
            "v5_vs_qwen":_compare(correctness,"v5_"+shot,"qwen_"+shot),
            "v61_vs_qwen":_compare(correctness,"v61_"+shot,"qwen_"+shot),
            "v61_vs_v5":_compare(correctness,"v61_"+shot,"v5_"+shot),
        }
    return {
        **evaluation_contract(),
        "status":"NERAL_SIX_ARM_PAIRED_DEV_REPORT_NONRELEASE",
        "suite_sha256":suite_fingerprint(cases),
        "scores":scores,"comparisons":paired,
        "cross_shot_weight_comparisons_reported":False,
        "model_agnostic_prompts_verified":True,
        "training_performed":False,"release_validated":False,
        "weight_superiority_proven":False,"adapter_promoted":False,
    }
