"""CPU-only frozen A/B research contract for native Arctus350.

This does NOT train models, attest existing checkpoints, run eval, approve
deployment or provide GPU authorization. The study design must be pinned
*before* inspecting candidate held-out scores. Real outcomes are compared
separately by arctus350_paired_evidence / arctus350_paired_stats.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
from pathlib import Path

from .chat350_contract import EXPECTED_TOKENIZER_SHA256

SCHEMA = "arctus350-preregistered-ablation-v1"
_TOP = frozenset({
    "schema","model_family","architecture","source_checkpoint_sha256",
    "tokenizer_sha256","train_manifest_sha256","val_manifest_sha256",
    "benchmark_sha256","evaluation_split","protected_test_access",
    "seed","budget","baseline","candidate","primary_metric",
    "secondary_metrics","intent",
})
_BUDGET = frozenset({
    "optimizer_updates","sequence_length","micro_batch_size",
    "gradient_accumulation","planned_assistant_target_tokens",
    "max_selection_rounds",
})
_ARM = frozenset({"peak_lr","eos_weight","sampling_mode"})
_SECONDARY = frozenset({
    "eos_rate","repeat_trigram_rate","german_exact_match",
    "english_exact_match","code_exact_match",
})
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class AblationError(ValueError):
    """Invalid, non-comparable or unsafe research pre-registration."""


def _no_duplicate_keys(pairs):
    result = {}
    for key,value in pairs:
        if key in result:
            raise AblationError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(_):
    raise AblationError("non-finite JSON values prohibited")


def read_ablation_json(text: str) -> dict:
    if not isinstance(text,str) or len(text.encode("utf-8"))>16384:
        raise AblationError("manifest too large")
    try:
        obj=json.loads(text,object_pairs_hook=_no_duplicate_keys,
                       parse_constant=_reject_constant)
    except (json.JSONDecodeError,UnicodeError,RecursionError) as exc:
        raise AblationError("invalid JSON") from exc
    if not isinstance(obj,dict):
        raise AblationError("manifest must be a JSON object")
    return obj


def _digest(value):
    raw=json.dumps(value,sort_keys=True,ensure_ascii=False,
                   separators=(",",":"),allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _int(value,name,minimum,maximum):
    if type(value) is not int or not minimum<=value<=maximum:
        raise AblationError(f"{name}: integer outside approved limits")
    return value


def _strict_sha(value,name):
    if not isinstance(value,str) or _SHA.fullmatch(value) is None:
        raise AblationError(f"{name}: lowercase SHA-256 required")
    return value


def _arm(row):
    if not isinstance(row,dict) or row.keys()!=_ARM:
        raise AblationError("only named training knobs allowed")
    lr=row["peak_lr"]
    eos=row["eos_weight"]
    if (type(lr) not in (int,float) or not math.isfinite(lr)
        or not 0 < lr <= 0.001):
        raise AblationError("peak_lr outside safe pre-registration bounds")
    if (type(eos) not in (int,float) or not math.isfinite(eos)
        or not 1 <= eos <= 3):
        raise AblationError("eos_weight outside existing trainer contract")
    if row["sampling_mode"] not in ("bucketed","compact","human-balanced"):
        raise AblationError("unknown sampling_mode")
    return row


def _budget(row):
    if not isinstance(row,dict) or row.keys()!=_BUDGET:
        raise AblationError("budget fields missing or unapproved")
    steps=_int(row["optimizer_updates"],"optimizer_updates",1,10000)
    seq=_int(row["sequence_length"],"sequence_length",32,8192)
    micro=_int(row["micro_batch_size"],"micro_batch_size",1,64)
    accum=_int(row["gradient_accumulation"],"gradient_accumulation",1,64)
    proposed=_int(row["planned_assistant_target_tokens"],
                  "planned_assistant_target_tokens",1,100_000_000)
    _int(row["max_selection_rounds"],"max_selection_rounds",1,1)
    max_capacity=steps*seq*micro*accum
    if proposed>max_capacity:
        raise AblationError("assistant target budget exceeds planned input-token capacity")
    # This capacity is a bound, NOT measured tokens or demonstrated throughput.
    return max_capacity


def freeze_ablation(study:dict) -> dict:
    """Validate one controlled comparison and pin the complete study SHA-256.

    Each arm inherits the *same* source/data, seed and budgets from one shared
    manifest. Differences are restricted to one explicitly named trainer knob.
    This cannot prove equal realized token exposure, clean holdouts or inference.
    """
    if not isinstance(study,dict) or study.keys()!=_TOP:
        raise AblationError("manifest fields missing or extra untrusted fields")
    if (study["schema"]!=SCHEMA or study["model_family"]!="neral-arctus350"
        or study["architecture"]!="frontier350-v1"):
        raise AblationError("only the native Arctus350 architecture is permitted")
    for key in ("source_checkpoint_sha256","tokenizer_sha256",
                "train_manifest_sha256","val_manifest_sha256","benchmark_sha256"):
        _strict_sha(study[key],key)
    if study["tokenizer_sha256"]!=EXPECTED_TOKENIZER_SHA256:
        raise AblationError("not the locked original Arctus 16K BPE tokenizer")
    if study["train_manifest_sha256"]==study["val_manifest_sha256"]:
        raise AblationError("train and validation manifests must be distinct")
    if study["evaluation_split"]!="val" or study["protected_test_access"] is not False:
        raise AblationError("TEST and frozen NIS2 may not be inspected for design selection")
    _int(study["seed"],"seed",0,(1<<32)-1)
    capacity=_budget(study["budget"])
    left=_arm(study["baseline"])
    right=_arm(study["candidate"])
    changed=[key for key in sorted(_ARM) if left[key]!=right[key]]
    if len(changed)!=1:
        raise AblationError("exactly ONE independent training knob must change")
    if study["primary_metric"]!="heldout_exact_match":
        raise AblationError("the primary heldout metric must be pre-registered")
    secondary=study["secondary_metrics"]
    if (not isinstance(secondary,list) or not secondary or
        any(type(x) is not str or x not in _SECONDARY for x in secondary) or
        len(set(secondary))!=len(secondary)):
        raise AblationError("secondary metrics must be named, unique and bounded")
    intent=study["intent"]
    if (not isinstance(intent,str) or not 32<=len(intent)<=500
        or any(ord(c)<32 or ord(c)==127 for c in intent)):
        raise AblationError("a bounded, clear scientific hypothesis is required")
    if len(json.dumps(study,ensure_ascii=False))>16384:
        raise AblationError("manifest too large")
    frozen=copy.deepcopy(study)
    frozen.update({
        "contract_sha256":_digest(study),
        "changed_variable":changed[0],
        "input_token_capacity_upper_bound":capacity,
        "status":"FROZEN_PLAN_NOT_RUN",
        "native_inference_verified":False,
        "training_executed":False,
        "release_authorization":"none",
        "limitations":(
            "A manifest hash verifies recorded design bytes only; source files, "
            "model weights, contamination and real token exposure are NOT verified. "
            "Same step count does not guarantee equal FLOPs or gradient updates. "
            "One comparison/seed cannot prove statistical confidence; use independent "
            "paired heldout evidence and keep TEST/NIS2 sealed."
        ),
    })
    return frozen


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse overwrite of pre-registered scientific record")
    frozen=freeze_ablation(read_ablation_json(args.manifest.read_text(encoding="utf-8")))
    # Exclusive create prevents races and does not silently revise the policy.
    with args.out.open("x",encoding="utf-8") as out:
        json.dump(frozen,out,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)
        out.write("\n")
    print("FROZEN_PLAN_NOT_RUN",frozen["contract_sha256"],flush=True)


if __name__=="__main__":
    main()
