"""Paired RTX ablation of *weights alone* versus original 100-case control.

A: already-measured untrained raw model flags from Actions #43242062250.
B: new raw QLoRA model, same messages, seed and decoder options, no tools.
C: new QLoRA proposal/structured extractor + exact PR537 guard, but ALWAYS
   original untrained model for *all* fallback answers.

No training, adapter changes, internet models, raw user text uploads or
execution of model-proposed Python outside the existing safe TIR sandbox.
Original 100-case control is pinned by its ordered correctness bitset:
the earlier per-case artifact is available in GitHub Actions #43242062250.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

from neral_tir_experimental import run_experimental_tir
from neral_tir_regression_guard import choose_guarded_tir_reply
from scripts.neral_generalization_100_cases import build, disjoint
from scripts.neral_generalization_100_run import (
    BASE_TAG, TUNED_TAG, SEED, SYSTEM, BASE_OPTIONS, VALID_POLICIES,
    raw_chat, parse_answer, query_structured, require_exact_guard,
    require_exact_models, gpu_offload_mib,
)

AUTH="NERAL_QWEN35_ABLATION_GPU_MANUAL_ONLY"
REFERENCE_RUN=43242062250
REFERENCE_ARTIFACT=13956491402
CASE_SHA="ecff6f8e88c2c0f6c1f6a7d2bc77f1c6f0f94dfa047f035f4d5157d458c02560"
IDS_SHA="716367ee76212b5669437f5450e7ba29825c65d1420f9106cc8f5742b15ae3da"

# These three 100-bit strings are copied from the original individual JSON case
# results, rather than assuming scores by category. Test pins counts and source.
OLD_CONTROL_BITS="0100000100011010110101101000000000000000000000000000001000010000100101000000010101101011010010101101"
OLD_GUARDED_BITS="0100000000011010100101001000000000000000000000000111101111011110111101111000010000101011011010101101"
OLD_VERIFIED_BITS="0000000000000000000000000000000000000000000000000011100111001110011100111000000000000000000000000000"

def require_prior_bitstrings(cases:list[dict])->None:
    if len(cases)!=100:
        raise ValueError("Only the original frozen 100 tasks can be compared")
    if (hashlib.sha256("\n".join(c["id"] for c in cases).encode()).hexdigest()!=IDS_SHA
        or len(set(c["id"] for c in cases))!=100):
        raise RuntimeError("100-case ID order differs from prior Actions artifact")
    if any(len(bits)!=100 or set(bits)-{"0","1"}
           for bits in (OLD_CONTROL_BITS, OLD_GUARDED_BITS, OLD_VERIFIED_BITS)):
        raise RuntimeError("Historical result bitstrings corrupt")
    if (OLD_CONTROL_BITS.count("1")!=29 or OLD_GUARDED_BITS.count("1")!=40
            or OLD_VERIFIED_BITS.count("1")!=15):
        raise RuntimeError("Historical score does not match signed evaluation source")
    if any(v=="1" and g!="1" for v,g in zip(OLD_VERIFIED_BITS,OLD_GUARDED_BITS)):
        raise RuntimeError("A verified answer in prior source was marked incorrect")

def baseline_by_id(cases:list[dict])->dict[str,bool]:
    require_prior_bitstrings(cases)
    return {c["id"]: OLD_CONTROL_BITS[i]=="1" for i,c in enumerate(cases)}

def classify_regressions(cases:list[dict])->list[dict]:
    require_prior_bitstrings(cases)
    return [
        {"id":c["id"],"category":c["category"],
         "question":c["question"],"expected_integer":c["expected_integer"],
         "original_guarded_correct":False,"original_control_correct":True,
         "original_guard_selection":"raw_qwen_already_selected"}
        for i,c in enumerate(cases)
        if OLD_CONTROL_BITS[i]=="1" and OLD_GUARDED_BITS[i]=="0"
    ]

DIAGNOSTIC_IDS=tuple(x["id"] for x in classify_regressions(build()))

def expected_counterfactual(cases:list[dict])->dict:
    """Prior 100-case *conditional replay*, NOT an inference measurement."""
    require_prior_bitstrings(cases)
    old=[x=="1" for x in OLD_CONTROL_BITS]
    verified=[x=="1" for x in OLD_VERIFIED_BITS]
    merged=[a or b for a,b in zip(old,verified)]
    return {"original_control_correct":sum(old),
            "original_guarded_correct":OLD_GUARDED_BITS.count("1"),
            "counterfactual_correct":sum(merged),
            "counterfactual_gained":sum(not a and b for a,b in zip(old,merged)),
            "counterfactual_lost":sum(a and not b for a,b in zip(old,merged)),
            "assumes_identical_raw_replays":True,
            "not_a_new_gpu_measurement":True}

def raw_qlora_case(case:dict,seed:int)->dict:
    """The ONLY model operation in this arm is one normal Ollama /api/chat."""
    start=time.monotonic()
    parsed=None;status="completed";answer_type="integer"
    try:
        answer=raw_chat(TUNED_TAG,[{"role":"system","content":SYSTEM},
                                 {"role":"user","content":case["question"]}],seed)
        parsed=parse_answer(answer)
        if parsed is None:answer_type="non_integer"
    except (RuntimeError,OSError,ValueError,TimeoutError):
        status="inference_error";answer_type="inference_error"
    return {"id":case["id"],"category":case["category"],
            "correct":parsed==case["expected_integer"],
            "output_integer":parsed,"answer_type":answer_type,
            "status":status,"seconds":round(time.monotonic()-start,3),
            "arm":"qlora_raw_no_tools"}

def dual_routed_case(case:dict,seed:int)->dict:
    """QLoRA ONLY on planning/schema calls; BASE on every untouched raw fallback.

    Critical: run_experimental_tir may execute base_fallback() *internally* and
    signal base_fallback_unverified. PR537 normally retains that raw answer.
    Routing only the guard's raw_respond to base is insufficient. Instead,
    the respond callback itself routes an EXACT unmodified original dialog to
    the untrained control and routes modified TIR prompts to the adapter.
    """
    start=time.monotonic()
    messages=[{"role":"system","content":SYSTEM},
              {"role":"user","content":case["question"]}]
    extraction_turn=0
    raw_calls=0;trained_tool_chat_calls=0
    def respond(turns):
        nonlocal raw_calls,trained_tool_chat_calls
        # Exact structural equality prevents TIR prompt contamination of raw.
        raw_original=(turns==messages)
        tag=BASE_TAG if raw_original else TUNED_TAG
        if raw_original:raw_calls+=1
        else:trained_tool_chat_calls+=1
        return raw_chat(tag,turns,seed)
    def extract(turns):
        nonlocal extraction_turn
        extraction_turn+=1
        if extraction_turn>2:
            raise RuntimeError("Model schema requested more than 2 times")
        return query_structured(TUNED_TAG,turns,seed=seed,
                                repair=extraction_turn==2)
    def raw_control(turns):
        nonlocal raw_calls
        if turns!=messages:
            raise ValueError("Nonoriginal dialogue passed to base fallback")
        raw_calls+=1
        return raw_chat(BASE_TAG,turns,seed)

    result=None;model_output_integer=None
    selection="error";verified=False;source="none";tool_calls=0
    status="completed";reason="other"
    try:
        proposal=run_experimental_tir(messages,respond,extract_constraints=extract)
        telemetry=proposal.get("first_turn_telemetry",{})
        if type(telemetry) is dict:
            reason=telemetry.get("extraction_rejection_reason") or telemetry.get("decision","other")
        report=choose_guarded_tir_reply(proposal,messages,raw_control)
        selection=report.get("selection_policy","error")
        if selection not in VALID_POLICIES:
            raise RuntimeError("Unrecognized guard selection")
        verified=report.get("verified") is True
        source=report.get("response_source","none")
        tool_calls=report.get("tool_calls_triggered",0)
        if selection.startswith("raw_qwen") and raw_calls!=1:
            raise RuntimeError("Dual route did not use exactly one original base fallback")
        if selection=="independently_verified_tir" and raw_calls:
            raise RuntimeError("Verified tool answer improperly re-ran base")
        model_output_integer=parse_answer(report.get("text"))
    except (RuntimeError,OSError,ValueError,TimeoutError):
        status="inference_or_guard_error"
    return {"id":case["id"],"category":case["category"],
            "correct":model_output_integer==case["expected_integer"],
            "output_integer":model_output_integer,
            "status":status,"seconds":round(time.monotonic()-start,3),
            "selection_policy":selection,"verified":verified,
            "response_source":source if source in {
                "raw_qwen","verified_tir","deterministic_safety_refusal"} else "other",
            "schema_attempts":extraction_turn,
            "model_tool_chat_calls":trained_tool_chat_calls,
            "raw_control_calls":raw_calls,
            "tool_calls_triggered":tool_calls if type(tool_calls) is int else None,
            "extraction_code":reason if type(reason) is str and len(reason)<60 else "other",
            "fallback_model":BASE_TAG if selection.startswith("raw_qwen") else None,
            "arm":"qlora_tools_base_raw_dual_route"}

def paired_comparison(cases:list[dict],candidate:list[dict],label:str)->dict:
    baseline=baseline_by_id(cases)
    if len(candidate)!=100 or len({r.get("id") for r in candidate})!=100:
        raise ValueError("Exactly 100 nonduplicate observations required")
    indexed={r["id"]:r for r in candidate}
    if set(indexed)!=set(baseline):
        raise ValueError("Candidate set differs from original 100 IDs")
    if any(r.get("status")!="completed" or type(r.get("correct")) is not bool
           for r in candidate):
        raise ValueError("Incomplete/errored GPU results cannot score")
    totals={};gain=loss=0
    for category in ("mathematics","logic","programming","mixed"):
        members=[x for x in cases if x["category"]==category]
        if len(members)!=25:raise ValueError("Not the exact 25-per-category corpus")
        a=sum(baseline[x["id"]] for x in members)
        b=sum(indexed[x["id"]]["correct"] for x in members)
        g=sum(not baseline[x["id"]] and indexed[x["id"]]["correct"] for x in members)
        l=sum(baseline[x["id"]] and not indexed[x["id"]]["correct"] for x in members)
        totals[category]={"control":a,"candidate":b,"gained":g,"lost":l,"total":25}
        gain+=g;loss+=l
    n=gain+loss
    pvalue=min(1.0,2*sum(math.comb(n,k) for k in range(min(gain,loss)+1))/2**n) if n else 1.0
    values=[r["seconds"] for r in candidate]
    return {"arm":label,"control_correct":sum(baseline.values()),
            "candidate_correct":sum(x["correct"] for x in candidate),
            "net_change":gain-loss,"gained":gain,"lost":loss,
            "paired_exact_mcnemar_two_sided_p":pvalue,"by_category":totals,
            "latency_p50_seconds":round(statistics.median(values),3),
            "latency_p95_seconds":round(sorted(values)[94],3)}

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--authorize",required=True)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--holdout",type=Path,required=True)
    p.add_argument("--corpus",type=Path,required=True)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    if args.authorize!=AUTH or os.environ.get("NERAL_EXPERIMENTAL_TIR")!="1":
        raise RuntimeError("No automatic GPU operation: explicit local authorization required")
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Refuse to overwrite prior ablation")
    progress=args.out.with_suffix(".progress.jsonl")
    if progress.exists() or progress.is_symlink():
        raise FileExistsError("Prior partial ablation already exists")
    cases=build()
    require_prior_bitstrings(cases)
    receipt=disjoint(cases,args.train,args.holdout)
    if receipt["case_sha256"]!=CASE_SHA:
        raise RuntimeError("The original 100-case evaluation corpus has changed")
    frozen=args.corpus.read_bytes()
    if hashlib.sha256(frozen).hexdigest()!=CASE_SHA:
        raise RuntimeError("Corpus file hash is not exactly the original 100")
    if [json.loads(s) for s in frozen.decode("utf8").splitlines() if s]!=cases:
        raise RuntimeError("Task content differs from original 100")
    if json.loads(args.manifest.read_text(encoding="utf8"))!=receipt:
        raise RuntimeError("Train/holdout/case hash manifest differs")
    require_exact_guard()
    hashes=require_exact_models()
    out=args.out.parent
    out.mkdir(parents=True,exist_ok=True)
    raw=[];dual=[]
    with progress.open("x",encoding="utf8",newline="\n") as f:
        for i,case in enumerate(cases):
            seed=SEED+i
            # Counterbalance order to mitigate local runtime/cache order bias.
            if i%2==0:
                pure=raw_qlora_case(case,seed)
                routed=dual_routed_case(case,seed)
            else:
                routed=dual_routed_case(case,seed)
                pure=raw_qlora_case(case,seed)
            if pure["status"]!="completed" or routed["status"]!="completed":
                raise RuntimeError("Model inference error; no partial score allowed")
            if max(gpu_offload_mib(BASE_TAG),gpu_offload_mib(TUNED_TAG))<4096:
                raise RuntimeError("Neither model is resident on RTX; refusing CPU-only fallback")
            raw.append(pure);dual.append(routed)
            f.write(json.dumps({"id":case["id"],"category":case["category"],
                    "pure_qlora_correct":pure["correct"],
                    "dual_routed_correct":routed["correct"],
                    "dual_policy":routed["selection_policy"],
                    "dual_schema_attempts":routed["schema_attempts"],
                    "dual_raw_calls":routed["raw_control_calls"]},sort_keys=True)+"\n")
            f.flush()
            print(f"ABLATION_PROGRESS {i+1}/100",flush=True)
    pure_stats=paired_comparison(cases,raw,"pure_qlora_raw_no_tools")
    dual_stats=paired_comparison(cases,dual,"qlora_tool_base_fallback_pr537")
    regression_ids=set(DIAGNOSTIC_IDS)
    regressions=[]
    old_results=classify_regressions(cases)
    for item in old_results:
        index=next(i for i,x in enumerate(cases) if x["id"]==item["id"])
        pure=raw[index];rt=dual[index]
        regressions.append({**item,
            "rerun_raw_qlora_output_integer":pure["output_integer"],
            "rerun_raw_qlora_correct":pure["correct"],
            "rerun_raw_qlora_answer_type":pure["answer_type"],
            "rerun_dual_output_integer":rt["output_integer"],
            "rerun_dual_correct":rt["correct"],
            "rerun_dual_policy":rt["selection_policy"],
            "rerun_schema_attempts":rt["schema_attempts"],
            "rerun_tool_calls_triggered":rt["tool_calls_triggered"],
            "rerun_extraction_code":rt["extraction_code"],
            "rerun_fallback_model":rt["fallback_model"],
            "caveat":"Original 100-case job did not capture incorrect numeric output; these are fresh same-profile replays."})
    record={"status":"REAL_LOCAL_RTX_100_ABLATION_WITH_PRIOR_FROZEN_CONTROL",
      "no_training":True,"reference_run":REFERENCE_RUN,
      "reference_artifact":REFERENCE_ARTIFACT,
      "case_sha256":CASE_SHA,
      "original_case_id_sha256":IDS_SHA,
      "train_sha256":receipt["train_sha256"],
      "old_holdout_sha256":receipt["prior_holdout_sha256"],
      "model_digests":hashes,"guard_git_blob":require_exact_guard(),
      "model_profile":{"temperature":0,"num_predict":2048,"num_ctx":4096,
                       "seed_start":SEED,"system":SYSTEM},
      "counterfactual_old_100":expected_counterfactual(cases),
      "pure_qlora":pure_stats,
      "dual_routing":dual_stats,
      "dual_policy_counts":dict(Counter(r["selection_policy"] for r in dual)),
      "dual_verified_count":sum(r["verified"] for r in dual),
      "dual_base_fallback_count":sum(bool(r["fallback_model"]) for r in dual),
      "error_regressions":regressions,
      "per_case":[{"id":case["id"],"category":case["category"],
            "historical_control_correct":baseline_by_id(cases)[case["id"]],
            "raw_qlora_correct":pure["correct"],"dual_correct":rt["correct"],
            "pure_qlora_integer_format":pure["output_integer"] is not None,
            "dual_policy":rt["selection_policy"],
            "dual_verified":rt["verified"],
            "schema_attempts":rt["schema_attempts"],
            "tool_calls_triggered":rt["tool_calls_triggered"],
            "dual_raw_control_calls":rt["raw_control_calls"],
            "pure_qlora_seconds":pure["seconds"],"dual_seconds":rt["seconds"]}
            for case,pure,rt in zip(cases,raw,dual)],
      "limitations":[
          "Historical control correctness reused from earlier GPU run, not replayed in ablation.",
          "The 100-case corpus is public synthetic development evidence, not a blind transfer benchmark.",
          "Pure QLoRA has no tools; dual route has exact-source Python verified tools, so comparison is architectural.",
          "A verified Python output does not imply new model weights are better.",
          "Raw answers differ from the historical control even at temperature 0 due to possible runtime nondeterminism.",
          "Original five wrong answer contents were not logged; diagnostic outputs here are replays."
        ]}
    with args.out.open("x",encoding="utf8",newline="\n") as f:
        json.dump(record,f,ensure_ascii=False,sort_keys=True,allow_nan=False)
        f.write("\n")
    print("ABLATION100_SCALARS "+json.dumps({k:v for k,v in record.items()
        if k not in ("per_case","error_regressions","limitations")},sort_keys=True),
        flush=True)
    print("ABLATION100_REGRESSION_DIAGNOSTICS "+json.dumps(regressions,ensure_ascii=False),flush=True)
    return 0

if __name__=="__main__":
    raise SystemExit(main())
