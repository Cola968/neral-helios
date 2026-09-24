"""Bounded Chatbot-v1 behavior SFT sweep for the own Neral/Arctus 350M anchor.

This is a DEVELOPMENT experiment. It trains only the top four transformer blocks on
project-owned deterministic behavior data, evaluates every candidate against the unchanged
2.073B-token anchor, and keeps a candidate only when strict public-development gates pass.
No external SFT text, private holdout text, production promotion, or automatic merge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

from chatbot_v1_data import build_encoded, build_split, norm, prompt_fingerprints, text_sha
from helios350_preview import TOKENIZER_SHA, trusted_asset
from phoenix_continuity_cpu_probe import require_pinned_checkpoint
from phoenix_auto_sft_public_dev_eval import run_model, sha256_file
from phoenix_master_precision import require_fp32_trainable_subset
from sft_eval_contract import dynamic_batch_width, token_weighted_loss, validation_indices

HISTORICAL_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
PARAMETERS=349_825_152
SCHEMA="phoenix_chatbot_v1_behavior_pilot_v1"
CLASSIFICATION="project_owned_chatbot_v1_behavior_development_only"
CONFIGS=(
    {"name":"lr5e6_s60","lr":5e-6,"steps":60},
    {"name":"lr1e5_s60","lr":1e-5,"steps":60},
    {"name":"lr2e5_s40","lr":2e-5,"steps":40},
)
MAX_RESERVED=9*1024**3
PAIR_KEYS=(
    "nis2_instruction_following_pct",
    "nis2_dialogue_state_pct",
    "nis2_truthfulness_pct",
    "nis2_robustness_pct",
    "nis2_compositional_reasoning_pct",
)
PRIMARY_KEYS=("nis2_instruction_following_pct","nis2_dialogue_state_pct","nis2_exact_generation_pct")
PROTECTED_PAIR_KEYS=("nis2_truthfulness_pct","nis2_robustness_pct","nis2_compositional_reasoning_pct")


def _cpu_tree(torch,value):
    if torch.is_tensor(value):return value.detach().cpu()
    if isinstance(value,dict):return {k:_cpu_tree(torch,v) for k,v in value.items()}
    if isinstance(value,list):return [_cpu_tree(torch,v) for v in value]
    if isinstance(value,tuple):return tuple(_cpu_tree(torch,v) for v in value)
    return value


def atomic_save(torch,state:dict,path:Path)->None:
    if path.exists():raise FileExistsError("refusing candidate overwrite")
    temp=path.with_suffix(path.suffix+".tmp")
    if temp.exists():raise FileExistsError("stale candidate temp exists")
    try:
        torch.save(state,temp)
        with temp.open("rb") as handle:os.fsync(handle.fileno())
        os.replace(temp,path)
    finally:temp.unlink(missing_ok=True)


def development_eval_hashes()->set[str]:
    from benchmark_nis2_chat50 import PAIR_CASES,EXACT_GENERATION_CASES
    from helios_free_answer_dev_probe import selected_cases,instruction
    values=set()
    for _category,prefix,_good,_bad in PAIR_CASES:
        values.add(text_sha(norm(prefix)))
    for _ident,prompt,_accepted in EXACT_GENERATION_CASES:
        values.add(text_sha(norm(prompt)))
    for _ident,_domain,language,question,_expected,_meta in selected_cases():
        values.add(text_sha(norm(instruction(question,language))))
    return values


def verify_training_eval_disjoint()->dict:
    eval_hashes=development_eval_hashes()
    train_raw=build_split("train");valid_raw=build_split("validation")
    train_hashes=prompt_fingerprints(train_raw);valid_hashes=prompt_fingerprints(valid_raw)
    if train_hashes&valid_hashes:raise RuntimeError("behavior train/validation fingerprint overlap")
    if train_hashes&eval_hashes or valid_hashes&eval_hashes:
        raise RuntimeError("Chatbot-v1 behavior curriculum overlaps public development prompt bytes")
    return {
        "public_development_hashes_checked":len(eval_hashes),
        "train_prompt_fingerprints":len(train_hashes),
        "validation_prompt_fingerprints":len(valid_hashes),
        "exact_hash_overlap":0,
    }


def gate(baseline:dict,candidate:dict,val_before:float,val_after:float)->dict:
    delta={key:candidate[key]-baseline[key] for key in baseline
           if isinstance(baseline[key],(int,float)) and isinstance(candidate.get(key),(int,float))}
    primary_nonregression=all(delta.get(key,0)>=0 for key in PRIMARY_KEYS)
    primary_improvement=any(delta.get(key,0)>0 for key in PRIMARY_KEYS)
    protected_ok=all(delta.get(key,0)>=-12.5 for key in PROTECTED_PAIR_KEYS)
    free_nonempty_ok=candidate["free_answer_nonempty"]>=baseline["free_answer_nonempty"]-1
    free_digit_ok=candidate["free_answer_digit_prefix"]>=baseline["free_answer_digit_prefix"]
    validation_ok=val_after<val_before
    passed=all((primary_nonregression,primary_improvement,protected_ok,
                free_nonempty_ok,free_digit_ok,validation_ok))
    return {
        "passed":passed,
        "validation_improved":validation_ok,
        "primary_nonregression":primary_nonregression,
        "primary_improvement":primary_improvement,
        "protected_pair_regression_within_one_case":protected_ok,
        "free_answer_nonempty_protected":free_nonempty_ok,
        "free_answer_digit_prefix_protected":free_digit_ok,
        "delta_candidate_minus_base":delta,
    }


def selection_key(result:dict)->tuple:
    d=result["gate"]["delta_candidate_minus_base"]
    primary=sum(d.get(key,0.0) for key in PRIMARY_KEYS)
    protected=sum(d.get(key,0.0) for key in PROTECTED_PAIR_KEYS)
    return (primary,protected,-result["validation_loss_after"])


def main(argv=None)->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--execute",action="store_true")
    p.add_argument("--trusted-own-checkpoint",action="store_true")
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--assets-root",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args(argv)
    if not args.execute or not args.trusted_own_checkpoint:
        p.error("explicit execution and trust in the owner's checkpoint required")
    if args.out.exists() or args.out.is_symlink() or not args.out.is_absolute():
        p.error("new absolute output directory required")

    source=require_pinned_checkpoint(args.checkpoint)
    if sha256_file(source)!=HISTORICAL_SHA:raise ValueError("historical source SHA changed")
    assets=args.assets_root.resolve(strict=True)
    tokenizer_path=trusted_asset(assets,assets/"frontier350_assets"/"tokenizer.json",TOKENIZER_SHA)
    disjoint=verify_training_eval_disjoint()

    import phoenix_cuda_diagnostic as gpu_guard
    gpu_guard.require_stably_idle_gpu()
    import numpy as np
    import torch
    from tokenizers import Tokenizer
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from model350 import FRONTIER_350,Neral350M
    from phoenix_weight_only_pilot import _checked_historical_weights
    from protection_v9 import configure_trainable

    gpu_guard.require_cuda_ready(torch)
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    if tokenizer.get_vocab_size()!=FRONTIER_350.vocab_size:
        raise ValueError("tokenizer vocabulary mismatch")
    bos=tokenizer.token_to_id("<|bos|>");eos=tokenizer.token_to_id("<|eos|>")
    if bos is None or eos is None or bos==eos:raise ValueError("invalid BOS/EOS")
    splits,data_report=build_encoded(tokenizer,int(bos),int(eos),512)
    data_report.update(disjoint)

    historical=_checked_historical_weights(source,torch,Neral350M,FRONTIER_350.to_dict())
    base_weights=historical["model"];historical_tokens=int(historical.get("processed_tokens",-1))
    if historical_tokens!=2_073_161_216:raise ValueError("unexpected historical token anchor")
    device=torch.device("cuda")
    val_indices=validation_indices(len(splits["validation"]))

    def sft_batch(rows:list[dict],picks:list[int]):
        selected=[rows[i] for i in picks]
        width=dynamic_batch_width(selected,512)
        x=torch.full((len(selected),width),int(eos),dtype=torch.long)
        y=torch.full((len(selected),width),-100,dtype=torch.long)
        for i,row in enumerate(selected):
            n=len(row["input_ids"])
            x[i,:n]=torch.tensor(row["input_ids"],dtype=torch.long)
            y[i,:n]=torch.tensor(row["labels"],dtype=torch.long)
        return x.to(device),y.to(device)

    def fresh_model(dtype=torch.float32):
        model=Neral350M().to(device=device,dtype=dtype)
        model.load_state_dict(base_weights,strict=True)
        if model.parameter_report()["total"]!=PARAMETERS:raise RuntimeError("architecture changed")
        return model

    def validation(model)->float:
        model.eval();pairs=[]
        with torch.no_grad():
            for index in val_indices:
                x,y=sft_batch(splits["validation"],[index])
                with torch.autocast("cuda",dtype=torch.bfloat16):
                    _,loss=model(x,y)
                if loss is None or not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite behavior validation")
                pairs.append((float(loss),int((y!=-100).sum().item())))
        model.train()
        return token_weighted_loss(pairs)

    # Baseline model-only development vector.
    base_eval_model=fresh_model(torch.bfloat16).eval()
    baseline_metrics=run_model(torch,base_eval_model,tokenizer,int(bos),int(eos))["summary"]
    del base_eval_model;torch.cuda.empty_cache()
    base_val_model=fresh_model(torch.float32)
    baseline_val=validation(base_val_model)
    del base_val_model;torch.cuda.empty_cache()

    args.out.mkdir(parents=True,exist_ok=False)
    results=[];best=None
    for cfg in CONFIGS:
        gpu_guard.require_safe_runtime_temperature()
        seed=20260924
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed);np.random.seed(seed)
        torch.cuda.reset_peak_memory_stats()
        model=fresh_model(torch.float32)
        named=configure_trainable(model,4);params=[param for _name,param in named]
        trainable=sum(param.numel() for param in params)
        optimizer=torch.optim.AdamW(params,lr=cfg["lr"],betas=(0.9,0.95),eps=1e-8,
                                    weight_decay=0.01,fused=True)
        require_fp32_trainable_subset(model,optimizer,torch,require_moments=False)
        rng=random.Random(seed);supervised=0;losses=[];started=time.monotonic()
        model.train()
        for step in range(1,cfg["steps"]+1):
            gpu_guard.require_safe_runtime_temperature()
            picks=[rng.randrange(len(splits["train"])) for _ in range(4)]
            x,y=sft_batch(splits["train"],picks)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                _,loss=model(x,y)
            if loss is None or not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite behavior loss")
            loss.backward()
            grad=torch.nn.utils.clip_grad_norm_(params,1.0,error_if_nonfinite=True)
            if not math.isfinite(float(grad)):raise FloatingPointError("nonfinite behavior gradient")
            optimizer.step()
            if step==1:require_fp32_trainable_subset(model,optimizer,torch,require_moments=True)
            supervised+=int((y!=-100).sum().item());losses.append(float(loss))
            if torch.cuda.memory_reserved()>MAX_RESERVED:
                raise RuntimeError("Chatbot-v1 pilot exceeds 9 GiB reserved VRAM")
            if step==1 or step%20==0 or step==cfg["steps"]:
                print(json.dumps({"event":"chatbot_v1_step","config":cfg["name"],"step":step,
                    "train_loss":float(loss),"supervised_tokens":supervised},sort_keys=True),flush=True)

        elapsed=time.monotonic()-started
        after_val=validation(model)
        eval_model=Neral350M().to(device=device,dtype=torch.bfloat16)
        eval_model.load_state_dict(model.state_dict(),strict=True);eval_model.eval()
        candidate_metrics=run_model(torch,eval_model,tokenizer,int(bos),int(eos))["summary"]
        del eval_model;torch.cuda.empty_cache()
        gate_result=gate(baseline_metrics,candidate_metrics,baseline_val,after_val)
        result={
            "config":cfg,
            "trainable_parameters":trainable,
            "supervised_tokens":supervised,
            "training_wall_seconds":elapsed,
            "mean_train_loss":sum(losses)/len(losses),
            "validation_loss_before":baseline_val,
            "validation_loss_after":after_val,
            "metrics":candidate_metrics,
            "gate":gate_result,
            "peak_cuda_allocated_mib":round(torch.cuda.max_memory_allocated()/1024**2,1),
            "peak_cuda_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
            "candidate_sha256":None,
            "candidate_file":None,
        }
        if gate_result["passed"]:
            path=args.out/f"candidate_{cfg['name']}.pt"
            state={
                "checkpoint_schema":SCHEMA,
                "classification":CLASSIFICATION,
                "model":{name:tensor.detach().to("cpu",dtype=torch.float32)
                         for name,tensor in model.state_dict().items()},
                "config":FRONTIER_350.to_dict(),
                "tokenizer_sha256":TOKENIZER_SHA,
                "base_checkpoint_sha256":HISTORICAL_SHA,
                "historical_pretraining_tokens":historical_tokens,
                "sft_supervised_tokens":supervised,
                "sft_steps":cfg["steps"],
                "train_blocks":4,
                "learning_rate":cfg["lr"],
                "dataset_report":data_report,
                "development_metrics":candidate_metrics,
                "development_gate":gate_result,
                "master_parameter_dtype":"float32",
                "compute_autocast_dtype":"bfloat16",
                "production_eligible":False,
                "independent_holdout_checked":False,
                "quality_gain_verified":False,
            }
            atomic_save(torch,state,path)
            result["candidate_sha256"]=sha256_file(path)
            result["candidate_file"]=path.name
        results.append(result)
        if gate_result["passed"] and (best is None or selection_key(result)>selection_key(best)):
            best=result
        del model,optimizer;torch.cuda.empty_cache()

    source_unchanged=sha256_file(source)==HISTORICAL_SHA
    report={
        "schema":SCHEMA,
        "classification":CLASSIFICATION,
        "parameters":PARAMETERS,
        "historical_source_sha256":HISTORICAL_SHA,
        "historical_source_unchanged":source_unchanged,
        "dataset":data_report,
        "baseline":{"validation_loss":baseline_val,"metrics":baseline_metrics},
        "results":results,
        "development_gate_passed":best is not None,
        "selected_candidate":None if best is None else {
            "config":best["config"],"candidate_file":best["candidate_file"],
            "candidate_sha256":best["candidate_sha256"],"metrics":best["metrics"],
            "gate":best["gate"],
        },
        "external_sft_text_used":False,
        "private_holdout_opened":False,
        "production_eligible":False,
        "quality_gain_verified":False,
        "next_action":"discard_or_iterate_behavior_recipe" if best is None else "blind_free_answer_review_then_knowledge_lane",
    }
    (args.out/"report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({
        "event":"phoenix_chatbot_v1_pilot_complete",
        "development_gate_passed":report["development_gate_passed"],
        "selected_candidate":None if best is None else {
            "config":best["config"],"sha256":best["candidate_sha256"],
            "delta":best["gate"]["delta_candidate_minus_base"],
        },
        "historical_source_unchanged":source_unchanged,
        "production_eligible":False,
    },sort_keys=True),flush=True)
    if not source_unchanged:raise RuntimeError("historical source changed")
    return 0 if best is not None else 2


if __name__=="__main__":
    raise SystemExit(main())
