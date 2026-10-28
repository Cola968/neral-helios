"""One-day-only native Arctus350 math rescue from preserved 96-step FP32 parent.

Up to 256 genuinely new supervised updates. Balanced math/exact/OASST2 human.
Checks true, free-generated answers on exactly 76 withheld synthetic VAL rows
plus 12 established developer cases; never uses the synthetic TEST split.
No training source or model uploaded; rejects all generation regressions.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import time

PARENT_SHA="9c13b315465b74312c347effee8ef9c9e520ce95749ce9844a0a1fd2b66f48b2"
ORIGINAL_2B_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
CLEAN_SHA="fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31"
NEW_SHA="7e78a672786454c655b5dcf9e1cdc0967efee0d7381121623c3eebd8e93b7e1c"
CUTOFF=datetime(2026,10,28,21,0,tzinfo=ZoneInfo("Europe/Berlin"))
BUDGET=256
SEQUENCE=128
QUOTAS={"small_math":128,"exact":64,"chat":64}


def admit_math_rescue(authorized,win_free,cuda_free,temp,when,wall_seconds):
    if (type(authorized) is not bool or type(win_free) is not int
        or type(cuda_free) is not int or min(win_free,cuda_free)<0
        or type(temp) not in (float,int) or not math.isfinite(temp)
        or type(wall_seconds) is not int):
        raise ValueError("invalid time/GPU safety parameters")
    try: start=datetime.fromisoformat(when)
    except (ValueError,TypeError) as e:raise ValueError("invalid timestamp") from e
    if start.tzinfo is None:raise ValueError("timezone required")
    local=start.astimezone(ZoneInfo("Europe/Berlin"))
    return bool(authorized and win_free>=9216 and cuda_free>=9216 and temp<80
                and 0<wall_seconds<=1200 and local.date()==CUTOFF.date()
                and start.timestamp()+wall_seconds<=CUTOFF.timestamp())


def choose_repair_schedule(groups,quotas,prior_ids):
    if (not isinstance(groups,dict) or not isinstance(quotas,dict)
        or not isinstance(prior_ids,set) or not quotas or
        any(type(x) is not int or x<=0 or x>256 for x in quotas.values())):
        raise ValueError("malformed curriculum selection")
    ids=[r[0] for c in groups.values() for r in c]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate source row IDs")
    available={}
    for key,limit in quotas.items():
        rows=sorted((x for x in groups.get(key,[]) if x[0] not in prior_ids),
                    key=lambda x:(hashlib.sha256(x[0].encode()).hexdigest(),x[0]))
        if len(rows)<limit:
            raise ValueError("not enough novel training rows for "+key)
        available[key]=rows[:limit]
    result=[]
    for index in range(max(quotas.values())):
        for key in quotas:
            if index<len(available[key]):
                result.append(available[key][index])
    if len(result)!=sum(quotas.values()) or len(set(x[0] for x in result))!=len(result):
        raise ValueError("non-unique training schedule")
    return result


def keep_candidate(old_math,new_math,old_format,new_format,old_eos,new_eos,
                   old_rep,new_rep,old_human,new_human,old_dev_rep,new_dev_rep,
                   old_dev_eos,new_dev_eos,old_dev_exact,new_dev_exact):
    numbers=(old_math,new_math,old_format,new_format,old_eos,new_eos,old_rep,new_rep,
             old_human,new_human,old_dev_rep,new_dev_rep,old_dev_eos,new_dev_eos,
             old_dev_exact,new_dev_exact)
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in numbers):
        raise ValueError("nonfinite/invalid quality gate")
    return bool(
        new_math>old_math and new_format>=old_format
        and new_eos>=old_eos and new_rep<=old_rep+1e-12
        and new_human<=old_human+.20
        and new_dev_rep<=old_dev_rep+1e-12
        and new_dev_eos>=old_dev_eos
        and new_dev_exact>=old_dev_exact
    )


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(4*1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _measure_tasks(model,tok,heldout):
    import torch
    from eval_chat350 import generate
    from .arctus350_unseen_task_eval import record_aggregate
    from .arctus350_unattended_curriculum import gpu_temperature
    eos=int(tok.token_to_id("<|eos|>"))
    model.eval()
    observed=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for index,row in enumerate(heldout):
            if index%12==0:
                temp=gpu_temperature()
                if temp is None or temp>=80 or torch.cuda.mem_get_info(0)[0]//2**20<2048:
                    raise RuntimeError("cannot safely run genuine heldout generation")
            ids,generated=generate(
                model,tok,
                "User: "+row["messages"][0]["content"]+"\nAssistant:",
                16,"cuda",
            )
            content=ids[:ids.index(eos)] if eos in ids else ids
            observed.append({
                "id":row["id"],"domain":row["domain"],"language":row["language"],
                "expected":row["messages"][1]["content"],"generated":generated,
                "eos":eos in ids,"tokens":content,
            })
    model.train()
    return record_aggregate(observed)


def _math_word_totals(report):
    groups=report["by_family_language"]
    return {
        "math":sum(groups[x]["exact_correct"] for x in ("math/de","math/en")),
        "format":sum(groups[x]["exact_correct"] for x in ("format/de","format/en")),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent",required=True,type=Path)
    ap.add_argument("--tokenizer",required=True,type=Path)
    ap.add_argument("--clean",required=True,type=Path)
    ap.add_argument("--supplement",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--windows-free-mib",required=True,type=int)
    ap.add_argument("--windows-temp-c",required=True,type=int)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError("never overwrite existing experiment")
    if os.environ.get("NERAL_LOCAL_SFT_ALLOW")!="1":
        raise PermissionError("fresh own-machine owner approval required")
    from .arctus350_unattended_curriculum import gpu_temperature,dev_evaluate
    from .arctus350_exact_math_curriculum import independently_verify_row
    from .arctus350_targeted_exact_pilot import choose_schedule,_eligible
    from .arctus350_unseen_task_eval import validate_eval_partition
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .train_chat350 import eval_sft,assistant_loss,collate,load_tokenizer
    from .secure_checkpoint350 import load_checkpoint
    from .train_precision350 import checkpoint_model_state
    from .model350 import FRONTIER_350,Neral350M
    import torch

    start=datetime.now(timezone.utc)
    temp=gpu_temperature()
    if temp is None:raise RuntimeError("NVIDIA temperature telemetry missing")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 unavailable")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("wrong GPU")
    cuda_free=torch.cuda.mem_get_info(0)[0]//2**20
    if not admit_math_rescue(True,a.windows_free_mib,cuda_free,max(temp,a.windows_temp_c),start.isoformat(),1200):
        raise RuntimeError("Wednesday-only GPU admission refused")
    torch.cuda.set_per_process_memory_fraction(.65,device=0)
    torch.manual_seed(350_1028_2)
    torch.set_float32_matmul_precision("high")

    if sha(a.parent)!=PARENT_SHA or sha(a.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("pinned 96step model/tokenizer file changed")
    m0,original=validate_dataset_root(a.clean)
    m1,augmented=validate_dataset_root(a.supplement)
    if m0["manifest_sha256"]!=CLEAN_SHA or m1["manifest_sha256"]!=NEW_SHA:
        raise ValueError("dataset split manifest changed")
    validate_prompt_holdouts({s:original[s]+augmented[s] for s in ("train","val","test")})
    heldout=validate_eval_partition(augmented)
    if len(heldout)!=76 or any(not independently_verify_row(r) for r in heldout):
        raise ValueError("expected 76 intact unseen VAL reference rows")
    tok=load_tokenizer(a.tokenizer)
    format_rows=_eligible(tok,[r for r in augmented["train"] if r["domain"]=="format"])
    maths_rows=_eligible(tok,[r for r in augmented["train"] if r["domain"]=="math"])
    human_filter=lambda r:r["source"].startswith("oasst2-human-") and r["license"]=="Apache-2.0" and r["domain"]=="chat"
    human_rows=_eligible(tok,[r for r in original["train"] if human_filter(r)])
    if not all(independently_verify_row(r) for r in augmented["train"]):
        raise ValueError("new synthetic task answer label corrupted")
    previous=choose_schedule(format_rows,maths_rows,human_rows,32)
    prior_ids={r[0] for r in previous}
    groups={"small_math":maths_rows,"exact":format_rows,"chat":human_rows}
    scheduled=choose_repair_schedule(groups,QUOTAS,prior_ids)
    human_val=_eligible(tok,[r for r in original["val"] if human_filter(r)])
    human_val=sorted(human_val,key=lambda r:hashlib.sha256(r[0].encode()).hexdigest())[:16]
    if len(human_val)<16:
        raise RuntimeError("missing independent human heldout rows")

    ck=load_checkpoint(a.parent)
    if (ck.get("config")!=FRONTIER_350.to_dict() or
        ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256 or
        ck.get("source_checkpoint_sha256")!=ORIGINAL_2B_SHA or
        ck.get("experimental_non_promotable") is not True):
        raise ValueError("model lineage mismatch")
    model=Neral350M()
    model.load_state_dict(ck["model"],strict=True)
    del ck
    if torch.cuda.mem_get_info(0)[0]//2**20<9216:
        raise RuntimeError("GPU no longer free after data preflight")
    model=model.to("cuda",dtype=torch.float32)
    model.set_gradient_checkpointing(True)
    pad=int(tok.token_to_id("<|pad|>"))
    eos=int(tok.token_to_id("<|eos|>"))
    human_before=eval_sft(model,[r[1] for r in human_val],1,pad,eos,"cuda")
    baseline=_measure_tasks(model,tok,heldout)
    baseline_dev=dev_evaluate(model,tok)
    baseline_totals=_math_word_totals(baseline)
    if (baseline_totals["format"]!=32 or baseline_totals["math"]!=3
        or baseline["eos_count"]!=76 or baseline["repeated_trigram_fraction"]!=0
        or baseline_dev["summary"]["exact_instruction_pct"]!=50.0):
        raise RuntimeError("96-step baseline generation contract drift")
    if torch.cuda.mem_get_info(0)[0]//2**20<4096:
        raise RuntimeError("VRAM busy before new optimizer")

    optimizer=torch.optim.AdamW(model.parameters(),lr=4.0e-6,
                                betas=(.9,.95),eps=1e-8,
                                weight_decay=.025,fused=True)
    torch.cuda.reset_peak_memory_stats(0)
    a.out.mkdir(parents=True,exist_ok=False)
    start_time=time.monotonic()
    trials=[]
    losses=[]
    temps=[]
    chosen=None
    final_step=0
    reason="budget_complete"
    deadline=CUTOFF.astimezone(timezone.utc)
    for index,(_,encoded) in enumerate(scheduled):
        if datetime.now(timezone.utc)>=deadline or time.monotonic()-start_time>=1200:
            reason="today_only_time_stop"
            break
        if index%24==0:
            measured=gpu_temperature()
            temps.append(measured)
            if measured is None or measured>=80:
                reason="GPU_telemetry_or_heat_stop"
                break
        if torch.cuda.mem_get_info(0)[0]//2**20<2500:
            reason="cuda_vram_pressure"
            break
        x,y=collate([encoded],pad,"cuda")
        optimizer.zero_grad(set_to_none=True)
        loss=assistant_loss(model,x,y,eos,1.5)
        if not torch.isfinite(loss):raise RuntimeError("nonfinite training loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        if not torch.isfinite(norm):raise RuntimeError("nonfinite gradient")
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        del x,y,loss,norm
        if torch.cuda.max_memory_reserved(0)>8192*2**20:
            reason="GPU_memory_reserved_limit"
            break
        final_step=index+1
        if final_step in (128,256):
            result=_measure_tasks(model,tok,heldout)
            general=dev_evaluate(model,tok)
            h=eval_sft(model,[r[1] for r in human_val],1,pad,eos,"cuda")
            r=_math_word_totals(result)
            allowed=keep_candidate(
                baseline_totals["math"],r["math"],
                baseline_totals["format"],r["format"],
                baseline["eos_count"],result["eos_count"],
                baseline["repeated_trigram_fraction"],result["repeated_trigram_fraction"],
                human_before["answer_nll"],h["answer_nll"],
                baseline_dev["summary"]["mean_repeated_trigram_fraction"],
                general["summary"]["mean_repeated_trigram_fraction"],
                baseline_dev["summary"]["eos_fraction"],general["summary"]["eos_fraction"],
                baseline_dev["summary"]["exact_instruction_pct"],general["summary"]["exact_instruction_pct"],
            )
            observed={
                "step":final_step,"cohort_counts":r,"holdout":result,
                "human_val":h,"development":general,"strict_accept":allowed,
                "elapsed_seconds":int(time.monotonic()-start_time),
            }
            trials.append(observed)
            (a.out/"interim_aggregate.json").write_text(
                json.dumps({"baseline":baseline,"trials":trials},ensure_ascii=False,indent=2)+"\n",encoding="utf8")
            print(json.dumps({"trial_step":final_step,"math_correct":r["math"],
                              "format_correct":r["format"],"dev_exact_pct":general["summary"]["exact_instruction_pct"],
                              "strict_accept":allowed},ensure_ascii=False),flush=True)
            if allowed:
                if shutil.disk_usage(a.out.parent).free<2*2**30:
                    raise RuntimeError("insufficient local disk to save experimental candidate")
                path=a.out/"math_rescue_nonpromotable_only.pt"
                torch.save({
                    "model":checkpoint_model_state(model,"fp32-master",resume=False),
                    "config":FRONTIER_350.to_dict(),
                    "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
                    "source_checkpoint_sha256":ORIGINAL_2B_SHA,
                    "parent_checkpoint_sha256":PARENT_SHA,
                    "data_manifest_sha256":NEW_SHA,
                    "clean_manifest_sha256":CLEAN_SHA,
                    "experimental_non_promotable":True,
                    "steps_after_96":final_step,
                    "frozen_benchmark_never_used":True,
                },path)
                chosen=sha(path)
                reason="strict_gate_met_stop_before_overtraining"
                break
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    if sha(a.parent)!=PARENT_SHA:
        raise RuntimeError("immutable 96-step parent changed")
    report={
        "schema":"arctus350-targeted-small-arithmetic-256step-v1",
        "status":"completed_local_experiment","stop_reason":reason,
        "executed_steps":final_step,"max_steps":BUDGET,
        "train_quota":QUOTAS,"previous_96_train_ids_excluded":len(prior_ids),
        "baseline_task_accuracy":baseline,
        "baseline_human_val":human_before,
        "baseline_dev_12":baseline_dev,
        "trials":trials,
        "candidate_saved_local_only":bool(chosen),
        "candidate_sha256":chosen,
        "temperature_samples":temps,
        "peak_pytorch_reserved_mib":torch.cuda.max_memory_reserved(0)//2**20,
        "parent_sha256":PARENT_SHA,"parent_unchanged":True,
        "original_2b_sha256":ORIGINAL_2B_SHA,
        "train_model_uploaded":False,"raw_prompts_uploaded":False,
        "cloud_cost_eur":0,"automatic_release_allowed":False,
        "not_run":"NIS2 or sealed independent final benchmark",
    }
    (a.out/"aggregate_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
