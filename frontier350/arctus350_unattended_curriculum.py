"""One day-only, bounded and monitored Arctus350 FP32-master SFT experiment.

Source: immutable native original-2B-based 64-step candidate. Reuses ONLY
local verified NERAL-created task references, source-declared Apache-2.0
human OASST2 chats, plus independently validated new synthetic DE/EN tasks.
No overwrite, external compute, external model, automatic promotion or release.
Hard cutoff: 28 October 2026, 21:30 Europe/Berlin, even on owner absence.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

ORIGINAL_2B_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
PARENT_SHA="57bbd1f70fcca6aee5e557b132d6b1b3f7b5e3378d0248923e5478bae36caf58"
CLEAN_SHA="fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31"
NEW_SHA="7e78a672786454c655b5dcf9e1cdc0967efee0d7381121623c3eebd8e93b7e1c"
LICENSE="Project-generated deterministic data; no external corpus"
MAX_WALL_SECONDS=7200
MAX_STEPS=3630
SEQ=192
QUOTAS={
    "exact":260, "small_math":260,"chat":800, "code":900,
    "math":650,"reasoning":600,"format":160,
}
BERLIN=ZoneInfo("Europe/Berlin")
LOCAL_CUTOFF=datetime(2026,10,28,21,30,tzinfo=BERLIN)


def permit_run(authorized,win_free,cuda_free,temp,iso_start,wall_seconds,cap):
    if (type(authorized) is not bool or type(win_free) is not int
        or type(cuda_free) is not int or min(win_free,cuda_free)<0
        or type(temp) not in (int,float) or not math.isfinite(temp)
        or type(wall_seconds) is not int or type(cap) not in (int,float)
        or not math.isfinite(cap)):
        raise ValueError("malformed GPU/date/time admission data")
    try: start=datetime.fromisoformat(iso_start)
    except (TypeError,ValueError) as e:raise ValueError("invalid ISO start timestamp") from e
    if start.tzinfo is None:raise ValueError("timezone-aware timestamp required")
    start_local=start.astimezone(BERLIN)
    return (authorized and win_free>=9216 and cuda_free>=9216
            and temp<80 and 0<wall_seconds<=MAX_WALL_SECONDS
            and 0<cap<=.68 and start_local.date()==LOCAL_CUTOFF.date()
            and start_local<LOCAL_CUTOFF
            and start_local.timestamp()+wall_seconds<=LOCAL_CUTOFF.timestamp())


def bounded_quotas():
    if sum(QUOTAS.values())!=MAX_STEPS:raise AssertionError("curriculum budget drift")
    return dict(QUOTAS)


def schedule_unique_rows(groups,quotas):
    if not isinstance(groups,dict) or not isinstance(quotas,dict) or not quotas:
        raise ValueError("invalid groups")
    if any(type(x) is not int or x<=0 or x>MAX_STEPS for x in quotas.values()):
        raise ValueError("invalid quotas")
    ids=[item[0] for group in groups.values() for item in group]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate cross-source training IDs")
    selected={}
    for key,quota in quotas.items():
        if key not in groups or len(groups[key])<quota:
            raise ValueError("insufficient verified, short, disjoint rows for "+key)
        selected[key]=sorted(groups[key],key=lambda x:(
            hashlib.sha256(x[0].encode("utf8")).hexdigest(),x[0]))[:quota]
    ordered=[]
    cursor=0
    while len(ordered)<sum(quotas.values()):
        for key in quotas:
            if cursor<len(selected[key]):
                ordered.append(selected[key][cursor])
        cursor+=1
    if len({x[0] for x in ordered})!=len(ordered):
        raise ValueError("invalid schedule repeated row")
    return ordered


def candidate_gate(target_before,target_after,human_before,human_after,rep_before,rep_after):
    vals=(target_before,target_after,human_before,human_after,rep_before,rep_after)
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in vals):
        raise ValueError("invalid validation observation")
    return (target_after<=target_before-.05
            and human_after<=human_before+.25
            and rep_after<=rep_before+.10)


def stop_conditions(elapsed,limit,temp,cuda_free,user_gpu_busy):
    if not all(type(x) is int and x>=0 for x in (elapsed,limit,cuda_free)) or type(user_gpu_busy) is not bool:
        raise ValueError("invalid monitor parameters")
    if elapsed>=limit:return "deadline"
    if temp is None:return "missing_gpu_telemetry"
    if type(temp) not in (int,float) or not math.isfinite(temp):
        return "missing_gpu_telemetry"
    if temp>=80:return "temperature"
    if cuda_free<2048:return "cuda_vram"
    if user_gpu_busy:return "user_gpu_busy"
    return None


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as handle:
        for b in iter(lambda:handle.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def gpu_temperature():
    try:
        cmd=subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu",
                            "--format=csv,noheader,nounits"],
                           capture_output=True,text=True,check=False,timeout=6)
        s=cmd.stdout.strip().splitlines()
        return int(s[0]) if cmd.returncode==0 and len(s)==1 else None
    except (ValueError,OSError,subprocess.TimeoutExpired):
        return None


def _encode_verified(tok,rows,max_seq,require_reference=False):
    from .chat350_sft_format import encode_messages
    from .arctus350_reference_answer_audit import inspect_reference
    from .arctus350_exact_math_curriculum import independently_verify_row
    accepted=[]
    for r in rows:
        if not isinstance(r,dict) or r.get("language") not in ("de","en"):
            continue
        if require_reference:
            if r["source"].startswith("neral-generated-exact-") or r["source"].startswith("neral-generated-small-add-"):
                if not independently_verify_row(r):continue
            elif len(r["messages"])!=2 or inspect_reference(
                r["domain"],r["messages"][0]["content"],r["messages"][1]["content"]
            )["status"]!="pass":
                continue
        try:
            e=encode_messages(tok,r["messages"],max_seq_len=max_seq)
        except ValueError:
            continue
        accepted.append((r["id"],e))
    return accepted


def dev_evaluate(model,tok):
    from .chat350_generation_metrics import summarize_generations
    from eval_chat350 import generate
    import torch
    path=Path(__file__).parent/"chat350_dev_cases.json"
    cases=json.loads(path.read_text(encoding="utf8"))
    if len(cases)!=12:raise ValueError("original 12 case dev suite changed")
    rows=[]
    model.eval()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for case in cases:
            ids,txt=generate(model,tok,case["prefix"],int(case["max_new_tokens"]),"cuda")
            out={"id":case["id"],"category":case["category"],
                 "generated_ids":ids,"text":txt[:512]}
            if "expected_exact" in case:out["expected_exact"]=case["expected_exact"]
            rows.append(out)
    report=summarize_generations(rows,int(tok.token_to_id("<|eos|>")))
    model.train()
    return {"summary":report,"rows":[{"id":r["id"],"text":r["text"]} for r in rows]}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent",required=True,type=Path)
    ap.add_argument("--tokenizer",required=True,type=Path)
    ap.add_argument("--clean-data",required=True,type=Path)
    ap.add_argument("--supplement",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--windows-free-mib",required=True,type=int)
    ap.add_argument("--windows-temperature-c",required=True,type=int)
    ap.add_argument("--max-wall-seconds",default=7200,type=int)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError("refuse overwriting prior experiment")
    if os.getenv("NERAL_LOCAL_SFT_ALLOW")!="1":
        raise PermissionError("one-time local GPU owner approval missing")
    start=datetime.now(timezone.utc)
    initial_temp=gpu_temperature()
    if initial_temp is None or initial_temp>=80:
        raise RuntimeError("missing/hot NVIDIA telemetry before CUDA load")
    import torch
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .train_chat350 import load_tokenizer,eval_sft,assistant_loss,collate
    from .model350 import FRONTIER_350,Neral350M
    from .secure_checkpoint350 import load_checkpoint
    from .train_precision350 import checkpoint_model_state
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA/BF16 not available")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("wrong CUDA device")
    free=torch.cuda.mem_get_info(0)[0]//2**20
    if not permit_run(True,a.windows_free_mib,free,max(initial_temp,a.windows_temperature_c),
                      start.isoformat(),a.max_wall_seconds,.65):
        raise RuntimeError("Wednesday-only time/temperature/VRAM admission failed")
    torch.cuda.set_per_process_memory_fraction(.65,device=0)
    torch.manual_seed(350_1028)
    torch.set_float32_matmul_precision("high")
    if sha(a.parent)!=PARENT_SHA:raise ValueError("source experimental 64-step SHA mismatch")
    if sha(a.tokenizer)!=EXPECTED_TOKENIZER_SHA256:raise ValueError("locked 16K BPE mismatch")
    m0,original=validate_dataset_root(a.clean_data)
    m1,supplement=validate_dataset_root(a.supplement)
    if m0["manifest_sha256"]!=CLEAN_SHA or m1["manifest_sha256"]!=NEW_SHA:
        raise ValueError("source or supplemental local split drift")
    validate_prompt_holdouts({x:original[x]+supplement[x] for x in ("train","val","test")})
    tok=load_tokenizer(a.tokenizer)
    src=original["train"]
    new=supplement["train"]
    gen_ok=lambda r:r["source"].startswith("neral-generated-") and r["license"]==LICENSE
    human_ok=lambda r:r["domain"]=="chat" and r["source"].startswith("oasst2-human-") and r["license"]=="Apache-2.0"
    groups={
        "exact":_encode_verified(tok,[r for r in new if r["domain"]=="format"],SEQ,True),
        "small_math":_encode_verified(tok,[r for r in new if r["domain"]=="math"],SEQ,True),
        "chat":_encode_verified(tok,[r for r in src if human_ok(r)],SEQ),
        "code":_encode_verified(tok,[r for r in src if r["domain"]=="code" and gen_ok(r)],SEQ,True),
        "math":_encode_verified(tok,[r for r in src if r["domain"]=="math" and gen_ok(r)],SEQ,True),
        "reasoning":_encode_verified(tok,[r for r in src if r["domain"]=="reasoning" and gen_ok(r)],SEQ),
        "format":_encode_verified(tok,[r for r in src if r["domain"]=="format" and gen_ok(r)],SEQ,True),
    }
    available={key:len(v) for key,v in groups.items()}
    quotas=bounded_quotas()
    print(json.dumps({"preflight_eligible_by_cohort":available,"quotas":quotas},
                     sort_keys=True),flush=True)
    training=schedule_unique_rows(groups,quotas)
    human_val=_encode_verified(tok,[r for r in original["val"] if human_ok(r)],SEQ)
    human_val=sorted(human_val,key=lambda x:hashlib.sha256(x[0].encode()).hexdigest())[:16]
    target_val=_encode_verified(tok,supplement["val"],SEQ,True)
    general_val=_encode_verified(tok,[r for r in original["val"]
                                    if gen_ok(r) and r["domain"] in ("code","math","reasoning","format")],SEQ,True)
    general_val=sorted(general_val,key=lambda x:hashlib.sha256(x[0].encode()).hexdigest())[:24]
    if len(human_val)<12 or len(target_val)<40 or len(general_val)<20:
        raise RuntimeError("not enough original-human/independent target/general VAL")
    checkpoint=load_checkpoint(a.parent)
    if (checkpoint.get("config")!=FRONTIER_350.to_dict() or
        checkpoint.get("source_checkpoint_sha256")!=ORIGINAL_2B_SHA or
        checkpoint.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256 or
        checkpoint.get("experimental_non_promotable") is not True):
        raise RuntimeError("untrusted checkpoint origin")
    model=Neral350M()
    model.load_state_dict(checkpoint["model"],strict=True)
    del checkpoint
    if torch.cuda.mem_get_info(0)[0]//2**20<9216:
        raise RuntimeError("other CUDA user claimed GPU before transfer")
    model=model.to("cuda",dtype=torch.float32)
    model.set_gradient_checkpointing(True)
    pad=int(tok.token_to_id("<|pad|>"))
    eos=int(tok.token_to_id("<|eos|>"))
    val_target=[r[1] for r in target_val]
    val_human=[r[1] for r in human_val]
    val_general=[r[1] for r in general_val]
    initial_target=eval_sft(model,val_target,1,pad,eos,"cuda")
    initial_human=eval_sft(model,val_human,1,pad,eos,"cuda")
    initial_general=eval_sft(model,val_general,1,pad,eos,"cuda")
    baseline_dev=dev_evaluate(model,tok)
    if torch.cuda.mem_get_info(0)[0]//2**20<4096:
        raise RuntimeError("GPU contention before optimizer")
    optimizer=torch.optim.AdamW(model.parameters(),lr=7.5e-6,betas=(.9,.95),
                                eps=1e-8,weight_decay=.025,fused=True)
    a.out.mkdir(parents=True,exist_ok=False)
    torch.cuda.reset_peak_memory_stats(0)
    start_time=time.monotonic()
    deadline_utc=LOCAL_CUTOFF.astimezone(timezone.utc)
    max_seconds=min(a.max_wall_seconds,
                    max(1,int((deadline_utc-start).total_seconds())))
    checks=[]
    losses=[]
    observed_temps=[]
    epochs=0
    trained_steps=0
    stop_reason="completed_budget"
    prior_target=initial_target["answer_nll"]
    best_nll=prior_target
    no_gain=0
    for idx,(_,sample) in enumerate(training):
        elapsed=int(time.monotonic()-start_time)
        if datetime.now(timezone.utc)>=deadline_utc:
            stop_reason="today_only_cutoff"
            break
        gpu_free=torch.cuda.mem_get_info(0)[0]//2**20
        if idx%25==0:
            current_temp=gpu_temperature()
            observed_temps.append(current_temp)
        reason=stop_conditions(elapsed,max_seconds,current_temp,gpu_free,False)
        if reason:
            stop_reason=reason
            break
        x,y=collate([sample],pad,"cuda")
        optimizer.zero_grad(set_to_none=True)
        loss=assistant_loss(model,x,y,eos,1.5)
        if not torch.isfinite(loss):
            stop_reason="nonfinite_loss"
            break
        loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        if not torch.isfinite(grad):
            stop_reason="nonfinite_gradient"
            break
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        del x,y,loss,grad
        trained_steps=idx+1
        if torch.cuda.max_memory_reserved(0)>8192*2**20:
            stop_reason="pytorch_gpu_memory_limit"
            break
        if trained_steps%400==0 or trained_steps==MAX_STEPS:
            t=eval_sft(model,val_target,1,pad,eos,"cuda")
            h=eval_sft(model,val_human,1,pad,eos,"cuda")
            g=eval_sft(model,val_general,1,pad,eos,"cuda")
            record={
                "step":trained_steps,"target":t,"human":h,"general":g,
                "elapsed_seconds":int(time.monotonic()-start_time),
            }
            checks.append(record)
            (a.out/"interim_aggregate.json").write_text(
                json.dumps({"checks":checks,"trained_steps":trained_steps,
                            "note":"non-promotable local experimental evidence only"},
                            ensure_ascii=False,indent=2)+"\n",encoding="utf8")
            print(json.dumps({"interim":record},ensure_ascii=False),flush=True)
            if (h["answer_nll"]>initial_human["answer_nll"]+.35 or
                g["answer_nll"]>initial_general["answer_nll"]+.40):
                stop_reason="validation_regression_stop"
                break
            if t["answer_nll"]>=best_nll-.02:
                no_gain+=1
            else:
                best_nll=t["answer_nll"]
                no_gain=0
            if no_gain>=2:
                stop_reason="validation_plateau_stop"
                break
            optimizer.zero_grad(set_to_none=True)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    final_target=eval_sft(model,val_target,1,pad,eos,"cuda")
    final_human=eval_sft(model,val_human,1,pad,eos,"cuda")
    final_general=eval_sft(model,val_general,1,pad,eos,"cuda")
    final_dev=dev_evaluate(model,tok)
    b=baseline_dev["summary"]
    c=final_dev["summary"]
    # A positive exact-instruction result AND no regression on *full* DEV
    # plus separately held-out human/general NLL are required to SAVE.
    from .chat350_generation_metrics import compare_health
    health=compare_health(b,c)
    decision=(candidate_gate(initial_target["answer_nll"],final_target["answer_nll"],
                             initial_human["answer_nll"],final_human["answer_nll"],
                             b["mean_repeated_trigram_fraction"],
                             c["mean_repeated_trigram_fraction"])
              and health["health_no_regression"]
              and c["exact_instruction_pct"]>0
              and final_general["answer_nll"]<=initial_general["answer_nll"]+.30
              and stop_reason not in ("nonfinite_loss","nonfinite_gradient","pytorch_gpu_memory_limit",
                                      "validation_regression_stop","user_gpu_busy","temperature",
                                      "missing_gpu_telemetry","cuda_vram"))
    if sha(a.parent)!=PARENT_SHA:
        raise ValueError("original parent checkpoint changed")
    candidate_sha=None
    if decision:
        if shutil.disk_usage(a.out.parent).free<2*2**30:
            raise RuntimeError("not enough disk for separate experimental checkpoint")
        candidate=a.out/"best_nonpromotable_experimental_only.pt"
        torch.save({
            "model":checkpoint_model_state(model,"fp32-master",resume=False),
            "config":FRONTIER_350.to_dict(),
            "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
            "source_checkpoint_sha256":ORIGINAL_2B_SHA,
            "parent_checkpoint_sha256":PARENT_SHA,
            "data_manifest_sha256":NEW_SHA,
            "clean_manifest_sha256":CLEAN_SHA,
            "chat_step":trained_steps,
            "experimental_non_promotable":True,
            "no_LM_replay":True,
            "frozen_benchmark_never_used":True,
        },candidate)
        candidate_sha=sha(candidate)
    report={
        "schema":"arctus350-today-only-guarded-fp32-curriculum-v1",
        "date_window":"2026-10-28 Europe/Berlin ONLY; hard stop 21:30",
        "status":"completed_nonpromotable" if trained_steps else "no_training_updates",
        "stop_reason":stop_reason,"trained_steps":trained_steps,
        "maximum_steps":MAX_STEPS,"elapsed_seconds":int(time.monotonic()-start_time),
        "train_quotas":quotas,"preflight_eligible":available,
        "human_val_rows":len(val_human),"general_val_rows":len(val_general),
        "target_val_rows":len(val_target),
        "target_before":initial_target,"target_after":final_target,
        "human_before":initial_human,"human_after":final_human,
        "general_before":initial_general,"general_after":final_general,
        "dev_before":baseline_dev,"dev_after":final_dev,
        "health_gate":health,"candidate_passes_dev_gate":bool(decision),
        "candidate_saved_local_only":bool(candidate_sha),"candidate_sha256":candidate_sha,
        "temperature_samples_c":observed_temps,
        "peak_pytorch_reserved_mib":torch.cuda.max_memory_reserved(0)//2**20,
        "peak_pytorch_allocated_mib":torch.cuda.max_memory_allocated(0)//2**20,
        "loss_first_8":losses[:8],"loss_last_8":losses[-8:],
        "interim_validation":checks,
        "original_2b_sha256":ORIGINAL_2B_SHA,"parent_sha256":PARENT_SHA,
        "original_data_sha256":CLEAN_SHA,"synthetic_data_sha256":NEW_SHA,
        "source_parent_unchanged":True,"cloud_cost_eur":0,
        "private_data_uploaded":False,"model_weights_uploaded":False,
        "automatic_promotion":False,"merge_or_release_allowed":False,
        "not_measured":"NIS2 and independent frozen final capability suite",
        "limitations":"One day's FP32 SFT; small development suite. Synthetic data and no foundation-LM replay; no claim of human-level competence.",
    }
    (a.out/"aggregate_report.json").write_text(
        json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
