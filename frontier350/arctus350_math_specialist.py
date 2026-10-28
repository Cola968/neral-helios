"""A math-ONLY native Arctus350 specialist fork; never replaces general NERAL.

Separate SHA-pinned 96+256 SFT model parent remains untouched. Up to 1600
FP32-master AdamW updates on 800 DISTINCT unordered number pairs, each with
two syntactically different, verified DE/EN prompts (4 balanced strata).
Every 400 updates: generate real answers on 32 v2 VAL + 44 protected v1 VAL.
Only save specialist on a strong, explicitly experimental arithmetic gate.
No NIS2/TEST evaluation, no app integration, cloud or automatic promotion.
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

PARENT_SHA="34f248b2392de3cbef51bae4300bb6bcf270dd8d7af7a95ce6c90bc56af37ff4"
ORIGINAL_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
V1_SHA="7e78a672786454c655b5dcf9e1cdc0967efee0d7381121623c3eebd8e93b7e1c"
V2_SHA="6354abc9b7731fac01842b30b1f44deb7929ba47f6589d768d069be1596553f8"
CUTOFF=datetime(2026,10,28,20,45,tzinfo=ZoneInfo("Europe/Berlin"))
MAX_WALL_SECONDS=1200
STEPS=1600
PAIRS_PER_BUCKET=200
SEQ=128


def can_run_today(owner,win_free,cuda_free,temp,when,seconds):
    if (type(owner) is not bool or type(win_free) is not int or
        type(cuda_free) is not int or min(win_free,cuda_free)<0 or
        type(temp) not in (int,float) or not math.isfinite(temp) or
        type(seconds) is not int):
        raise ValueError("bad safety telemetry")
    try: start=datetime.fromisoformat(when)
    except (ValueError,TypeError) as exc:raise ValueError("bad timezone aware time") from exc
    if start.tzinfo is None:raise ValueError("missing timezone")
    berlin=start.astimezone(ZoneInfo("Europe/Berlin"))
    return bool(owner and win_free>=8500 and cuda_free>=8500
                and temp<80 and 0<seconds<=MAX_WALL_SECONDS
                and berlin.date()==CUTOFF.date()
                and start.timestamp()+seconds<=CUTOFF.timestamp())


def choose_unique_math_pairs(rows,pairs_per_bucket):
    """Select exactly 2 task variants per pair, 200 pairs per language/parity."""
    if type(pairs_per_bucket) is not int or not 1<=pairs_per_bucket<=240:
        raise ValueError("invalid budget")
    if not isinstance(rows,list):raise ValueError("math data rows expected")
    ids=[r[0] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate task identity")
    bins={}
    for row in rows:
        if (not isinstance(row,tuple) or len(row)!=5 or row[1] not in ("de","en")
            or row[2] not in (0,1) or not isinstance(row[3],tuple)
            or len(row[3])!=2):
            raise ValueError("malformed verified math training row")
        pair=tuple(sorted(row[3]))
        bins.setdefault((row[1],row[2],pair),[]).append(row)
    selected={}
    occupied=set()
    for lang,parity in (("de",0),("en",0),("de",1),("en",1)):
        candidates=[]
        for (l,p,pair),variants in bins.items():
            if (l,p)==(lang,parity) and len(variants)>=2:
                candidates.append((pair,variants))
        candidates.sort(key=lambda v:(hashlib.sha256(
            ("math-spec-v1-%s-%s-%s"%(v[0][0],v[0][1],lang)).encode()).hexdigest(),v[0]))
        chosen=[]
        for pair,variants in candidates:
            if pair in occupied:continue
            variants.sort(key=lambda r:(hashlib.sha256(r[0].encode()).hexdigest(),r[0]))
            chosen.extend(variants[:2])
            occupied.add(pair)
            if len(chosen)==2*pairs_per_bucket:break
        if len(chosen)!=2*pairs_per_bucket:
            raise ValueError("insufficient unique verified math operand pairs for "+lang+str(parity))
        selected[(lang,parity)]=chosen
    plan=[]
    # First complete one pass of all unique pairs, THEN a second prompt view.
    for view in (0,1):
        for index in range(pairs_per_bucket):
            for key in (("de",0),("en",0),("de",1),("en",1)):
                plan.append(selected[key][index*2+view])
    if (len(plan)!=8*pairs_per_bucket or len({r[0] for r in plan})!=len(plan)
        or len({r[3] for r in plan})!=4*pairs_per_bucket):
        raise ValueError("math specialist pair leakage")
    return plan


def accept_specialist(*,new_before,new_after,prior_before,prior_after,
                      eos_count,total,repetition):
    values=locals().values()
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in values):
        raise ValueError("invalid output quality metric")
    return bool(total==32 and new_after>=max(12,new_before+8)
                and prior_after>=max(10,prior_before+5)
                and eos_count>=30 and 0<=repetition<=.02)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def count_math(report):
    by=report["by_family_language"]
    return sum(by["math/"+lang]["exact_correct"] for lang in ("de","en"))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("parent","tokenizer","v1","v2","out"):
        p.add_argument("--"+k,required=True,type=Path)
    p.add_argument("--windows-free-mib",required=True,type=int)
    p.add_argument("--windows-temperature-c",required=True,type=int)
    args=p.parse_args()
    if args.out.exists():raise FileExistsError("refuse overwrite existing experiment")
    if os.getenv("NERAL_LOCAL_SFT_ALLOW")!="1":
        raise PermissionError("one-time owner training approval missing")
    import torch
    from .arctus350_balanced_arithmetic_v2 import parse_reference
    from .arctus350_balanced_math_v2_pilot import balanced_unique_selection
    from .arctus350_unattended_curriculum import gpu_temperature
    from .arctus350_math_rescue import _measure_tasks
    from .arctus350_unseen_task_eval import validate_eval_partition
    from .arctus350_exact_math_curriculum import independently_verify_row
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .chat350_sft_format import encode_messages
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .secure_checkpoint350 import load_checkpoint
    from .train_chat350 import assistant_loss,collate,load_tokenizer
    from .train_precision350 import checkpoint_model_state
    from .model350 import FRONTIER_350,Neral350M

    first=datetime.now(timezone.utc)
    temperature=gpu_temperature()
    if temperature is None:raise RuntimeError("NVIDIA temperature unknown")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required on native RTX4070")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("unrecognized CUDA device")
    free=torch.cuda.mem_get_info(0)[0]//2**20
    if not can_run_today(True,args.windows_free_mib,free,
        max(temperature,args.windows_temperature_c),first.isoformat(),MAX_WALL_SECONDS):
        raise RuntimeError("Today-only GPU safety/admission did not pass")
    torch.cuda.set_per_process_memory_fraction(.65,device=0)
    torch.manual_seed(35020261028)
    torch.set_float32_matmul_precision("high")
    if file_sha(args.parent)!=PARENT_SHA or file_sha(args.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("original best parent / tokenizer checksum changed")
    m1,d1=validate_dataset_root(args.v1)
    m2,d2=validate_dataset_root(args.v2)
    if m1["manifest_sha256"]!=V1_SHA or m2["manifest_sha256"]!=V2_SHA:
        raise ValueError("locked arithmetic manifest drift")
    validate_prompt_holdouts(d1)
    validate_prompt_holdouts(d2)
    original_val=validate_eval_partition(d1)
    v2_val=validate_eval_partition(d2)
    if len(original_val)!=76 or len(v2_val)!=808:
        raise ValueError("heldout VAL split count changed")
    old_math=[r for r in original_val if r["domain"]=="math"]
    if len(old_math)!=44 or not all(independently_verify_row(x) for x in old_math):
        raise ValueError("old 44 arithmetic VAL damaged")
    if not all(parse_reference(row) is not None for row in v2_val):
        raise ValueError("new arithmetic VAL reference corrupt")
    tok=load_tokenizer(args.tokenizer)
    source=[]
    for row in d2["train"]:
        pair=parse_reference(row)
        if pair is None:raise ValueError("incorrect v2 training label")
        try: enc=encode_messages(tok,row["messages"],max_seq_len=SEQ)
        except ValueError:continue
        source.append((row["id"],row["language"],sum(pair)%2,tuple(sorted(pair)),enc))
    train=choose_unique_math_pairs(source,PAIRS_PER_BUCKET)
    val_rows=[]
    for row in v2_val:
        pair=parse_reference(row)
        try:encode_messages(tok,row["messages"],max_seq_len=SEQ)
        except ValueError:continue
        val_rows.append((row["id"],row["language"],sum(pair)%2,tuple(sorted(pair)),row))
    selected_val=[r[4] for r in balanced_unique_selection(val_rows,8)]
    parent=load_checkpoint(args.parent)
    if (parent.get("config")!=FRONTIER_350.to_dict()
        or parent.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256
        or parent.get("source_checkpoint_sha256")!=ORIGINAL_SHA
        or parent.get("experimental_non_promotable") is not True):
        raise RuntimeError("native 2B lineage/experimental only contract failed")
    model=Neral350M()
    model.load_state_dict(parent["model"],strict=True)
    del parent
    if torch.cuda.mem_get_info(0)[0]//2**20<8500:
        raise RuntimeError("RTX4070 claimed during dataset preflight")
    model=model.to("cuda",dtype=torch.float32)
    model.set_gradient_checkpointing(True)
    pad=int(tok.token_to_id("<|pad|>"))
    eos=int(tok.token_to_id("<|eos|>"))
    b_new=_measure_tasks(model,tok,selected_val)
    b_old=_measure_tasks(model,tok,old_math)
    if b_new["exact_correct"]!=0 or count_math(b_old)!=5:
        raise RuntimeError("pinned specialist parent behavior baseline mismatch")
    if torch.cuda.mem_get_info(0)[0]//2**20<4000:
        raise RuntimeError("VRAM busy before optimizer")
    opt=torch.optim.AdamW(model.parameters(),lr=8.0e-6,betas=(.9,.95),
                          eps=1e-8,weight_decay=.02,fused=True)
    args.out.mkdir(parents=True,exist_ok=False)
    torch.cuda.reset_peak_memory_stats(0)
    started=time.monotonic()
    checks=[]
    temps=[]
    executed=0
    reason="completed_budget"
    best_digest=None
    for idx,(_,_,_,_,enc) in enumerate(train):
        if (datetime.now(ZoneInfo("Europe/Berlin"))>=CUTOFF
            or time.monotonic()-started>=MAX_WALL_SECONDS):
            reason="today_only_time_stop";break
        if idx%24==0:
            t=gpu_temperature()
            temps.append(t)
            if t is None or t>=80:
                reason="missing_gpu_telemetry_or_heat";break
        if torch.cuda.mem_get_info(0)[0]//2**20<2000:
            reason="cuda_vram_pressure";break
        x,y=collate([enc],pad,"cuda")
        opt.zero_grad(set_to_none=True)
        loss=assistant_loss(model,x,y,eos,1.5)
        if not torch.isfinite(loss):raise RuntimeError("nonfinite specialist loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        if not torch.isfinite(norm):raise RuntimeError("nonfinite specialist gradient")
        opt.step()
        torch.cuda.synchronize()
        del x,y,loss,norm
        executed=idx+1
        if torch.cuda.max_memory_reserved(0)>8192*2**20:
            reason="pytorch_gpu_8gib_reserved_guard";break
        if executed%400==0:
            new=_measure_tasks(model,tok,selected_val)
            old=_measure_tasks(model,tok,old_math)
            correct=count_math(old)
            accepted=accept_specialist(
                new_before=b_new["exact_correct"],new_after=new["exact_correct"],
                prior_before=count_math(b_old),prior_after=correct,
                eos_count=new["eos_count"],total=new["total"],
                repetition=new["repeated_trigram_fraction"])
            row={"step":executed,"new_arithmetic_val_32":new,
                 "previous_arithmetic_val_44":old,"prior_correct":correct,
                 "specialist_gate":accepted}
            checks.append(row)
            (args.out/"interim_aggregate.json").write_text(
                json.dumps({"checks":checks},ensure_ascii=False,indent=2)+"\n",encoding="utf8")
            print(json.dumps({"step":executed,
                  "new_math_correct":new["exact_correct"],
                  "old_math_correct":correct,"specialist_gate":accepted}),flush=True)
            if accepted:
                if shutil.disk_usage(args.out.parent).free<2*2**30:
                    raise RuntimeError("no disk for separate specialist candidate")
                destination=args.out/"arctus350_math_specialist_nonpromotable_only.pt"
                torch.save({
                    "model":checkpoint_model_state(model,"fp32-master",resume=False),
                    "config":FRONTIER_350.to_dict(),
                    "source_checkpoint_sha256":ORIGINAL_SHA,
                    "parent_checkpoint_sha256":PARENT_SHA,
                    "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
                    "arithmetic_v2_manifest_sha256":V2_SHA,
                    "arithmetic_v1_manifest_sha256":V1_SHA,
                    "math_specialist_only":True,
                    "general_chat_unsafe":True,
                    "automatic_routing_disabled":True,
                    "experimental_non_promotable":True,
                    "frozen_nis2_and_TEST_not_used":True,
                    "specialist_steps":executed,
                },destination)
                best_digest=file_sha(destination)
                reason="math_only_gate_passed_before_further_training"
                break
    opt.zero_grad(set_to_none=True)
    del opt
    if file_sha(args.parent)!=PARENT_SHA:
        raise RuntimeError("immutable general parent modified")
    report={
        "schema":"arctus350-isolated-arithmetic-specialist-v1",
        "status":"completed_local_math_only_experiment",
        "parent_general_checkpoint_sha256":PARENT_SHA,
        "original_2b_sha256":ORIGINAL_SHA,
        "data_source_v2_sha256":V2_SHA,
        "is_separate_math_only_fork":True,
        "never_default_chat_model":True,
        "training_steps":executed,"maximum_steps":STEPS,
        "distinct_unordered_train_pairs":PAIRS_PER_BUCKET*4,
        "distinct_math_train_rows":STEPS,
        "heldout_new_math_before":b_new,
        "heldout_prior_math_before":b_old,
        "validation_checks":checks,
        "stop_reason":reason,
        "specialist_checkpoint_saved_local_only":bool(best_digest),
        "specialist_checkpoint_sha256":best_digest,
        "measured_temperature_c":temps,
        "max_pytorch_gpu_reserved_mib":torch.cuda.max_memory_reserved(0)//2**20,
        "user_texts_uploaded":False,
        "model_weights_uploaded":False,
        "hf_cloud_cost_eur":0,
        "public_release_or_merge":False,
        "limitations":"Specialist is trained exclusively for bounded addition templates; synthetic VAL reused for model selection, not independent general arithmetic verification. General dialog abilities not measured and should be considered at risk.",
    }
    (args.out/"aggregate_report.json").write_text(
        json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
