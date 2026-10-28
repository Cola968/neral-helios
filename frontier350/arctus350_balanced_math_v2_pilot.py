"""One-time Wednesday-only balanced arithmetic FP32 SFT from verified 256step parent.

Genuine generation tests on 32 *new* bilingual arithmetic VAL facts plus 76
previously withheld exact/math samples, original human chat and DEV12.
Original 2B native parent lineage, all original data, sealed tests immutable.
No cloud, no external model, no promotion; failure leaves prior parent intact.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

PARENT_SHA="34f248b2392de3cbef51bae4300bb6bcf270dd8d7af7a95ce6c90bc56af37ff4"
ORIG_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
CLEAN_SHA="fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31"
V1_SHA="7e78a672786454c655b5dcf9e1cdc0967efee0d7381121623c3eebd8e93b7e1c"
V2_SHA="6354abc9b7731fac01842b30b1f44deb7929ba47f6589d768d069be1596553f8"
LIMIT=datetime(2026,10,28,20,45,tzinfo=ZoneInfo("Europe/Berlin"))
STEPS=512
SEQ=128


def admit_today(owner,windows_free,cuda_free,temp,when,seconds):
    if (type(owner) is not bool or type(windows_free) is not int
        or type(cuda_free) is not int or min(windows_free,cuda_free)<0
        or type(temp) not in (float,int) or not math.isfinite(temp)
        or type(seconds) is not int):
        raise ValueError("invalid CPU-independent safety parameters")
    try: start=datetime.fromisoformat(when)
    except (ValueError,TypeError) as exc:raise ValueError("timezone date required") from exc
    if start.tzinfo is None:raise ValueError("ambiguous local clock")
    local=start.astimezone(ZoneInfo("Europe/Berlin"))
    return bool(owner and windows_free>=9216 and cuda_free>=9216 and temp<80
                and 0<seconds<=1800 and local.date()==LIMIT.date()
                and start.timestamp()+seconds<=LIMIT.timestamp())


def balanced_unique_selection(rows,per_bucket):
    """Four balanced DE/EN × odd/even strata; each operand pair selected once."""
    if type(per_bucket) is not int or not 1<=per_bucket<=160:
        raise ValueError("invalid balanced math selection size")
    if not isinstance(rows,list):
        raise ValueError("invalid math records")
    ids=[r[0] for r in rows]
    if len(ids)!=len(set(ids)):raise ValueError("repeated math training row ID")
    groups={}
    for lang in ("de","en"):
        for parity in (0,1):
            group=[r for r in rows if len(r)>=5 and r[1]==lang and r[2]==parity]
            groups[(lang,parity)]=sorted(group,key=lambda r:(
                hashlib.sha256(r[0].encode("utf8")).hexdigest(),r[0]))
    chosen={}
    seen_pairs=set()
    for key,group in groups.items():
        selected=[]
        for row in group:
            pair=tuple(sorted(row[3]))
            if pair in seen_pairs:continue
            seen_pairs.add(pair)
            selected.append(row)
            if len(selected)>=per_bucket:break
        if len(selected)<per_bucket:raise ValueError("insufficient distinct operands for "+str(key))
        chosen[key]=selected
    result=[]
    for index in range(per_bucket):
        for key in (("de",0),("en",0),("de",1),("en",1)):
            result.append(chosen[key][index])
    if len(result)!=per_bucket*4 or len({x[3] for x in result})!=len(result):
        raise ValueError("unsafe balanced selection")
    return result


def keep_v2_candidate(*,new_before,new_after,old_math_before,old_math_after,
                      old_words_before,old_words_after,old_eos_before,old_eos_after,
                      old_repeat_before,old_repeat_after,human_before,human_after,
                      dev_eos_before,dev_eos_after,dev_repeat_before,dev_repeat_after,
                      dev_exact_before,dev_exact_after):
    nums=locals().values()
    if any(type(x) not in (int,float) or not math.isfinite(x) or x<0 for x in nums):
        raise ValueError("invalid heldout health indicator")
    return bool(
        new_after>=new_before+3 and old_math_after>=old_math_before
        and old_words_after>=old_words_before and old_words_after==32
        and old_eos_after>=old_eos_before
        and old_repeat_after<=old_repeat_before+1e-12
        and human_after<=human_before+.15
        and dev_eos_after>=dev_eos_before
        and dev_repeat_after<=dev_repeat_before+1e-12
        and dev_exact_after>=dev_exact_before
    )


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as src:
        for chunk in iter(lambda:src.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _source_eligible(tok,rows):
    from .chat350_sft_format import encode_messages
    result=[]
    for row in rows:
        try: enc=encode_messages(tok,row["messages"],max_seq_len=SEQ)
        except ValueError:continue
        result.append((row["id"],enc))
    return result


def _round_robin(cohorts):
    if any(not c for c in cohorts.values()):
        raise ValueError("empty supervised cohort")
    if len({r[0] for c in cohorts.values() for r in c})!=sum(len(c) for c in cohorts.values()):
        raise ValueError("reused training row")
    result=[]
    for i in range(max(map(len,cohorts.values()))):
        for cohort in ("math","human","format"):
            if i<len(cohorts[cohort]):
                result.append(cohorts[cohort][i])
    return result


def _measure(model,tok,rows):
    from .arctus350_math_rescue import _measure_tasks
    return _measure_tasks(model,tok,rows)


def _totals(result):
    groups=result["by_family_language"]
    return {
        "math":sum(groups.get("math/"+lang,{}).get("exact_correct",0) for lang in ("de","en")),
        "format":sum(groups.get("format/"+lang,{}).get("exact_correct",0) for lang in ("de","en")),
    }


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent",required=True,type=Path)
    ap.add_argument("--tokenizer",required=True,type=Path)
    ap.add_argument("--original",required=True,type=Path)
    ap.add_argument("--v1",required=True,type=Path)
    ap.add_argument("--v2",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--windows-free-mib",required=True,type=int)
    ap.add_argument("--windows-temp-c",required=True,type=int)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError("refuse overwrite")
    if os.environ.get("NERAL_LOCAL_SFT_ALLOW")!="1":
        raise PermissionError("owner did not authorize local SFT")
    import torch
    from .arctus350_unattended_curriculum import gpu_temperature,dev_evaluate
    from .arctus350_unseen_task_eval import validate_eval_partition
    from .arctus350_balanced_arithmetic_v2 import parse_reference
    from .arctus350_exact_math_curriculum import independently_verify_row
    from .arctus350_targeted_exact_pilot import choose_schedule
    from .arctus350_math_rescue import choose_repair_schedule
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .train_chat350 import eval_sft,assistant_loss,collate,load_tokenizer
    from .secure_checkpoint350 import load_checkpoint
    from .train_precision350 import checkpoint_model_state
    from .model350 import FRONTIER_350,Neral350M

    start=datetime.now(timezone.utc)
    first_temperature=gpu_temperature()
    if first_temperature is None:raise RuntimeError("NVIDIA telemetry not available")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 unavailable")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("unexpected GPU; no training")
    free=torch.cuda.mem_get_info(0)[0]//2**20
    if not admit_today(True,a.windows_free_mib,free,
                       max(a.windows_temp_c,first_temperature),start.isoformat(),1800):
        raise RuntimeError("no safe Wednesday-only time/VRAM available")
    torch.cuda.set_per_process_memory_fraction(.65,device=0)
    torch.manual_seed(350_1028_3)
    torch.set_float32_matmul_precision("high")
    if sha(a.parent)!=PARENT_SHA or sha(a.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("immutable 256step parent or tokenizer SHA mismatch")
    manifest0,original=validate_dataset_root(a.original)
    manifest1,v1=validate_dataset_root(a.v1)
    manifest2,v2=validate_dataset_root(a.v2)
    if (manifest0["manifest_sha256"]!=CLEAN_SHA or
        manifest1["manifest_sha256"]!=V1_SHA or manifest2["manifest_sha256"]!=V2_SHA):
        raise ValueError("original or generated dataset manifest mismatch")
    for dataset in (original,v1,v2):validate_prompt_holdouts(dataset)
    tok=load_tokenizer(a.tokenizer)
    v2_val=validate_eval_partition(v2)
    orig_v1_val=validate_eval_partition(v1)
    if len(v2_val)!=808 or len(orig_v1_val)!=76:
        raise ValueError("unexpected or leaking heldout split")
    if not all(parse_reference(row) is not None for row in v2_val):
        raise ValueError("new math val labels untrustworthy")
    if not all(independently_verify_row(row) for row in orig_v1_val):
        raise ValueError("old VAL labels mutated")
    new_train=[]
    from .chat350_sft_format import encode_messages
    for row in v2["train"]:
        pair=parse_reference(row)
        if pair is None:raise ValueError("v2 training answer mismatch")
        try:enc=encode_messages(tok,row["messages"],max_seq_len=SEQ)
        except ValueError:continue
        new_train.append((row["id"],row["language"],sum(pair)%2,tuple(sorted(pair)),enc))
    math_rows=balanced_unique_selection(new_train,80)
    # 32 genuinely new arithmetic operand pairs, balanced languages/sum parity.
    new_val=[]
    for row in v2_val:
        pair=parse_reference(row)
        try:enc=encode_messages(tok,row["messages"],max_seq_len=SEQ)
        except ValueError:continue
        new_val.append((row["id"],row["language"],sum(pair)%2,tuple(sorted(pair)),row))
    selected_val=[r[4] for r in balanced_unique_selection(new_val,8)]
    human_ok=lambda r:r["domain"]=="chat" and r["source"].startswith("oasst2-human-") and r["license"]=="Apache-2.0"
    human=_source_eligible(tok,[r for r in original["train"] if human_ok(r)])
    small=_source_eligible(tok,[r for r in v1["train"] if r["domain"]=="math"])
    exact=_source_eligible(tok,[r for r in v1["train"] if r["domain"]=="format"])
    if not all(independently_verify_row(r) for r in v1["train"]):
        raise ValueError("unverified reference label in retained exact curriculum")
    prior=choose_schedule(exact,small,human,32)
    prior_ids={row[0] for row in prior}
    prior_rescue=choose_repair_schedule({
        "small_math":small,"exact":exact,"chat":human
    },{"small_math":128,"exact":64,"chat":64},prior_ids)
    prior_ids.update(row[0] for row in prior_rescue)
    def novel(rows,amount):
        eligible=sorted((r for r in rows if r[0] not in prior_ids),
                        key=lambda r:(hashlib.sha256(r[0].encode()).hexdigest(),r[0]))
        if len(eligible)<amount:raise ValueError("insufficient fresh supervised examples")
        return eligible[:amount]
    cohorts={
        "math":[(r[0],r[4]) for r in math_rows],
        "human":novel(human,128),
        "format":novel(exact,64),
    }
    train=_round_robin(cohorts)
    if len(train)!=STEPS or set(row[0] for row in train)&prior_ids:
        raise ValueError("unsafe/oversized training selection")
    human_val=_source_eligible(tok,[r for r in original["val"] if human_ok(r)])
    human_val=sorted(human_val,key=lambda r:hashlib.sha256(r[0].encode()).hexdigest())[:16]
    if len(human_val)<16:raise ValueError("human VAL too small")
    checkpoint=load_checkpoint(a.parent)
    if (checkpoint.get("config")!=FRONTIER_350.to_dict() or
        checkpoint.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256 or
        checkpoint.get("source_checkpoint_sha256")!=ORIG_SHA or
        checkpoint.get("parent_checkpoint_sha256")!="9c13b315465b74312c347effee8ef9c9e520ce95749ce9844a0a1fd2b66f48b2" or
        checkpoint.get("experimental_non_promotable") is not True):
        raise ValueError("native original 2B model lineage check failed")
    model=Neral350M()
    model.load_state_dict(checkpoint["model"],strict=True)
    del checkpoint
    if torch.cuda.mem_get_info(0)[0]//2**20<9216:
        raise RuntimeError("GPU not idle after data selection")
    model=model.to("cuda",dtype=torch.float32)
    model.set_gradient_checkpointing(True)
    pad=int(tok.token_to_id("<|pad|>"))
    eos=int(tok.token_to_id("<|eos|>"))
    human_before=eval_sft(model,[r[1] for r in human_val],1,pad,eos,"cuda")
    new_before=_measure(model,tok,selected_val)
    old_before=_measure(model,tok,orig_v1_val)
    dev_before=dev_evaluate(model,tok)["summary"]
    old_groups=_totals(old_before)
    if (old_groups["math"]!=5 or old_groups["format"]!=32 or old_before["eos_count"]!=76
        or old_before["repeated_trigram_fraction"]!=0.0 or
        dev_before["exact_instruction_pct"]!=50.0):
        raise RuntimeError("pinned 256step baseline no longer matches audited answer behavior")
    if torch.cuda.mem_get_info(0)[0]//2**20<4096:
        raise RuntimeError("VRAM pressure before optimizer")
    opt=torch.optim.AdamW(model.parameters(),lr=3.0e-6,betas=(.9,.95),
                          eps=1e-8,weight_decay=.025,fused=True)
    torch.cuda.reset_peak_memory_stats(0)
    a.out.mkdir(parents=True,exist_ok=False)
    start_clock=time.monotonic()
    checked=[]
    temps=[]
    executed=0
    reason="completed_budget"
    checkpoint_sha=None
    for i,(_,sample) in enumerate(train):
        if datetime.now(timezone.utc)>=LIMIT.astimezone(timezone.utc) or time.monotonic()-start_clock>=1800:
            reason="today_only_time_stop"
            break
        if i%25==0:
            current_temp=gpu_temperature()
            temps.append(current_temp)
            if current_temp is None or current_temp>=80:
                reason="thermal_or_telemetry_stop"
                break
        if torch.cuda.mem_get_info(0)[0]//2**20<2500:
            reason="cuda_vram_pressure"
            break
        x,y=collate([sample],pad,"cuda")
        opt.zero_grad(set_to_none=True)
        loss=assistant_loss(model,x,y,eos,1.5)
        if not torch.isfinite(loss):raise RuntimeError("nonfinite model loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        if not torch.isfinite(norm):raise RuntimeError("nonfinite gradient")
        opt.step()
        torch.cuda.synchronize()
        del x,y,loss,norm
        executed=i+1
        if torch.cuda.max_memory_reserved(0)>8192*2**20:
            raise RuntimeError("GPU reserved more than 8GiB")
        if executed in (256,512):
            current_new=_measure(model,tok,selected_val)
            current_old=_measure(model,tok,orig_v1_val)
            current_human=eval_sft(model,[r[1] for r in human_val],1,pad,eos,"cuda")
            current_dev=dev_evaluate(model,tok)["summary"]
            totals=_totals(current_old)
            accepted=keep_v2_candidate(
                new_before=new_before["exact_correct"],
                new_after=current_new["exact_correct"],
                old_math_before=old_groups["math"],
                old_math_after=totals["math"],
                old_words_before=old_groups["format"],
                old_words_after=totals["format"],
                old_eos_before=old_before["eos_count"],
                old_eos_after=current_old["eos_count"],
                old_repeat_before=old_before["repeated_trigram_fraction"],
                old_repeat_after=current_old["repeated_trigram_fraction"],
                human_before=human_before["answer_nll"],
                human_after=current_human["answer_nll"],
                dev_eos_before=dev_before["eos_fraction"],
                dev_eos_after=current_dev["eos_fraction"],
                dev_repeat_before=dev_before["mean_repeated_trigram_fraction"],
                dev_repeat_after=current_dev["mean_repeated_trigram_fraction"],
                dev_exact_before=dev_before["exact_instruction_pct"],
                dev_exact_after=current_dev["exact_instruction_pct"],
            )
            record={"step":executed,"new_v2_32":current_new,
                    "prior_v1_76":current_old,"original_human_nll":current_human,
                    "dev_12_metrics":current_dev,"strict_gate":accepted}
            checked.append(record)
            (a.out/"interim_aggregate.json").write_text(
                json.dumps({"checks":checked},ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
            print(json.dumps({"step":executed,"new_v2_exact_correct":current_new["exact_correct"],
                "old_v1_exact_correct":current_old["exact_correct"],
                "dev_exact_pct":current_dev["exact_instruction_pct"],
                "strict_candidate_gate":accepted},ensure_ascii=False),flush=True)
            if accepted:
                if shutil.disk_usage(a.out.parent).free<2*2**30:
                    raise RuntimeError("insufficient free local disk")
                candidate=a.out/"arctus350_math_v2_nonpromotable.pt"
                torch.save({
                    "model":checkpoint_model_state(model,"fp32-master",resume=False),
                    "config":FRONTIER_350.to_dict(),
                    "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
                    "source_checkpoint_sha256":ORIG_SHA,
                    "parent_checkpoint_sha256":PARENT_SHA,
                    "clean_manifest_sha256":CLEAN_SHA,
                    "math_v1_manifest_sha256":V1_SHA,
                    "math_v2_manifest_sha256":V2_SHA,
                    "experimental_non_promotable":True,
                    "steps_after_prior":executed,
                    "frozen_suite_never_used":True,
                },candidate)
                checkpoint_sha=sha(candidate)
                reason="genuine_math_gain_no_regression_early_stop"
                break
    opt.zero_grad(set_to_none=True)
    del opt
    if sha(a.parent)!=PARENT_SHA:
        raise RuntimeError("immutable source checkpoint was modified")
    report={
        "schema":"arctus350-balanced-bilingual-math-v2-gated-sft-v1",
        "status":"completed_nonpromotable","stop_reason":reason,
        "trained_steps":executed,"max_steps":STEPS,
        "training_cohorts":{"new_balanced_math":320,"fresh_human_chat":128,"fresh_exact_format":64},
        "v2_math_val_heldout_rows":len(selected_val),
        "v2_math_val_initial":new_before,
        "prior_v1_val_initial":old_before,
        "human_val_initial":human_before,
        "dev12_initial":dev_before,
        "checkpoints":checked,
        "candidate_saved_local_only":bool(checkpoint_sha),
        "candidate_sha256":checkpoint_sha,
        "parent_sha256":PARENT_SHA,
        "parent_unchanged":True,
        "original_2b_sha256":ORIG_SHA,
        "original_dataset_sha256":CLEAN_SHA,
        "v1_dataset_sha256":V1_SHA,
        "v2_dataset_sha256":V2_SHA,
        "gpu_temps_c":temps,
        "peak_gpu_reserved_mib":torch.cuda.max_memory_reserved(0)//2**20,
        "raw_prompts_uploaded":False,"weights_uploaded":False,
        "hf_gpu_jobs":0,"cloud_cost_eur":0,
        "automatic_promotion":False,"release_allowed":False,
        "limitations":"Small task-specific DEV and synthetic VAL reused for model selection; not a fresh unbiased benchmark. Original 2B language competence not proven.",
    }
    (a.out/"aggregate_report.json").write_text(
        json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=="__main__":
    main()
