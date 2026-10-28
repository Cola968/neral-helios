"""One-day-only FP32 SFT: frozen original Arctus350, last FFN replay experiment.

Only blocks.23.ffn gate/up/down are trainable, 10,838,016/349,825,152
(~3.10%) parameters. Loss scaling inverse to genuine observed domain FFN
gradients from the read-only probe, with balanced unique math DE/EN odd/even,
old exact-output replay and human-chat replay. Strict generation veto before
saving any isolated, NONPROMOTABLE experimental model. Never tests on TEST.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import hashlib
import json
import math
import os
import shutil
import time

PARENT_SHA="34f248b2392de3cbef51bae4300bb6bcf270dd8d7af7a95ce6c90bc56af37ff4"
ORIGINAL_2B_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
CLEAN_SHA="fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31"
V1_SHA="7e78a672786454c655b5dcf9e1cdc0967efee0d7381121623c3eebd8e93b7e1c"
V2_SHA="6354abc9b7731fac01842b30b1f44deb7929ba47f6589d768d069be1596553f8"
CUT_OFF=datetime(2026,10,28,20,30,tzinfo=ZoneInfo("Europe/Berlin"))
SAMPLES_PER_DOMAIN=96
N_STEPS=288
MAX_SECONDS=900
SEQ=128
MEASURED_NORMS={"math":1.6225994377651272,
                "format":0.7003853455282525,
                "human_chat":0.32576808021455705}
FFN_MATRIX_NAMES=tuple("blocks.23.ffn."+p+".weight" for p in ("gate","up","down"))


def choose_trainable_names(names):
    if not isinstance(names,list) or any(not isinstance(n,str) for n in names):
        raise ValueError("parameter name list required")
    if len(set(names))!=len(names):raise ValueError("duplicate tied/parameter name")
    picked=[n for n in names if n in FFN_MATRIX_NAMES]
    if set(picked)!=set(FFN_MATRIX_NAMES) or len(picked)!=3:
        raise ValueError("last FFN gate/up/down parameter topology drift")
    return picked


def normalized_task_scales(norms):
    if not isinstance(norms,dict) or set(norms)!=set(MEASURED_NORMS):
        raise ValueError("expected exactly three diagnostic gradient sources")
    if any(type(n) not in (int,float) or not math.isfinite(n) or n<=0 for n in norms.values()):
        raise ValueError("invalid gradient norm")
    low=min(norms.values())
    return {k:low/v for k,v in norms.items()}


def round_robin_replay(groups):
    if not isinstance(groups,dict) or set(groups)!={"math","format","human_chat"}:
        raise ValueError("expected all three supervised cohorts")
    if any(not isinstance(group,list) or not group for group in groups.values()):
        raise ValueError("no empty cohort")
    ids=[r[0] for rows in groups.values() for r in rows]
    if len(set(ids))!=len(ids):raise ValueError("training task identity repeated")
    selected=[]
    for i in range(max(len(x) for x in groups.values())):
        for kind in ("math","format","human_chat"):
            if i<len(groups[kind]):
                selected.append((kind,groups[kind][i]))
    return selected


def safe_candidate_gate(*,new_math_before,new_math_after,prior_word_before,
    prior_word_after,prior_math_before,prior_math_after,prior_eos_before,
    prior_eos_after,prior_rep_before,prior_rep_after,human_nll_before,
    human_nll_after,dev_exact_before,dev_exact_after,dev_eos_before,
    dev_eos_after,dev_rep_before,dev_rep_after):
    arguments=locals().values()
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in arguments):
        raise ValueError("invalid model quality metric")
    return bool(
        new_math_after>=new_math_before+2
        and prior_word_before==32 and prior_word_after==32
        and prior_math_after>=prior_math_before
        and prior_eos_after>=prior_eos_before
        and prior_rep_after<=prior_rep_before+1e-12
        and human_nll_after<=human_nll_before+.10
        and dev_exact_after>=dev_exact_before
        and dev_eos_after>=dev_eos_before
        and dev_rep_after<=dev_rep_before+1e-12
    )


def admission(authorized,windows_free,cuda_free,temp,start_time,seconds):
    if (type(authorized) is not bool or type(windows_free) is not int
        or type(cuda_free) is not int or min(windows_free,cuda_free)<0
        or type(temp) not in (int,float) or not math.isfinite(temp)
        or type(seconds) is not int):
        raise ValueError("invalid owner/GPU/time admission")
    try: start=datetime.fromisoformat(start_time)
    except (TypeError,ValueError) as exc:raise ValueError("invalid time ISO") from exc
    if start.tzinfo is None:raise ValueError("timezone offset required")
    local=start.astimezone(ZoneInfo("Europe/Berlin"))
    return bool(authorized and windows_free>=8192 and cuda_free>=8192
                and temp<80 and 0<seconds<=MAX_SECONDS
                and local.date()==CUT_OFF.date()
                and start.timestamp()+seconds<=CUT_OFF.timestamp())


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(4*1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def _encoded(tok,rows,verifier=None):
    from .chat350_sft_format import encode_messages
    out=[]
    for row in rows:
        if verifier is not None and not verifier(row):continue
        try: encoded=encode_messages(tok,row["messages"],max_seq_len=SEQ)
        except ValueError:continue
        out.append((row["id"],encoded))
    return out


def _take_sorted(rows,n,exclude=frozenset()):
    eligible=sorted((r for r in rows if r[0] not in exclude),
                    key=lambda r:(hashlib.sha256(r[0].encode("utf8")).hexdigest(),r[0]))
    if len(eligible)<n:raise ValueError("not enough independently sourced train examples")
    return eligible[:n]


def _task_summary(summary):
    by=summary["by_family_language"]
    return {
        "math":sum(by["math/"+l]["exact_correct"] for l in ("de","en")),
        "format":sum(by["format/"+l]["exact_correct"] for l in ("de","en")),
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for arg in ("parent","tokenizer","original","v1","v2","out"):
        parser.add_argument("--"+arg,required=True,type=Path)
    parser.add_argument("--windows-free-mib",required=True,type=int)
    parser.add_argument("--windows-temp-c",required=True,type=int)
    a=parser.parse_args()
    if a.out.exists():raise FileExistsError("never overwrite local experiment")
    if os.getenv("NERAL_LOCAL_SFT_ALLOW")!="1":
        raise PermissionError("user has not authorized one-time GPU experiment")
    import torch
    from .arctus350_unattended_curriculum import gpu_temperature,dev_evaluate
    from .arctus350_balanced_math_v2_pilot import balanced_unique_selection
    from .arctus350_balanced_arithmetic_v2 import parse_reference
    from .arctus350_math_rescue import _measure_tasks,choose_repair_schedule
    from .arctus350_targeted_exact_pilot import choose_schedule
    from .arctus350_unseen_task_eval import validate_eval_partition
    from .arctus350_exact_math_curriculum import independently_verify_row
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .secure_checkpoint350 import load_checkpoint
    from .train_chat350 import assistant_loss,collate,load_tokenizer,eval_sft
    from .train_precision350 import checkpoint_model_state
    from .model350 import FRONTIER_350,Neral350M

    now=datetime.now(timezone.utc)
    temp=gpu_temperature()
    if temp is None:raise RuntimeError("NVIDIA telemetry not available")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native CUDA BF16 needed")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("wrong GPU, only owner's local RTX4070 allowed")
    cuda_free=torch.cuda.mem_get_info(0)[0]//2**20
    if not admission(True,a.windows_free_mib,cuda_free,max(temp,a.windows_temp_c),
                     now.isoformat(),MAX_SECONDS):
        raise RuntimeError("today-only time/VRAM/thermal gate refused")
    torch.cuda.set_per_process_memory_fraction(.65,device=0)
    torch.manual_seed(350_20261028_4)
    torch.set_float32_matmul_precision("high")
    if digest(a.parent)!=PARENT_SHA or digest(a.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("best preserved native parent/tokenizer SHA mismatch")
    m0,d0=validate_dataset_root(a.original)
    m1,d1=validate_dataset_root(a.v1)
    m2,d2=validate_dataset_root(a.v2)
    if (m0["manifest_sha256"]!=CLEAN_SHA or m1["manifest_sha256"]!=V1_SHA
        or m2["manifest_sha256"]!=V2_SHA):
        raise ValueError("verified data manifest changed")
    for dataset in (d0,d1,d2):validate_prompt_holdouts(dataset)
    previous_val=validate_eval_partition(d1)
    new_v2_val=validate_eval_partition(d2)
    if len(previous_val)!=76 or len(new_v2_val)!=808:
        raise ValueError("recovered development splits differ")
    if not all(independently_verify_row(row) for row in previous_val):
        raise ValueError("original labels corrupt")
    if not all(parse_reference(row) is not None for row in new_v2_val):
        raise ValueError("v2 math development labels corrupt")
    tok=load_tokenizer(a.tokenizer)
    # 96 independent math training pairs across four language × parity buckets.
    from .chat350_sft_format import encode_messages
    math=[]
    for row in d2["train"]:
        pair=parse_reference(row)
        if pair is None:raise ValueError("source train math reference incorrect")
        try: encoding=encode_messages(tok,row["messages"],max_seq_len=SEQ)
        except ValueError:continue
        math.append((row["id"],row["language"],sum(pair)%2,tuple(sorted(pair)),encoding))
    selected_math=balanced_unique_selection(math,24)
    # select 8 per language/parity from existing v2 validation, never train on them.
    val_rows=[]
    for row in new_v2_val:
        pair=parse_reference(row)
        try: encode_messages(tok,row["messages"],max_seq_len=SEQ)
        except ValueError:continue
        val_rows.append((row["id"],row["language"],sum(pair)%2,tuple(sorted(pair)),row))
    selected_val=[r[4] for r in balanced_unique_selection(val_rows,8)]
    human=lambda r:r["domain"]=="chat" and r["source"].startswith("oasst2-human-") and r["license"]=="Apache-2.0"
    human_train=_encoded(tok,[r for r in d0["train"] if human(r)])
    exact_train=_encoded(tok,[r for r in d1["train"] if r["domain"]=="format"],independently_verify_row)
    small_train=_encoded(tok,[r for r in d1["train"] if r["domain"]=="math"],independently_verify_row)
    first=choose_schedule(exact_train,small_train,human_train,32)
    first_ids={row[0] for row in first}
    second=choose_repair_schedule(
        {"small_math":small_train,"exact":exact_train,"chat":human_train},
        {"small_math":128,"exact":64,"chat":64},first_ids)
    def old_used_in(rows):
        available={r[0]:r for r in rows}
        return [available[r[0]] for r in first+second if r[0] in available]
    def replay(kind_rows):
        used=old_used_in(kind_rows)
        if len(used)<64:raise RuntimeError("not enough prior SFT replay cases")
        return used[:64]+_take_sorted(kind_rows,32,{r[0] for r in used})
    training=round_robin_replay({
        "math":[(r[0],r[4]) for r in selected_math],
        "format":replay(exact_train),
        "human_chat":replay(human_train),
    })
    if len(training)!=N_STEPS:raise ValueError("training schedule drift")
    human_val=_encoded(tok,[r for r in d0["val"] if human(r)])
    human_val=_take_sorted(human_val,16)
    ck=load_checkpoint(a.parent)
    if (ck.get("config")!=FRONTIER_350.to_dict()
        or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256
        or ck.get("source_checkpoint_sha256")!=ORIGINAL_2B_SHA
        or ck.get("experimental_non_promotable") is not True):
        raise ValueError("wrong native parent lineage")
    model=Neral350M()
    model.load_state_dict(ck["model"],strict=True)
    del ck
    if torch.cuda.mem_get_info(0)[0]//2**20<8192:
        raise RuntimeError("owner GPU claimed after source validation")
    model=model.to("cuda",dtype=torch.float32)
    model.set_gradient_checkpointing(False)
    name_list=[n for n,p in model.named_parameters()]
    selected_names=choose_trainable_names(name_list)
    for name,param in model.named_parameters():
        param.requires_grad_(name in selected_names)
    params=[p for p in model.parameters() if p.requires_grad]
    trainable_count=sum(p.numel() for p in params)
    if len(params)!=3 or trainable_count!=10838016:
        raise RuntimeError("last-FFN trainable parameter count differs")
    pad=int(tok.token_to_id("<|pad|>"))
    eos=int(tok.token_to_id("<|eos|>"))
    before_new=_measure_tasks(model,tok,selected_val)
    before_old=_measure_tasks(model,tok,previous_val)
    before_dev=dev_evaluate(model,tok)["summary"]
    before_human=eval_sft(model,[row[1] for row in human_val],1,pad,eos,"cuda")
    original_task_scores=_task_summary(before_old)
    if (original_task_scores!={"math":5,"format":32}
        or before_old["eos_count"]!=76 or before_old["repeated_trigram_fraction"]!=0
        or before_dev["exact_instruction_pct"]!=50.0):
        raise RuntimeError("parent unchanged model behavior baseline mismatch")
    if torch.cuda.mem_get_info(0)[0]//2**20<2500:
        raise RuntimeError("GPU too busy for optimizer")
    scales=normalized_task_scales(MEASURED_NORMS)
    opt=torch.optim.AdamW(params,lr=1.0e-5,
                          betas=(.9,.95),eps=1e-8,weight_decay=.01,fused=True)
    a.out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    torch.cuda.reset_peak_memory_stats(0)
    trials=[]
    temps=[]
    finished=0
    stop="training_budget_complete"
    candidate_sha=None
    for i,(domain,(_,encoded)) in enumerate(training):
        if (datetime.now(ZoneInfo("Europe/Berlin"))>=CUT_OFF
            or time.monotonic()-started>=MAX_SECONDS):
            stop="today_only_deadline";break
        if i%24==0:
            t=gpu_temperature()
            temps.append(t)
            if t is None or t>=80:
                stop="thermal_or_missing_telemetry";break
        if torch.cuda.mem_get_info(0)[0]//2**20<2500:
            stop="GPU_VRAM_pressure";break
        x,y=collate([encoded],pad,"cuda")
        opt.zero_grad(set_to_none=True)
        loss=assistant_loss(model,x,y,eos,1.5)*scales[domain]
        if not torch.isfinite(loss):raise RuntimeError("nonfinite task loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(params,.7,error_if_nonfinite=True)
        if not torch.isfinite(norm):raise RuntimeError("nonfinite gradient")
        opt.step()
        torch.cuda.synchronize()
        del x,y,loss,norm
        finished=i+1
        if torch.cuda.max_memory_reserved(0)>7168*2**20:
            stop="GPU_reserved_memory_over_7GiB";break
        if finished in (96,192,288):
            after_new=_measure_tasks(model,tok,selected_val)
            after_old=_measure_tasks(model,tok,previous_val)
            after_dev=dev_evaluate(model,tok)["summary"]
            after_human=eval_sft(model,[r[1] for r in human_val],1,pad,eos,"cuda")
            scores=_task_summary(after_old)
            accept=safe_candidate_gate(
                new_math_before=before_new["exact_correct"],new_math_after=after_new["exact_correct"],
                prior_word_before=original_task_scores["format"],prior_word_after=scores["format"],
                prior_math_before=original_task_scores["math"],prior_math_after=scores["math"],
                prior_eos_before=before_old["eos_count"],prior_eos_after=after_old["eos_count"],
                prior_rep_before=before_old["repeated_trigram_fraction"],
                prior_rep_after=after_old["repeated_trigram_fraction"],
                human_nll_before=before_human["answer_nll"],
                human_nll_after=after_human["answer_nll"],
                dev_exact_before=before_dev["exact_instruction_pct"],
                dev_exact_after=after_dev["exact_instruction_pct"],
                dev_eos_before=before_dev["eos_fraction"],dev_eos_after=after_dev["eos_fraction"],
                dev_rep_before=before_dev["mean_repeated_trigram_fraction"],
                dev_rep_after=after_dev["mean_repeated_trigram_fraction"],
            )
            aggregate={"step":finished,
                       "new_math_32":after_new,"prior_76":after_old,"old_task_counts":scores,
                       "dev_12":after_dev,"human_validation":after_human,
                       "strict_pass":accept}
            trials.append(aggregate)
            (a.out/"interim_aggregate.json").write_text(json.dumps(
                {"comparisons":trials},ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
            print(json.dumps({"step":finished,
                "new_math_correct":after_new["exact_correct"],
                "old_words":scores["format"],"old_math":scores["math"],
                "dev_exact_pct":after_dev["exact_instruction_pct"],
                "strict_accept":accept}),flush=True)
            if accept:
                if shutil.disk_usage(a.out.parent).free<2*2**30:
                    raise RuntimeError("not enough local space to save a separated candidate")
                saved=a.out/"frozen_last_ffn_nonpromotable_only.pt"
                torch.save({
                    "model":checkpoint_model_state(model,"fp32-master",resume=False),
                    "config":FRONTIER_350.to_dict(),
                    "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
                    "source_checkpoint_sha256":ORIGINAL_2B_SHA,
                    "parent_checkpoint_sha256":PARENT_SHA,
                    "data_manifest_sha256":V2_SHA,
                    "experimental_non_promotable":True,
                    "only_trainable_parameters":selected_names,
                    "steps_after_prior_256":finished,
                    "sealed_benchmark_never_used":True,
                },saved)
                candidate_sha=digest(saved)
                stop="development_gate_pass_before_overtraining"
                break
    opt.zero_grad(set_to_none=True)
    del opt
    if digest(a.parent)!=PARENT_SHA:
        raise ValueError("pinned original best checkpoint modified")
    result={
        "schema":"arctus350-frozen-last-ffn-replay-gated-v1",
        "source_parent_sha256":PARENT_SHA,
        "native_foundation_sha256":ORIGINAL_2B_SHA,
        "source_manifests":{"original":CLEAN_SHA,"format_v1":V1_SHA,"math_v2":V2_SHA},
        "run_date":"2026-10-28 only",
        "status":"completed_experimental_only",
        "steps":finished,"max_steps":N_STEPS,"stop_reason":stop,
        "trainable_parameters":trainable_count,
        "total_native_parameters":349825152,
        "task_scales":scales,
        "train_cohorts":{"new_math":96,"old_format_replay_and_new":96,"human_chat_replay_and_new":96},
        "new_32_baseline":before_new,
        "prior_76_baseline":before_old,
        "human_baseline":before_human,
        "dev12_baseline":before_dev,
        "development_checks":trials,
        "saved_local_candidate":bool(candidate_sha),
        "saved_candidate_sha256":candidate_sha,
        "gpu_temperature_c_samples":temps,
        "peak_pytorch_gpu_reserved_mib":torch.cuda.max_memory_reserved(0)//2**20,
        "private_raw_text_or_weights_uploaded":False,
        "Hugging_Face_paid_jobs":0,
        "automatic_merge_or_release":False,
        "limitation":"Small, repeatedly consulted synthetic VAL/DEV are model-selection data; no unbiased sealed benchmark; not a general-purpose LLM readiness claim.",
    }
    (a.out/"aggregate_report.json").write_text(json.dumps(
        result,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
