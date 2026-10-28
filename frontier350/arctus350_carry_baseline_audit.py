"""ONE read-only native Arctus350 audit on entirely distinct v3 (50..99) VAL.

32 balanced, independently validated samples: DE/EN × direct/column trace ×
carry/no carry, 4 each, no reused unordered operand pairs across cohorts.
Only aggregate counts/strict exact/final value/EOS/repetition are exported.
TEST and sealed benchmarks are untouched. No training or checkpoint writes.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime,timezone
from zoneinfo import ZoneInfo
from pathlib import Path
import argparse
import hashlib
import json
import math
import re

PARENT_SHA="34f248b2392de3cbef51bae4300bb6bcf270dd8d7af7a95ce6c90bc56af37ff4"
FOUNDATION_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
V3_MANIFEST_SHA="509f43447369a65eef7318cfbb54ef30832c3a886877f68826d15afe2b19f30c"
CUTOFF=datetime(2026,10,28,20,35,tzinfo=ZoneInfo("Europe/Berlin"))
MAX_WALL=600


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda:fh.read(4*1024*1024),b""):
            h.update(block)
    return h.hexdigest()


def recognize_verified_case(row):
    """Reparse and recompute reference from immutable template, not row metadata."""
    from .arctus350_carry_curriculum_v3 import TEMPLATES,_reply,verified_addition_trace,SOURCE,SOURCE_LICENSE
    if (not isinstance(row,dict) or row.get("source")!=SOURCE
        or row.get("license")!=SOURCE_LICENSE or row.get("domain")!="math"
        or row.get("language") not in ("de","en")):
        raise ValueError("v3 row source/metadata mismatch")
    turns=row.get("messages")
    if (not isinstance(turns,list) or len(turns)!=2
        or any(not isinstance(t,dict) for t in turns)
        or [t.get("role") for t in turns]!=["user","assistant"]):
        raise ValueError("expected one safe user/assistant pair")
    prompt=turns[0].get("content")
    answer=turns[1].get("content")
    if not isinstance(prompt,str) or not isinstance(answer,str):
        raise ValueError("source prompt/answer not text")
    for (lang,kind),template in TEMPLATES.items():
        if lang!=row["language"]:continue
        pattern=re.escape(template).replace(re.escape("{a}"),r"(\d+)").replace(re.escape("{b}"),r"(\d+)")
        m=re.fullmatch(pattern,prompt)
        if m is None:continue
        a,b=map(int,m.groups())
        calc=verified_addition_trace(a,b)
        if answer!=_reply(a,b,lang,kind):
            raise ValueError("independent exact answer re-verification failed")
        if row.get("task_type",kind)!=kind:
            raise ValueError("task-type label mismatch")
        return lang,kind,min(a,b),max(a,b),calc["carry"]
    raise ValueError("source v3 template not recognized")


def select_val_cases(rows,n_per_group=4):
    if type(n_per_group) is not int or not 1<=n_per_group<=8:
        raise ValueError("invalid small read-only sample quota")
    if not isinstance(rows,list):raise ValueError("VAL list required")
    ids=[r.get("id") for r in rows]
    if any(not isinstance(x,str) for x in ids) or len(set(ids))!=len(ids):
        raise ValueError("duplicated or invalid source row identity")
    groups={}
    for row in rows:
        lang,kind,a,b,carry=recognize_verified_case(row)
        key=(lang,kind,carry)
        groups.setdefault(key,[]).append((a,b,row))
    keys=[(lang,kind,carry) for lang in ("de","en")
          for kind in ("direct","trace") for carry in (0,1)]
    if set(groups)!=set(keys):raise ValueError("not all eight holdout cohorts")
    selected={}
    occupied=set()
    for key in keys:
        candidates=sorted(groups[key],key=lambda t:(
            hashlib.sha256((f"native-20261028-v3-{t[0]}-{t[1]}-{key}").encode()).hexdigest(),
            t[0],t[1]))
        accepted=[]
        for a,b,row in candidates:
            if (a,b) in occupied:continue
            occupied.add((a,b))
            accepted.append(row)
            if len(accepted)==n_per_group:break
        if len(accepted)!=n_per_group:
            raise ValueError("insufficient mutually distinct heldout operand pairs")
        selected[key]=accepted
    result=[]
    for i in range(n_per_group):
        for key in keys:
            result.append(selected[key][i])
    if len(result)!=8*n_per_group:raise AssertionError("heldout selection drift")
    return result


def score_observations(rows):
    """Never emits raw user prompts, IDs, generated text or token sequences."""
    from .chat350_generation_metrics import repeated_ngram_fraction
    if not isinstance(rows,list) or not rows:raise ValueError("empty observations")
    sums=Counter()
    groups={}
    repeat_total=0.
    for obs in rows:
        if not isinstance(obs,dict):raise ValueError("malformed observation")
        lang,kind,a,b,carry=recognize_verified_case(obs.get("row"))
        text=obs.get("text")
        if not isinstance(text,str) or type(obs.get("eos")) is not bool:
            raise ValueError("missing model generation")
        ids=obs.get("tokens")
        if not isinstance(ids,list) or not all(type(t) is int and t>=0 for t in ids):
            raise ValueError("invalid token IDs")
        expected=obs["row"]["messages"][1]["content"]
        strict=(text.strip()==expected)
        if kind=="direct":
            last_correct=text.strip()==str(a+b)
        else:
            terminal=r"(?:Ergebnis|Result):\s*(\d+)\.\s*$"
            m=re.search(terminal,text)
            last_correct=bool(m and int(m.group(1))==a+b)
        rep=repeated_ngram_fraction(ids,3)
        repeat_total+=rep
        key=f"{lang}/{kind}/"+("carry" if carry else "no_carry")
        group=groups.setdefault(key,Counter())
        for g in (sums,group):
            g["total"]+=1
            g["strict_exact"]+=int(strict)
            g["final_value_correct"]+=int(last_correct)
            g["eos_count"]+=int(obs["eos"])
            g["completed_strict"]+=int(strict and obs["eos"])
    return {
        "overall":{**dict(sorted(sums.items())),
                   "mean_repeated_trigram_fraction":repeat_total/len(rows)},
        "cohorts":{k:dict(sorted(v.items())) for k,v in sorted(groups.items())},
    }


def admit_readonly(owner,win_free,cuda_free,temp,now,seconds):
    if (type(owner) is not bool or type(win_free) is not int or
        type(cuda_free) is not int or min(win_free,cuda_free)<0 or
        type(temp) not in (int,float) or not math.isfinite(temp) or
        type(seconds) is not int):
        raise ValueError("invalid owner/NVIDIA telemetry")
    try:local=datetime.fromisoformat(now)
    except (ValueError,TypeError) as exc:raise ValueError("expected timezone ISO") from exc
    if local.tzinfo is None:raise ValueError("ambiguous local time")
    local=local.astimezone(ZoneInfo("Europe/Berlin"))
    return bool(owner and win_free>=8192 and cuda_free>=8192 and temp<80
                and 0<seconds<=MAX_WALL and local.date()==CUTOFF.date()
                and local.timestamp()+seconds<=CUTOFF.timestamp())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for k in ("parent","tokenizer","data","out"):
        parser.add_argument("--"+k,type=Path,required=True)
    parser.add_argument("--windows-free-mib",type=int,required=True)
    parser.add_argument("--windows-temp-c",type=int,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError("refuse overwrite aggregated report")
    import torch
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .arctus350_unseen_task_eval import validate_eval_partition
    from .arctus350_unattended_curriculum import gpu_temperature
    from .secure_checkpoint350 import load_checkpoint
    from .model350 import FRONTIER_350,Neral350M
    from .train_chat350 import load_tokenizer
    from eval_chat350 import generate

    t=gpu_temperature()
    if t is None:raise RuntimeError("missing independent GPU thermometer")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("RTX4070 CUDA BF16 inference is required")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("not owner's RTX4070")
    free=torch.cuda.mem_get_info(0)[0]//2**20
    if not admit_readonly(True,args.windows_free_mib,free,
                          max(t,args.windows_temp_c),datetime.now(timezone.utc).isoformat(),MAX_WALL):
        raise RuntimeError("do not use owner PC outside Wednesday permitted window")
    torch.cuda.set_per_process_memory_fraction(.65,device=0)
    if _sha(args.parent)!=PARENT_SHA or _sha(args.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("locked native parent or BPE SHA drift")
    manifest,splits=validate_dataset_root(args.data)
    if manifest["manifest_sha256"]!=V3_MANIFEST_SHA:
        raise ValueError("v3 data differs from original verified 5100 task set")
    validate_prompt_holdouts(splits)
    val=validate_eval_partition(splits)
    if len(val)!=484:raise ValueError("unexpected v3 development validation size")
    cases=select_val_cases(val,4)
    train_pairs={recognize_verified_case(row)[2:4] for row in splits["train"]}
    test_pairs={recognize_verified_case(row)[2:4] for row in splits["test"]}
    if ({recognize_verified_case(row)[2:4] for row in cases}&(train_pairs|test_pairs)):
        raise ValueError("numeric train/TEST leakage into v3 heldout sample")
    tok=load_tokenizer(args.tokenizer)
    ck=load_checkpoint(args.parent)
    if (ck.get("config")!=FRONTIER_350.to_dict()
        or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256
        or ck.get("source_checkpoint_sha256")!=FOUNDATION_SHA
        or ck.get("experimental_non_promotable") is not True):
        raise ValueError("non-native parent lineage or premature release")
    model=Neral350M()
    model.load_state_dict(ck["model"],strict=True)
    del ck
    if torch.cuda.mem_get_info(0)[0]//2**20<8192:
        raise RuntimeError("owner device occupied during CPU source checks")
    model=model.to("cuda",dtype=torch.float32)
    model.eval()
    eos=int(tok.token_to_id("<|eos|>"))
    results=[]
    temperature_samples=[]
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for i,row in enumerate(cases):
            if datetime.now(ZoneInfo("Europe/Berlin"))>=CUTOFF:
                raise RuntimeError("Wednesday-only inference cutoff reached")
            if i%4==0:
                temp=gpu_temperature()
                temperature_samples.append(temp)
                if temp is None or temp>=80 or torch.cuda.mem_get_info(0)[0]//2**20<2048:
                    raise RuntimeError("read-only NVIDIA thermal or VRAM guard")
            lang,kind,a,b,carry=recognize_verified_case(row)
            max_new=96 if kind=="trace" else 16
            ids,generated=generate(
                model,tok,"User: "+row["messages"][0]["content"]+"\nAssistant:",
                max_new,"cuda")
            before_eos=ids[:ids.index(eos)] if eos in ids else ids
            results.append({"row":row,"text":generated,"eos":eos in ids,"tokens":before_eos})
    aggregate=score_observations(results)
    if _sha(args.parent)!=PARENT_SHA:
        raise RuntimeError("protected general model bytes changed during diagnostic")
    report={
        "schema":"arctus350-v3-50to99-untrained-carry-readonly-baseline-v1",
        "parent_model_sha256":PARENT_SHA,
        "foundation_model_sha256":FOUNDATION_SHA,
        "v3_manifest_sha256":V3_MANIFEST_SHA,
        "status":"completed_read_only",
        "evaluation_set":"ONLY 32 v3 VAL examples (no TEST), 4 per 8 cohorts",
        "new_arithmetic_training_performed":False,
        "optimizer_created":False,
        "checkpoint_saved":False,
        "model_weights_modified":False,
        "model_or_raw_prompt_uploaded":False,
        "run_only_on_wednesday":True,
        "aggregate_results":aggregate,
        "temperature_samples_c":temperature_samples,
        "peak_pytorch_gpu_reserved_mib":torch.cuda.max_memory_reserved(0)//2**20,
        "limitations":"Model might have seen some number facts during original 2B pretraining; new v3 VAL has not been SFT-trained but is now used diagnostically, not sealed. Small synthetic sample; no general reasoning proof.",
    }
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps(report,sort_keys=True,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
