"""Read-only native Arctus350 task-gradient audit on the final FFN ONLY.

Reports cosine similarity between averaged supervised math, exact-output and
human-chat last-block FFN gradients using 4 train-only cases per cohort,
balanced 2 German/2 English. Gradient cosine is local evidence of whether
tasks push the selected parameter subset in similar/opposing directions.
It is NOT causal proof of forgetting or a whole-model gradient measurement.
No optimization, model/dataset writes, test split evaluation, or raw exports.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib
import json
import math
import os
from pathlib import Path

PARENT_SHA="34f248b2392de3cbef51bae4300bb6bcf270dd8d7af7a95ce6c90bc56af37ff4"
ORIG_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
CLEAN_SHA="fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31"
V1_SHA="7e78a672786454c655b5dcf9e1cdc0967efee0d7381121623c3eebd8e93b7e1c"
V2_SHA="6354abc9b7731fac01842b30b1f44deb7929ba47f6589d768d069be1596553f8"
CUTOFF=datetime(2026,10,28,20,30,tzinfo=ZoneInfo("Europe/Berlin"))


def cosine_from_stats(dot,norm_a_sq,norm_b_sq):
    values=(dot,norm_a_sq,norm_b_sq)
    if any(type(x) not in (float,int) or not math.isfinite(x) for x in values):
        raise ValueError("invalid alignment statistics")
    if min(norm_a_sq,norm_b_sq)<=0:
        raise ValueError("zero gradient norm makes cosine undefined")
    denominator=math.sqrt(norm_a_sq)*math.sqrt(norm_b_sq)
    if abs(dot)>denominator*(1+1e-9):
        raise ValueError("invalid dot product compared with gradients")
    return max(-1.0,min(1.0,dot/denominator))


def choose_bilingual_probe(rows,n_per_language):
    if type(n_per_language) is not int or not 1<=n_per_language<=4:
        raise ValueError("invalid train-only sample quota")
    if not isinstance(rows,list):
        raise ValueError("nonlist source rows")
    ids=[r[0] for r in rows]
    if len(ids)!=len(set(ids)):
        raise ValueError("duplicate source row ID")
    result=[]
    for lang in ("de","en"):
        candidates=sorted((r for r in rows if len(r)>=3 and r[1]==lang),
                          key=lambda r:(hashlib.sha256(r[0].encode("utf8")).hexdigest(),r[0]))
        if len(candidates)<n_per_language:
            raise ValueError("missing genuine balanced examples")
        result.extend(candidates[:n_per_language])
    return result


def permit_readonly_probe(cuda_free_mib,temperature,iso_time):
    if (type(cuda_free_mib) is not int or cuda_free_mib<0
        or type(temperature) not in (int,float) or not math.isfinite(temperature)):
        raise ValueError("invalid GPU safety telemetry")
    try: now=datetime.fromisoformat(iso_time)
    except (ValueError,TypeError) as e:raise ValueError("invalid timestamp") from e
    if now.tzinfo is None:raise ValueError("offset-aware timestamp required")
    local=now.astimezone(ZoneInfo("Europe/Berlin"))
    return bool(cuda_free_mib>=9216 and temperature<80 and
                local.date()==CUTOFF.date() and local<CUTOFF)


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as src:
        for b in iter(lambda:src.read(1024*1024),b""):
            h.update(b)
    return h.hexdigest()


def _short(tok,rows,validator=None):
    from .chat350_sft_format import encode_messages
    found=[]
    for row in rows:
        if validator is not None and not validator(row):continue
        try: enc=encode_messages(tok,row["messages"],max_seq_len=128)
        except ValueError:continue
        found.append((row["id"],row["language"],enc))
    return found


def _gradients(model,tok,items,params):
    import torch
    from .train_chat350 import collate,assistant_loss
    from .arctus350_unattended_curriculum import gpu_temperature
    eos=int(tok.token_to_id("<|eos|>"))
    pad=int(tok.token_to_id("<|pad|>"))
    grad_length=sum(p.numel() for p in params)
    accum=torch.zeros(grad_length,dtype=torch.float32)
    observed_losses=[]
    for index,row in enumerate(items):
        if datetime.now(ZoneInfo("Europe/Berlin"))>=CUTOFF:
            raise RuntimeError("Owner's permitted GPU window ended")
        if index%2==0:
            temp=gpu_temperature()
            if temp is None or temp>=80 or torch.cuda.mem_get_info(0)[0]//2**20<2048:
                raise RuntimeError("GPU temperature or VRAM safety abort")
        x,y=collate([row[2]],pad,"cuda")
        loss=assistant_loss(model,x,y,eos,1.5)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite teacher-forced diagnostic loss")
        grads=torch.autograd.grad(loss,params,retain_graph=False,create_graph=False,allow_unused=False)
        vector=torch.cat([g.detach().to(device="cpu",dtype=torch.float32).flatten() for g in grads])
        if not torch.isfinite(vector).all():
            raise RuntimeError("nonfinite selected-gradient vector")
        accum.add_(vector)
        observed_losses.append(float(loss.detach()))
        del x,y,loss,grads,vector
    accum.div_(len(items))
    return accum,observed_losses


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent",required=True,type=Path)
    ap.add_argument("--tokenizer",required=True,type=Path)
    ap.add_argument("--clean",required=True,type=Path)
    ap.add_argument("--v1",required=True,type=Path)
    ap.add_argument("--v2",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--windows-free-mib",required=True,type=int)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError("refuse overwrite report")
    import torch
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .arctus350_exact_math_curriculum import independently_verify_row
    from .arctus350_balanced_arithmetic_v2 import parse_reference
    from .arctus350_unattended_curriculum import gpu_temperature
    from .train_chat350 import load_tokenizer
    from .secure_checkpoint350 import load_checkpoint
    from .model350 import FRONTIER_350,Neral350M
    temp=gpu_temperature()
    if temp is None:raise RuntimeError("no GPU thermal telemetry")
    if not torch.cuda.is_available() or "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("expected local RTX4070")
    free=torch.cuda.mem_get_info(0)[0]//2**20
    if (a.windows_free_mib<9216 or
        not permit_readonly_probe(free,temp,datetime.now(ZoneInfo("Europe/Berlin")).isoformat())):
        raise RuntimeError("read-only GPU preflight refused")
    torch.cuda.set_per_process_memory_fraction(.65,device=0)
    if _sha(a.parent)!=PARENT_SHA or _sha(a.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("pinned original native checkpoint/tokenizer SHA mismatch")
    m0,d0=validate_dataset_root(a.clean)
    m1,d1=validate_dataset_root(a.v1)
    m2,d2=validate_dataset_root(a.v2)
    if (m0["manifest_sha256"]!=CLEAN_SHA or m1["manifest_sha256"]!=V1_SHA
        or m2["manifest_sha256"]!=V2_SHA):
        raise ValueError("original/derived data source drift")
    for d in (d0,d1,d2):validate_prompt_holdouts(d)
    tok=load_tokenizer(a.tokenizer)
    human=lambda r:r["domain"]=="chat" and r["source"].startswith("oasst2-human-") and r["license"]=="Apache-2.0"
    source={
        "math":choose_bilingual_probe(_short(tok,d2["train"],lambda r:parse_reference(r) is not None),2),
        "format":choose_bilingual_probe(_short(tok,d1["train"],independently_verify_row),2),
        "human_chat":choose_bilingual_probe(_short(tok,d0["train"],human),2),
    }
    ck=load_checkpoint(a.parent)
    if (ck.get("config")!=FRONTIER_350.to_dict()
        or ck.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256
        or ck.get("source_checkpoint_sha256")!=ORIG_SHA
        or ck.get("experimental_non_promotable") is not True):
        raise ValueError("untrusted local model lineage")
    model=Neral350M()
    model.load_state_dict(ck["model"],strict=True)
    del ck
    if torch.cuda.mem_get_info(0)[0]//2**20<9216:
        raise RuntimeError("CUDA GPU claimed before model transfer")
    model=model.to("cuda",dtype=torch.float32)
    model.eval()
    model.set_gradient_checkpointing(False)
    for p in model.parameters():p.requires_grad_(False)
    chosen=[p for name,p in model.named_parameters() if name.startswith("blocks.23.ffn.")]
    if len(chosen)!=3 or sum(p.numel() for p in chosen)<9_000_000:
        raise RuntimeError("last-block FFN parameter selection changed")
    for p in chosen:p.requires_grad_(True)
    gradients={}
    means={}
    for domain,rows in source.items():
        vector,losses=_gradients(model,tok,rows,chosen)
        gradients[domain]=vector
        means[domain]=sum(losses)/len(losses)
    cosines={}
    norms={d:float((v.double()*v.double()).sum()) for d,v in gradients.items()}
    for a_name,b_name in (("math","format"),("math","human_chat"),("format","human_chat")):
        dot=float((gradients[a_name].double()*gradients[b_name].double()).sum())
        cosines[a_name+"_vs_"+b_name]=cosine_from_stats(
            dot,norms[a_name],norms[b_name])
    if _sha(a.parent)!=PARENT_SHA:
        raise RuntimeError("source checkpoint bytes changed during read-only diagnosis")
    report={
        "schema":"arctus350-last-ffn-task-gradient-compatibility-v1",
        "status":"completed_read_only",
        "source_model_sha256":PARENT_SHA,
        "source_data_sha256":{"clean":CLEAN_SHA,"math_v1":V1_SHA,"math_v2":V2_SHA},
        "last_layer_only":"blocks.23.ffn.*",
        "parameter_elements":sum(p.numel() for p in chosen),
        "cases_per_domain":{d:len(rows) for d,rows in source.items()},
        "language_balance_each_domain":{"de":2,"en":2},
        "mean_teacher_forced_loss":means,
        "average_gradient_l2_norm":{k:math.sqrt(v) for k,v in norms.items()},
        "mean_gradient_pair_cosines":cosines,
        "cosine_interpretation":{
            "negative":"task-averaged gradients oppose on measured last-block FFN",
            "zero":"near-orthogonal on selected FFN subset",
            "positive":"task-averaged gradients align on selected FFN subset",
        },
        "limitations":"12 selected train-only prompts and LAST FFN gradients only; not whole-model, not causal proof of forgetting, not unbiased heldout capability evaluation.",
        "model_parameters_modified":False,
        "optimizer_created":False,
        "checkpoint_saved":False,
        "raw_prompts_or_ids_exported":False,
        "model_weights_uploaded":False,
        "automatic_release":False,
        "gpu_temperature_start_c":temp,
        "peak_gpu_pytorch_reserved_mib":torch.cuda.max_memory_reserved(0)//2**20,
    }
    a.out.write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
