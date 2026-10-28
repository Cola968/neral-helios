"""Real native Arctus350 original-2B FP32-master 240-step curriculum experiment.

Strictly local, reversible, non-promotable, one experiment. No foreign
pretrained weights. Training inputs are only metadata-labelled NERAL-generated
deterministic train rows, disjoint from VAL/TEST by all-user-turn gate.
Original 2B checkpoint and sanitized data are never opened for write.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
DATA_SHA="fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31"
LICENSE="Project-generated deterministic data; no external corpus"
QUOTAS={"math":60,"code":48,"format":48,"chat":48,"reasoning":36}
VAL_QUOTAS={"math":4,"code":4,"format":4,"chat":4,"reasoning":4}
STEPS=240
SEQ=192


def admit_curriculum(authorized,windows_free,cuda_free,cap,steps,seq):
    if (type(authorized) is not bool or type(windows_free) is not int
        or type(cuda_free) is not int or windows_free<0 or cuda_free<0
        or type(cap) not in (int,float) or not math.isfinite(cap)
        or type(steps) is not int or type(seq) is not int):
        raise ValueError("invalid admission arguments")
    return bool(authorized and windows_free>=9216 and cuda_free>=9216
                and 0<cap<=.75 and 1<=steps<=240 and 16<=seq<=256)


def select_curriculum(rows,quotas):
    """Select *unique* supported source examples; balance task exposure."""
    if (not isinstance(quotas,dict) or not quotas
        or not set(quotas).issubset({"math","code","format","chat","reasoning","explanation"})
        or any(type(q) is not int or not 1<=q<=240 for q in quotas.values())):
        raise ValueError("invalid curriculum domain quotas")
    if len({x[0] for x in rows})!=len(rows):
        raise ValueError("duplicate candidate row ID")
    groups={}
    for domain,n in quotas.items():
        accepted=[
            r for r in rows if r[2]==domain and r[4]==LICENSE and
            r[3].startswith("neral-generated-") and r[1] in ("de","en")
        ]
        ordered=sorted(accepted,key=lambda r:(hashlib.sha256(r[0].encode("utf8")).hexdigest(),r[0]))
        if len(ordered)<n:
            raise ValueError("insufficient verified eligible metadata rows for "+domain)
        groups[domain]=ordered[:n]
    output=[]
    # Interleave domains to prevent grouped contiguous training segments.
    while any(groups.values()):
        for domain in quotas:
            if groups[domain]:
                output.append(groups[domain].pop(0))
    if len(output)!=sum(quotas.values()) or len({x[0] for x in output})!=len(output):
        raise ValueError("invalid curriculum selection")
    return output


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _eligible(tok,rows):
    from .chat350_sft_format import encode_messages
    out=[]
    for row in rows:
        if row["domain"] not in QUOTAS or row["license"]!=LICENSE:
            continue
        if not row["source"].startswith("neral-generated-"):
            continue
        try:
            e=encode_messages(tok,row["messages"],max_seq_len=SEQ)
        except ValueError:
            continue
        out.append((row["id"],row["language"],row["domain"],row["source"],row["license"],e))
    return out


def _dev_gen(model,tok):
    # This is the previously established, non-sealed four-case suite.
    from .arctus350_clean_fp32_pilot import _free_gen
    return _free_gen(model,tok)


def _temperature_if_available():
    # Windows NVIDIA telemetry is mandatory before the run. WSL repeated
    # thermal checks are optional, but a valid high temperature aborts.
    if not shutil.which("nvidia-smi"):
        return None
    try:
        proc=subprocess.run(
            ["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],
            capture_output=True,text=True,timeout=6,check=False,
        )
        if proc.returncode!=0:
            return None
        out=proc.stdout.strip().splitlines()
        return int(out[0]) if len(out)==1 else None
    except (ValueError,OSError,subprocess.TimeoutExpired):
        return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",type=Path,required=True)
    ap.add_argument("--tokenizer",type=Path,required=True)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--windows-free-mib",type=int,required=True)
    a=ap.parse_args()
    if a.out.exists():
        raise FileExistsError("refuse to reuse an experiment directory")
    if os.getenv("NERAL_LOCAL_SFT_ALLOW")!="1":
        raise PermissionError("fresh local-GPU training authorization required")

    import torch
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256,validate_checkpoint_metadata
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .train_chat350 import load_tokenizer,eval_sft,assistant_loss,collate
    from .train_precision350 import checkpoint_model_state
    from .secure_checkpoint350 import load_checkpoint
    from .model350 import Neral350M,FRONTIER_350

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("native RTX4070 BF16 CUDA required")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("not the pinned RTX4070 GPU")
    free=torch.cuda.mem_get_info(0)[0]//2**20
    if not admit_curriculum(True,a.windows_free_mib,free,.72,STEPS,SEQ):
        raise RuntimeError("shared GPU preflight failed: not enough free VRAM")
    torch.cuda.set_per_process_memory_fraction(.72,device=0)
    torch.manual_seed(350_1029)
    random.seed(350_1029)
    torch.set_float32_matmul_precision("high")
    if sha256(a.checkpoint)!=SOURCE_SHA:
        raise ValueError("original 2B foundation SHA256 mismatch")
    if sha256(a.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("locked 16K BPE mismatch")
    manifest,splits=validate_dataset_root(a.data)
    if manifest["manifest_sha256"]!=DATA_SHA:
        raise ValueError("sanitized data manifest SHA256 mismatch")
    validate_prompt_holdouts(splits)
    tok=load_tokenizer(a.tokenizer)
    train=select_curriculum(_eligible(tok,splits["train"]),QUOTAS)
    val=select_curriculum(_eligible(tok,splits["val"]),VAL_QUOTAS)
    if {x[0] for x in train}&{x[0] for x in val}:
        raise ValueError("train / dev row identity collision")
    if len(train)!=STEPS:
        raise AssertionError("exactly 240 distinct SFT train rows required")
    ck=load_checkpoint(a.checkpoint)
    validate_checkpoint_metadata(ck,min_processed_tokens=2_000_000_000)
    if ck.get("config")!=FRONTIER_350.to_dict():
        raise ValueError("model architecture mismatch")
    model=Neral350M()
    model.load_state_dict(ck["model"],strict=True)
    del ck
    if torch.cuda.mem_get_info(0)[0]//2**20<9216:
        raise RuntimeError("GPU pressure before model transfer")
    model=model.to(device="cuda",dtype=torch.float32)
    model.set_gradient_checkpointing(True)
    eos=int(tok.token_to_id("<|eos|>"))
    pad=int(tok.token_to_id("<|pad|>"))
    before=eval_sft(model,[row[5] for row in val],1,pad,eos,"cuda")
    before_dev=_dev_gen(model,tok)
    if torch.cuda.mem_get_info(0)[0]//2**20<4096:
        raise RuntimeError("insufficient VRAM before optimizer setup")
    opt=torch.optim.AdamW(model.parameters(),lr=1.5e-5,betas=(.9,.95),
                          eps=1e-8,weight_decay=.03,fused=True)
    losses=[]
    torch.cuda.reset_peak_memory_stats(0)
    interim=[]
    temp_samples=[]
    for i,row in enumerate(train):
        if torch.cuda.mem_get_info(0)[0]//2**20<3072:
            raise RuntimeError("RTX4070 shared VRAM pressure: stop at step "+str(i))
        if i%24==0:
            temperature=_temperature_if_available()
            if temperature is not None:
                temp_samples.append(temperature)
                if temperature>=80:
                    raise RuntimeError("GPU temperature exceeds safety threshold")
        x,y=collate([row[5]],pad,"cuda")
        opt.zero_grad(set_to_none=True)
        loss=assistant_loss(model,x,y,eos,1.5)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite SFT loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        if not torch.isfinite(norm):
            raise RuntimeError("nonfinite FP32 gradient norm")
        opt.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        if torch.cuda.max_memory_reserved(0)>8192*2**20:
            raise RuntimeError("PyTorch CUDA reserved more than 8GiB")
        del x,y,loss,norm
        if (i+1)%80==0:
            val_report=eval_sft(model,[v[5] for v in val],1,pad,eos,"cuda")
            interim.append({"after_steps":i+1,**val_report})
    opt.zero_grad(set_to_none=True)
    del opt
    after=interim[-1]
    after_dev=_dev_gen(model,tok)
    after_gain=after["answer_nll"]-before["answer_nll"]
    if sha256(a.checkpoint)!=SOURCE_SHA:
        raise ValueError("original 2B foundation mutated")
    if shutil.disk_usage(a.out.parent).free<2*2**30:
        raise RuntimeError("less than 2GiB local free disk")
    a.out.mkdir(parents=True,exist_ok=False)
    candidate_saved=False
    cand_sha=None
    if after_gain<-.05:
        candidate=a.out/"experimental_only_nonpromotable.pt"
        torch.save({
            "model":checkpoint_model_state(model,"fp32-master",resume=False),
            "config":FRONTIER_350.to_dict(),
            "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
            "source_checkpoint_sha256":SOURCE_SHA,
            "source_processed_tokens":2_073_161_216,
            "data_manifest_sha256":DATA_SHA,
            "chat_step":STEPS,
            "experimental_non_promotable":True,
            "no_replay_or_sealed_score":True,
            "dev_validation":after,
        },candidate)
        cand_sha=sha256(candidate)
        candidate_saved=True
    report={
        "schema":"arctus350-real-2b-fp32-curriculum240-v1",
        "status":"completed","optimizer_steps":STEPS,
        "original_2b_checkpoint_sha256":SOURCE_SHA,"source_unchanged":True,
        "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
        "clean_data_manifest_sha256":DATA_SHA,
        "training_rows":len(train),"validation_rows":len(val),
        "train_domains":dict(sorted(collections.Counter(r[2] for r in train).items())),
        "train_languages":dict(sorted(collections.Counter(r[1] for r in train).items())),
        "train_sources":dict(sorted(collections.Counter(r[3] for r in train).items())),
        "validation_domains":dict(sorted(collections.Counter(r[2] for r in val).items())),
        "sequence_max":SEQ,"batch":1,"precision":"fp32-master/bf16-autocast",
        "val_before":before,"val_checkpoints":interim,
        "val_nll_change":after_gain,
        "freegen_before":before_dev,"freegen_after":after_dev,
        "training_loss_first_10":losses[:10],
        "training_loss_last_10":losses[-10:],
        "peak_pytorch_reserved_mib":torch.cuda.max_memory_reserved(0)//2**20,
        "peak_pytorch_allocated_mib":torch.cuda.max_memory_allocated(0)//2**20,
        "gpu_temperature_samples_c":temp_samples,
        "cuda_free_mib_after":torch.cuda.mem_get_info(0)[0]//2**20,
        "candidate_saved_local_only":candidate_saved,"candidate_sha256":cand_sha,
        "replay_used":False,"frozen_benchmark_used":False,
        "automatic_promotion":False,"release_allowed":False,
        "limitations":"240 selected rows, incomplete provenance/license audit; no LM replay. "
                      "Lower NLL or less repetition is not general instruction competence.",
        "original_or_private_datasets_uploaded":False,
    }
    a.out.joinpath("aggregate_report.json").write_text(
        json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    sys.exit(main())
