"""Strictly bounded REAL 2B Arctus350 FP32-master SFT proof on cleaned local data.

Only original Arctus350 weights, 64 steps at <=128 tokens, 1 dialogue/step,
32 DE + 32 EN eligible training dialogues and disjoint dev validation.
No replay; cannot be promoted. Candidate optionally saved locally only with
measured dev NLL gain, never to GitHub or HF. No sealed benchmark scoring.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
from pathlib import Path


SOURCE_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
DATA_SHA="fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31"
DEV_IDS=("de_chat_01","en_chat_01","exact_ok_de","math_01")


def admit_pilot(authorized,windows_free,cuda_free,cap,steps,max_seq):
    if (type(authorized) is not bool or type(windows_free) is not int
        or type(cuda_free) is not int or windows_free<0 or cuda_free<0
        or type(cap) not in (float,int) or not math.isfinite(cap)
        or type(steps) is not int or type(max_seq) is not int):
        raise ValueError("invalid authorization or GPU bounds")
    return bool(authorized and windows_free>=9216 and cuda_free>=9216
                and 0<cap<=.70 and 1<=steps<=64 and 16<=max_seq<=128)


def choose_balanced(eligible,per_lang):
    if type(per_lang) is not int or per_lang<=0 or per_lang>64:
        raise ValueError("invalid per-language sample count")
    ids=[x[0] for x in eligible]
    if len(ids)!=len(set(ids)):raise ValueError("duplicate eligible row IDs")
    out=[]
    for lang in ("de","en"):
        group=sorted((x for x in eligible if x[1]==lang),
                     key=lambda x:(hashlib.sha256(x[0].encode("utf8")).hexdigest(),x[0]))
        if len(group)<per_lang:
            raise ValueError("not enough eligible short bilingual rows: "+lang)
        out.extend(group[:per_lang])
    # Alternate languages rather than training all German then all English.
    a,b=out[:per_lang],out[per_lang:]
    return [item for pair in zip(a,b) for item in pair]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def _select(tok,rows,max_seq,per_lang):
    from .chat350_sft_format import encode_messages
    eligible=[]
    for row in rows:
        if row["language"] not in ("de","en"):continue
        try:
            encoded=encode_messages(tok,row["messages"],max_seq_len=max_seq)
        except ValueError:
            continue
        eligible.append((row["id"],row["language"],encoded))
    return choose_balanced(eligible,per_lang)


def _free_gen(m,tok):
    import torch
    from .chat350_eval_contract import case_prefix_ids,generation_forbidden_ids
    from .model350 import FRONTIER_350
    cases=json.loads((Path(__file__).parent/"chat350_dev_cases.json").read_text(encoding="utf8"))
    by_id={item["id"]:item for item in cases}
    if len(by_id)!=len(cases) or not set(DEV_IDS).issubset(by_id):
        raise ValueError("missing or duplicate development case IDs")
    eos=int(tok.token_to_id("<|eos|>"))
    forbidden=generation_forbidden_ids(tok)
    out={}
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        m.eval()
        for rid in DEV_IDS:
            item=by_id[rid]
            ids=case_prefix_ids(tok,item["prefix"])
            generated=[]
            for _ in range(min(24,item["max_new_tokens"])):
                if torch.cuda.mem_get_info(0)[0]//2**20<2048:
                    raise RuntimeError("GPU memory pressure during free generation")
                x=torch.tensor([ids],device="cuda",dtype=torch.long)
                logits=m(x)[0][0,-1].clone()
                logits[list(forbidden)]=-float("inf")
                token=int(logits.argmax())
                generated.append(token)
                if token==eos:break
                ids.append(token)
            content=generated[:-1] if generated and generated[-1]==eos else generated
            text=tok.decode(content,skip_special_tokens=True).strip()
            out[rid]={"text":text[:180],"eos_generated":bool(generated and generated[-1]==eos),
                      "tokens":len(content),
                      "exact_correct":text=="OK" if rid=="exact_ok_de" else None}
    m.train()
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint",required=True,type=Path)
    ap.add_argument("--tokenizer",required=True,type=Path)
    ap.add_argument("--data",required=True,type=Path)
    ap.add_argument("--out",required=True,type=Path)
    ap.add_argument("--windows-free-mib",required=True,type=int)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("never overwrite model experiment results")
    if os.getenv("NERAL_LOCAL_SFT_ALLOW")!="1":
        raise PermissionError("fresh owner authorization missing")
    import torch
    from .chat350_contract import validate_checkpoint_metadata,EXPECTED_TOKENIZER_SHA256
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .train_chat350 import load_tokenizer,eval_sft,assistant_loss,collate
    from .train_precision350 import checkpoint_model_state
    from .secure_checkpoint350 import load_checkpoint
    from .model350 import Neral350M,FRONTIER_350
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 required")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("wrong GPU")
    free= torch.cuda.mem_get_info(0)[0]//2**20
    if not admit_pilot(True,args.windows_free_mib,free,.68,64,128):
        raise RuntimeError("insufficient GPU free VRAM")
    torch.cuda.set_per_process_memory_fraction(.68,device=0)
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(350_1028)
    random.seed(350_1028)
    if digest(args.checkpoint)!=SOURCE_SHA:
        raise ValueError("original 2B checkpoint SHA mismatch")
    if digest(args.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("locked tokenizer SHA mismatch")
    manifest,splits=validate_dataset_root(args.data)
    if manifest["manifest_sha256"]!=DATA_SHA:
        raise ValueError("derived sanitized dataset manifest mismatch")
    validate_prompt_holdouts(splits)
    tok=load_tokenizer(args.tokenizer)
    train=_select(tok,splits["train"],128,32)
    val=_select(tok,splits["val"],128,4)
    train_ids=[r[0] for r in train]
    val_ids=[r[0] for r in val]
    if set(train_ids)&set(val_ids):raise ValueError("split overlap")
    ck=load_checkpoint(args.checkpoint)
    validate_checkpoint_metadata(ck,min_processed_tokens=2_000_000_000)
    if ck.get("config")!=FRONTIER_350.to_dict():
        raise ValueError("architecture mismatch")
    m=Neral350M()
    m.load_state_dict(ck["model"],strict=True)
    del ck
    m=m.to(device="cuda",dtype=torch.float32)
    m.set_gradient_checkpointing(True)
    eos=int(tok.token_to_id("<|eos|>"))
    pad=int(tok.token_to_id("<|pad|>"))
    before=eval_sft(m,[r[2] for r in val],1,pad,eos,"cuda")
    pre_gen=_free_gen(m,tok)
    if torch.cuda.mem_get_info(0)[0]//2**20<4096:
        raise RuntimeError("less than 4GiB free before optimizer")
    opt=torch.optim.AdamW(m.parameters(),lr=1.5e-5,betas=(.9,.95),
                          eps=1e-8,weight_decay=.03,fused=True)
    losses=[]
    torch.cuda.reset_peak_memory_stats(0)
    for i in range(64):
        if torch.cuda.mem_get_info(0)[0]//2**20<2048:
            raise RuntimeError("GPU free memory below 2GiB at step "+str(i))
        x,y=collate([train[i][2]],pad,"cuda")
        opt.zero_grad(set_to_none=True)
        loss=assistant_loss(m,x,y,eos,1.5)
        if not torch.isfinite(loss):raise RuntimeError("nonfinite SFT loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(m.parameters(),1.0,error_if_nonfinite=True)
        if not torch.isfinite(norm):raise RuntimeError("nonfinite gradient norm")
        opt.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach()))
        if torch.cuda.max_memory_reserved(0)>8192*2**20:
            raise RuntimeError("PyTorch CUDA reserved >8GiB after step")
        del x,y,loss
    opt.zero_grad(set_to_none=True)
    del opt
    after=eval_sft(m,[r[2] for r in val],1,pad,eos,"cuda")
    post_gen=_free_gen(m,tok)
    changed=after["answer_nll"]-before["answer_nll"]
    source_after=digest(args.checkpoint)
    if source_after!=SOURCE_SHA:
        raise ValueError("source checkpoint changed")
    if shutil.disk_usage(args.out.parent).free<2*2**30:
        raise RuntimeError("need 2GiB free disk for local experimental candidate")
    args.out.mkdir(parents=True,exist_ok=False)
    candidate_saved=False
    # DEV-NLL is not a substitute for independent capability regression.
    if changed<-.002:
        ckpt={
            "model":checkpoint_model_state(m,"fp32-master",resume=False),
            "config":FRONTIER_350.to_dict(),
            "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
            "source_checkpoint_sha256":SOURCE_SHA,
            "source_processed_tokens":2_073_161_216,
            "data_manifest_sha256":DATA_SHA,
            "chat_step":64,
            "experimental_non_promotable":True,
            "no_replay_or_frozen_score":True,
            "dev_validation":after,
        }
        candidate=args.out/"experimental_nonpromotable_64step.pt"
        torch.save(ckpt,candidate)
        candidate_saved=True
        candidate_sha=digest(candidate)
    else:
        candidate_sha=None
    report={
        "schema":"arctus350-clean-2b-fp32-64step-sft-v1",
        "status":"completed",
        "source_sha256":SOURCE_SHA,"source_untouched":True,
        "data_manifest_sha256":DATA_SHA,
        "steps_completed":64,"sequence_cap":128,"batch_size":1,
        "train_languages":{"de":32,"en":32},"val_languages":{"de":4,"en":4},
        "train_ids_sha256":hashlib.sha256(json.dumps(train_ids).encode()).hexdigest(),
        "val_ids_sha256":hashlib.sha256(json.dumps(val_ids).encode()).hexdigest(),
        "baseline_dev":before,"final_dev":after,"dev_nll_change":changed,
        "before_dev_free_generation":pre_gen,"after_dev_free_generation":post_gen,
        "loss_first_8":losses[:8],"loss_last_8":losses[-8:],
        "peak_torch_reserved_mib":torch.cuda.max_memory_reserved(0)//2**20,
        "peak_torch_allocated_mib":torch.cuda.max_memory_allocated(0)//2**20,
        "cuda_free_mib_end":torch.cuda.mem_get_info(0)[0]//2**20,
        "candidate_saved_local_only":candidate_saved,
        "candidate_sha256":candidate_sha,
        "automatic_promotion":False,"release_allowed":False,
        "limitations":"64 supervised updates; no LM replay, no sealed capability suite. "
                      "Unreviewed semantic template overlap may remain. No claim of better intelligence.",
        "private_weights_uploaded":False,
    }
    args.out.joinpath("aggregate_report.json").write_text(
        json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    sys.exit(main())
