"""One bounded local-only Arctus350 native FP32-master exact-output SFT pilot.

Parent: saved original-2B-based 64-step FP32 SFT. New augmentation supplies
checked precise outputs and small sums; 1/3 OASST2-human chat replay in SFT.
Not a model release. Rejects model saving unless *actual* unseen DEV exact
instruction success and both development losses meet predeclared gates.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

PARENT_SHA="57bbd1f70fcca6aee5e557b132d6b1b3f7b5e3378d0248923e5478bae36caf58"
ORIGINAL_2B_SHA="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
CLEAN_SHA="fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31"
AUGMENT_SHA="7e78a672786454c655b5dcf9e1cdc0967efee0d7381121623c3eebd8e93b7e1c"
STEPS=96
SEQ=128


def admit_targeted(authorized,win_free,cuda_free,cap,steps,seq):
    if (type(authorized) is not bool or type(win_free) is not int
        or type(cuda_free) is not int or min(win_free,cuda_free)<0
        or type(cap) not in (int,float) or not math.isfinite(cap)
        or type(steps) is not int or type(seq) is not int):
        raise ValueError("invalid GPU safety admission")
    return bool(authorized and win_free>=9216 and cuda_free>=9216
                and 0<cap<=.68 and 1<=steps<=96 and 16<=seq<=128)


def choose_schedule(exact,small_math,human,count):
    if type(count) is not int or not 1<=count<=32:
        raise ValueError("invalid sample count")
    cohorts=(exact,small_math,human)
    all_ids=[r[0] for group in cohorts for r in group]
    if len(all_ids)!=len(set(all_ids)):
        raise ValueError("duplicate training record identity")
    groups=[]
    for group in cohorts:
        if len(group)<count:
            raise ValueError("insufficient distinct examples per source cohort")
        sorted_group=sorted(group,key=lambda r:(hashlib.sha256(r[0].encode("utf8")).hexdigest(),r[0]))
        groups.append(sorted_group[:count])
    return [sample for triple in zip(*groups) for sample in triple]


def candidate_worth_saving(target_before,target_after,human_before,human_after,exact_successes):
    vals=(target_before,target_after,human_before,human_after)
    if any(type(v) not in (int,float) or not math.isfinite(v) or v<0 for v in vals):
        raise ValueError("invalid validation values")
    if type(exact_successes) is not int or exact_successes<0:
        raise ValueError("invalid exact success count")
    return (target_after<=target_before-.05
            and human_after<=human_before+.25 and exact_successes>=1)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda:source.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def _eligible(tok,rows,*,domain=None,source=None):
    from .chat350_sft_format import encode_messages
    result=[]
    for row in rows:
        if domain is not None and row["domain"]!=domain:continue
        if source is not None and row["source"]!=source:continue
        try: encoded=encode_messages(tok,row["messages"],max_seq_len=SEQ)
        except ValueError:continue
        result.append((row["id"],encoded))
    return result


def _dev_generation(model,tok):
    """Only existing public, non-sealed development questions."""
    import torch
    from .arctus350_clean_fp32_pilot import _free_gen
    from .chat350_eval_contract import case_prefix_ids,generation_forbidden_ids
    responses=_free_gen(model,tok)
    cases=json.loads((Path(__file__).parent/"chat350_dev_cases.json").read_text(encoding="utf8"))
    match=[x for x in cases if x.get("id")=="exact_ok_en"]
    if len(match)!=1:raise ValueError("missing exact English dev case")
    ids=case_prefix_ids(tok,match[0]["prefix"])
    invalid=generation_forbidden_ids(tok)
    eos=int(tok.token_to_id("<|eos|>"))
    tokens=[]
    model.eval()
    with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
        for _ in range(8):
            if torch.cuda.mem_get_info(0)[0]//2**20<2048:
                raise RuntimeError("GPU pressure in exact-dev inference")
            x=torch.tensor([ids],device="cuda",dtype=torch.long)
            logits=model(x)[0][0,-1].clone()
            logits[list(invalid)]=-float("inf")
            nxt=int(logits.argmax())
            if nxt==eos:break
            tokens.append(nxt)
            ids.append(nxt)
    rendered=tok.decode(tokens,skip_special_tokens=True).strip()
    responses["exact_ok_en"]={"text":rendered[:150],
                              "exact_correct":rendered=="OK",
                              "generated_tokens":len(tokens)}
    model.train()
    return responses


def _temperature():
    if not shutil.which("nvidia-smi"):return None
    try:
        t=subprocess.run(["nvidia-smi","--query-gpu=temperature.gpu","--format=csv,noheader,nounits"],
                         capture_output=True,text=True,timeout=5,check=False)
        result=t.stdout.strip().splitlines()
        return int(result[0]) if t.returncode==0 and len(result)==1 else None
    except (ValueError,OSError,subprocess.TimeoutExpired):
        return None


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parent",type=Path,required=True)
    ap.add_argument("--tokenizer",type=Path,required=True)
    ap.add_argument("--clean-data",type=Path,required=True)
    ap.add_argument("--augment-data",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--windows-free-mib",type=int,required=True)
    a=ap.parse_args()
    if a.out.exists():raise FileExistsError("refuse existing output directory")
    if os.getenv("NERAL_LOCAL_SFT_ALLOW")!="1":
        raise PermissionError("explicit local model training authorization missing")

    import torch
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .arctus350_exact_math_curriculum import independently_verify_row
    from .secure_checkpoint350 import load_checkpoint
    from .train_chat350 import load_tokenizer,eval_sft,assistant_loss,collate
    from .train_precision350 import checkpoint_model_state
    from .model350 import Neral350M,FRONTIER_350

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA BF16 not available")
    if "RTX 4070" not in torch.cuda.get_device_name(0):
        raise RuntimeError("expected local RTX4070")
    available=torch.cuda.mem_get_info(0)[0]//2**20
    if not admit_targeted(True,a.windows_free_mib,available,.65,STEPS,SEQ):
        raise RuntimeError("RTX4070 free VRAM preflight below 9GiB")
    torch.cuda.set_per_process_memory_fraction(.65,device=0)
    torch.manual_seed(350_1028)
    torch.set_float32_matmul_precision("high")
    if digest(a.parent)!=PARENT_SHA:raise ValueError("pinned original 64-step model SHA mismatch")
    if digest(a.tokenizer)!=EXPECTED_TOKENIZER_SHA256:raise ValueError("original 16K-BPE tokenizer mismatch")
    src_manifest,original=validate_dataset_root(a.clean_data)
    aug_manifest,augment=validate_dataset_root(a.augment_data)
    if src_manifest["manifest_sha256"]!=CLEAN_SHA or aug_manifest["manifest_sha256"]!=AUGMENT_SHA:
        raise ValueError("pinned cleaned/augmented dataset manifest mismatch")
    validate_prompt_holdouts({
        split:original[split]+augment[split] for split in ("train","val","test")
    })
    if not all(independently_verify_row(row) for split in augment.values() for row in split):
        raise ValueError("augmented mathematical or exact reference corrupted")
    tok=load_tokenizer(a.tokenizer)
    exact=[r for r in augment["train"] if r["domain"]=="format"]
    maths=[r for r in augment["train"] if r["domain"]=="math"]
    human=[r for r in original["train"] if r["domain"]=="chat"
           and r["source"].startswith("oasst2-human-") and r["license"]=="Apache-2.0"]
    training=choose_schedule(_eligible(tok,exact),_eligible(tok,maths),_eligible(tok,human),32)
    target_val=_eligible(tok,augment["val"])
    human_val=_eligible(tok,[r for r in original["val"] if r["domain"]=="chat"
                          and r["source"].startswith("oasst2-human-") and r["license"]=="Apache-2.0"])
    human_val=sorted(human_val,key=lambda r:hashlib.sha256(r[0].encode()).hexdigest())[:8]
    if len(target_val)<32 or len(human_val)<8:
        raise ValueError("insufficient disjoint validation examples")
    parent=load_checkpoint(a.parent)
    if (parent.get("config")!=FRONTIER_350.to_dict() or
        parent.get("tokenizer_sha256")!=EXPECTED_TOKENIZER_SHA256 or
        parent.get("source_checkpoint_sha256")!=ORIGINAL_2B_SHA or
        parent.get("experimental_non_promotable") is not True):
        raise ValueError("64-step experiment checkpoint lineage mismatch")
    model=Neral350M()
    model.load_state_dict(parent["model"],strict=True)
    del parent
    if torch.cuda.mem_get_info(0)[0]//2**20<9216:
        raise RuntimeError("GPU pressure before model transfer")
    model=model.to(device="cuda",dtype=torch.float32)
    model.set_gradient_checkpointing(True)
    pad=int(tok.token_to_id("<|pad|>"))
    eos=int(tok.token_to_id("<|eos|>"))
    target_before=eval_sft(model,[r[1] for r in target_val],1,pad,eos,"cuda")
    human_before=eval_sft(model,[r[1] for r in human_val],1,pad,eos,"cuda")
    dev_before=_dev_generation(model,tok)
    if torch.cuda.mem_get_info(0)[0]//2**20<4096:
        raise RuntimeError("shared GPU memory pressure before optimizer")
    opt=torch.optim.AdamW(model.parameters(),lr=1.0e-5,betas=(.9,.95),
                          weight_decay=.03,eps=1e-8,fused=True)
    losses=[]
    torch.cuda.reset_peak_memory_stats(0)
    temp_samples=[]
    for index,(_,encoded) in enumerate(training):
        if torch.cuda.mem_get_info(0)[0]//2**20<3072:
            raise RuntimeError("insufficient CUDA free VRAM at step "+str(index))
        if index%24==0:
            t=_temperature()
            if t is not None:
                temp_samples.append(t)
                if t>=80:raise RuntimeError("GPU temperature >=80C")
        x,y=collate([encoded],pad,"cuda")
        opt.zero_grad(set_to_none=True)
        loss=assistant_loss(model,x,y,eos,1.5)
        if not torch.isfinite(loss):raise RuntimeError("nonfinite loss")
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0,error_if_nonfinite=True)
        if not torch.isfinite(norm):raise RuntimeError("nonfinite gradient")
        opt.step()
        torch.cuda.synchronize()
        if torch.cuda.max_memory_reserved(0)>8192*2**20:
            raise RuntimeError("PyTorch CUDA >8GiB reserved")
        losses.append(float(loss.detach()))
        del x,y,loss,norm
    opt.zero_grad(set_to_none=True)
    del opt
    target_after=eval_sft(model,[r[1] for r in target_val],1,pad,eos,"cuda")
    human_after=eval_sft(model,[r[1] for r in human_val],1,pad,eos,"cuda")
    dev_after=_dev_generation(model,tok)
    exact_wins=sum(bool(dev_after[r].get("exact_correct"))
                   for r in ("exact_ok_de","exact_ok_en"))
    decision=candidate_worth_saving(target_before["answer_nll"],
                                    target_after["answer_nll"],
                                    human_before["answer_nll"],
                                    human_after["answer_nll"],exact_wins)
    if digest(a.parent)!=PARENT_SHA:
        raise ValueError("64-step parent checkpoint mutated")
    a.out.mkdir(parents=True,exist_ok=False)
    candidate_sha=None
    if decision:
        if shutil.disk_usage(a.out.parent).free<2*2**30:
            raise RuntimeError("insufficient local disk for candidate checkpoint")
        checkpoint=a.out/"experimental_only_nonpromotable.pt"
        torch.save({
            "model":checkpoint_model_state(model,"fp32-master",resume=False),
            "config":FRONTIER_350.to_dict(),
            "tokenizer_sha256":EXPECTED_TOKENIZER_SHA256,
            "source_checkpoint_sha256":ORIGINAL_2B_SHA,
            "parent_checkpoint_sha256":PARENT_SHA,
            "data_manifest_sha256":AUGMENT_SHA,
            "original_clean_data_manifest_sha256":CLEAN_SHA,
            "chat_step":96,
            "experimental_non_promotable":True,
            "frozen_suite_never_used":True,
        },checkpoint)
        candidate_sha=digest(checkpoint)
    report={
        "schema":"arctus350-native-targeted-fp32-96step-v1",
        "status":"completed",
        "steps":STEPS,"train_exact":32,"train_small_math":32,"train_human":32,
        "target_val_rows":len(target_val),"human_val_rows":len(human_val),
        "parent_sha256":PARENT_SHA,"original_2b_sha256":ORIGINAL_2B_SHA,
        "augment_manifest_sha256":AUGMENT_SHA,"original_manifest_sha256":CLEAN_SHA,
        "target_nll_before":target_before,"target_nll_after":target_after,
        "human_nll_before":human_before,"human_nll_after":human_after,
        "public_dev_before":dev_before,"public_dev_after":dev_after,
        "absolute_dev_exact_success_count":exact_wins,
        "training_loss_first_8":losses[:8],"training_loss_last_8":losses[-8:],
        "peak_torch_reserved_mib":torch.cuda.max_memory_reserved(0)//2**20,
        "gpu_temps_c":temp_samples,
        "candidate_saved_local_only":decision,"candidate_sha256":candidate_sha,
        "parent_weights_unchanged":True,
        "no_replay_LM":True,"no_frozen_release_test":True,
        "automatic_promotion":False,"release_allowed":False,
        "limitations":"96 tiny updates, no LM replay; candidate requires independent larger development/frozen gate. These metrics do not establish general intelligence.",
        "private_weights_uploaded":False,
    }
    (a.out/"aggregate_report.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
