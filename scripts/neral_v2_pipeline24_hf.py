"""Actual NERAL1.1 24-case full controller with fixed HF v1 and v2-full adapters.

Uses original run_case/NEARL1.1 Z3 and Python-Docker verifier, not a fake
adapter-only count. Both runs use the same NF4 Qwen3.5:9B HF backend, so v1-v2
is comparable *within this experiment*. The historical 23/24 Ollama backend
cannot be claimed identical to the HF transport.
"""
from __future__ import annotations
import argparse, gc, json, math, statistics, subprocess, time
from collections.abc import Mapping
from pathlib import Path

from scripts.neral_1_1_postverify_dev24_cases import build,manifest
from scripts.neral_1_1_postverify_dev24_gpu import offline_metrics,cpu_preflight
from scripts.neral_1_1_prospective_blind100 import run_case,BASE_TAG,TUNED_TAG
from scripts.neral_qlora_train_peft import BASE_MODEL,create_config
from scripts.neral_v2_final_eval40 import validate_frozen

def run_with_adapter(label,adapter_path,base_path,tokenizer,cases):
    import torch
    from transformers import Qwen3_5ForCausalLM
    from peft import PeftModel
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base=Qwen3_5ForCausalLM.from_pretrained(
        str(base_path),local_files_only=True,quantization_config=create_config(BASE_MODEL),
        dtype=dtype,device_map={"":0},low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(base,str(adapter_path),is_trainable=False)
    model.eval()
    calls=[0]
    def hf_transport(packet):
        if not isinstance(packet,dict) or packet.get("model") not in (BASE_TAG,TUNED_TAG):
            raise ValueError("Unknown pipeline model tag")
        roles=packet.get("messages")
        if not isinstance(roles,list) or not 1<=len(roles)<=14:
            raise ValueError("Invalid model messages")
        ids=tokenizer.apply_chat_template(roles,tokenize=True,
            add_generation_prompt=True,enable_thinking=False)
        if isinstance(ids,Mapping):ids=ids["input_ids"]
        if type(ids) is not list or len(ids)>4000:
            raise ValueError("Input exceeds bounded HF context")
        requested=packet.get("options",{}).get("num_predict",512)
        if type(requested) is not int or not 1<=requested<=2048:
            raise ValueError("Token budget outside approved range")
        maxnew=min(requested,4096-len(ids))
        if maxnew<16:
            raise RuntimeError("No model output budget")
        inp=torch.tensor([ids],dtype=torch.long,device="cuda")
        torch.cuda.synchronize()
        try:
            with torch.inference_mode():
                if packet["model"]==BASE_TAG:
                    with model.disable_adapter():
                        out=model.generate(input_ids=inp,do_sample=False,max_new_tokens=maxnew,
                           use_cache=True,pad_token_id=tokenizer.pad_token_id,
                           eos_token_id=tokenizer.eos_token_id)
                else:
                    out=model.generate(input_ids=inp,do_sample=False,max_new_tokens=maxnew,
                       use_cache=True,pad_token_id=tokenizer.pad_token_id,
                       eos_token_id=tokenizer.eos_token_id)
            torch.cuda.synchronize()
            produced=out[0,len(ids):].tolist()
            eos={tokenizer.eos_token_id}
            for tok in (tokenizer.convert_tokens_to_ids("<|im_end|>"),
                        tokenizer.convert_tokens_to_ids("<|endoftext|>")):
                if type(tok) is int and tok>=0:eos.add(tok)
            stopped=bool(produced and produced[-1] in eos)
            raw=tokenizer.decode(produced,skip_special_tokens=True)
            calls[0]+=1
            return {"done":True,"done_reason":"stop" if stopped else "length",
                    "model":packet["model"],"message":{"role":"assistant","content":raw},
                    "eval_count":len(produced)}
        finally:
            del inp
    scored=[]
    for i,case in enumerate(cases):
        outcome=run_case(case,hf_transport)
        outcome.pop("raw",None)
        scored.append(outcome)
        print("NERAL_V2_FULL_PIPELINE24_PROGRESS "+label+" "+str(i+1)+"/24",flush=True)
    metrics=offline_metrics(cases,scored)
    metrics.update({"engine":"HF_NF4_4bit_QLoRA","adapter":label,
                   "gpu_peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1),
                   "hf_model_calls":calls[0],"strict_case_correct_flags":"".join(
                     "1" if x["neral_strict_correct"] else "0" for x in scored),
                   "base_correct_flags":"".join(
                     "1" if x["base_strict_correct"] else "0" for x in scored)})
    del model,base
    gc.collect();torch.cuda.empty_cache()
    print("NERAL_V2_FULL_PIPELINE24_SCALAR "+json.dumps({
      "adapter":label,"neral_correct":metrics["neral_correct"],
      "base_correct":metrics["base_correct"],"regressions":metrics["regressions"],
      "z3_verified":metrics["z3_verified"],"python_verified":metrics["python_independently_verified"],
      "median_s":metrics["neral_median_s"],"gpu_peak_reserved_mib":metrics["gpu_peak_reserved_mib"],
      "hf_model_calls":metrics["hf_model_calls"]},sort_keys=True),flush=True)
    return metrics


def main():
    ap=argparse.ArgumentParser()
    for opt in ("base","v1","full","out"):ap.add_argument("--"+opt,required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("One-time development report exists")
    validate_frozen(args.full,args.v1,args.out)
    # Original original casepack hash and actual Docker/source validators.
    cpu_preflight()
    cases=build()
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Live RTX4070 required")
    # Separate the historical Ollama baseline 23/24 from the comparable HF model pair.
    tokenizer=AutoTokenizer.from_pretrained(str(args.base),local_files_only=True,use_fast=True)
    before=run_with_adapter("QLoRA-v1-HF",args.v1,args.base,tokenizer,cases)
    after=run_with_adapter("QLoRA-v2-full-HF",args.full/"adapter",args.base,tokenizer,cases)
    a=before["strict_case_correct_flags"];b=after["strict_case_correct_flags"]
    if len(a)!=24 or len(b)!=24:raise RuntimeError("Incomplete 24-case score bits")
    report={"status":"24_original_cases_through_real_neral_verifier_HF_engine",
        "historical_neral_1_1_ollama_score":23,
        "historical_backend_directly_comparable":False,
        "same_hf_base_for_v1_v2":True,
        "casepack_sha256":manifest()["sha256"],"no_training":True,
        "v1_pipeline":before,"v2full_pipeline":after,
        "v2full_vs_v1_pipeline":{"gained":sum(x=="0" and y=="1" for x,y in zip(a,b)),
                                "regressed":sum(x=="1" and y=="0" for x,y in zip(a,b)),
                                "net":after["neral_correct"]-before["neral_correct"]}}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("NERAL_V2_FULL_PIPELINE24_FINISHED "+json.dumps({
        "v1":before["neral_correct"],"full":after["neral_correct"],
        **report["v2full_vs_v1_pipeline"]},sort_keys=True),flush=True)

if __name__=="__main__":main()
