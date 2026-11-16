"""Known DEV24 smoke: compare frozen v2 and v1 PEFT, NOT the 23/24 pipeline score.

Never uses DEV24 for training, validation, adapter selection or prompt tuning.
Both use same cached Qwen3.5:9B base, greedy decoding and exact prompts.
"""
from __future__ import annotations
import argparse, gc, json, re, statistics, time
from pathlib import Path
from scripts.neral_1_1_postverify_dev24_cases import build, manifest
from scripts.neral_qlora_train_peft import BASE_MODEL, create_config


def score(text:str,gold:int)->tuple[bool,bool]:
    clean=text.strip()
    if re.fullmatch(r"[0-9]{1,13}",clean):
        return int(clean)==gold,False
    try:
        obj=json.loads(clean)
        if type(obj) is dict and set(obj) in ({"answer"},{"answer","tool","spec"}):
            ans=obj["answer"]
            if type(ans) is int: return ans==gold,False
    except (ValueError,TypeError): pass
    return False,True


def eval_adapter(label,adapter,checkpoint,tokenizer,cases):
    import torch
    from transformers import Qwen3_5ForCausalLM
    from peft import PeftModel
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base=Qwen3_5ForCausalLM.from_pretrained(
        str(checkpoint),local_files_only=True,device_map={"":0},
        quantization_config=create_config(BASE_MODEL),dtype=dtype,low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    model.eval()
    counts={c:{"correct":0,"total":0,"format_errors":0}
            for c in ("mathematics","logic","programming","mixed")}
    latencies=[]
    for idx,case in enumerate(cases):
        messages=[{"role":"system","content":
            "Answer the user's task with only the exact nonnegative integer. Do not guess; no explanations."},
            {"role":"user","content":case["question"]}]
        tokens=tokenizer.apply_chat_template(messages,tokenize=True,
                         add_generation_prompt=True,enable_thinking=False)
        if isinstance(tokens,dict):tokens=tokens["input_ids"]
        if type(tokens) is not list or len(tokens)>3500:
            raise RuntimeError("DEV24 prompt tokenized longer than 3500")
        inputs=torch.tensor([tokens],dtype=torch.long,device="cuda")
        torch.cuda.synchronize();now=time.monotonic()
        with torch.inference_mode():
            out=model.generate(input_ids=inputs,max_new_tokens=256,
              do_sample=False,pad_token_id=tokenizer.pad_token_id,
              eos_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        latencies.append(time.monotonic()-now)
        text=tokenizer.decode(out[0,len(tokens):],skip_special_tokens=True)
        correct,invalid=score(text,case["gold_integer"])
        cat=counts[case["category"]]
        cat["correct"]+=int(correct);cat["total"]+=1
        cat["format_errors"]+=int(invalid)
        print("DEV24_FORGETTING_PROGRESS",label,idx+1,"/24",flush=True)
        del inputs,out
    summary={"label":label,
      "correct":sum(x["correct"] for x in counts.values()),
      "total":24,"by_category":counts,
      "format_errors":sum(x["format_errors"] for x in counts.values()),
      "median_latency_s":round(statistics.median(latencies),3),
      "peak_reserved_mib":round(torch.cuda.max_memory_reserved()/1024**2,1)}
    del model,base; gc.collect();torch.cuda.empty_cache()
    print("DEV24_FORGETTING_SCALAR "+json.dumps(summary,sort_keys=True),flush=True)
    return summary


def main():
    ap=argparse.ArgumentParser()
    for name in ("checkpoint","v1","v2","out"):
        ap.add_argument("--"+name,required=True,type=Path)
    args=ap.parse_args()
    if args.out.exists():raise FileExistsError("Do not overwrite evaluation")
    if manifest()["cases"]!=24:raise RuntimeError("Original DEV24 not 24")
    cases=build()
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available():raise RuntimeError("Real GPU required")
    tokenizer=AutoTokenizer.from_pretrained(
        str(args.checkpoint),local_files_only=True,use_fast=True)
    v1=eval_adapter("QLoRA-v1",args.v1,args.checkpoint,tokenizer,cases)
    v2=eval_adapter("QLoRA-v2-pilot",args.v2,args.checkpoint,tokenizer,cases)
    summary={"status":"direct_adapter_dev24_not_the_neral_pipeline",
      "historical_neral_1_1_pipeline_baseline":23,
      "baseline_comparable":False,"no_training":True,
      "casepack_sha256":manifest()["sha256"],"v1":v1,"v2":v2,
      "net_v2_vs_v1":v2["correct"]-v1["correct"]}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(summary,sort_keys=True,indent=2)+"\n",encoding="utf8")


if __name__=="__main__":main()
