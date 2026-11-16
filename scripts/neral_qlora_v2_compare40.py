"""One-time, read-only v1 versus frozen v2 evaluation on 40 held-out cases.

No training, no optimizer, no data export of holdout prompts/responses.
Both adapters run in the same HF 4-bit model, native chat template, exact
system prompt, greedy decoding, seed and token budget; never compare a
different Ollama quantized v1 engine with HF v2 as if equivalent.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

from scripts.neral_qlora_v2_pilot import SYSTEM, TRAIN_SHA, VAL_SHA, _digest
from scripts.neral_qlora_train_peft import BASE_MODEL, create_config

TEST_SHA="c5e3962768a7a870df837f63624b2904cbcd6dfef559445f9c220e7ed6a7a7a7"
SEED=2026111602
CONTEXT=2048
MAX_NEW_TOKENS=512


def frozen_inputs(*, trained:Path, v1:Path, test:Path,train:Path,val:Path,output:Path)->list[dict]:
    if output.exists() or output.is_symlink():
        raise FileExistsError("Report is one-shot; no overwrite")
    manifest=json.loads((trained/"manifest.json").read_text(encoding="utf8"))
    if (manifest.get("status")!="frozen_adapter_ready_for_heldout_comparison"
            or manifest.get("max_steps")!=0
            or manifest.get("epochs") not in (2,3)
            or manifest.get("train_rows")!=184
            or manifest.get("val_rows")!=32
            or manifest.get("test_rows_opened")!=0
            or manifest.get("train_sha256")!=TRAIN_SHA
            or manifest.get("val_sha256")!=VAL_SHA):
        raise RuntimeError("Frozen full QLoRA-v2 training proof missing")
    if not (trained/"adapter/adapter_config.json").is_file() or not (
        v1/"adapter_config.json").is_file():
        raise FileNotFoundError("Both PEFT adapters must exist locally")
    if hashlib.sha256((trained/"adapter/adapter_model.safetensors").read_bytes()).hexdigest()!=manifest["frozen_adapter_sha256"]:
        raise RuntimeError("Frozen v2 adapter file changed")
    # This is the FIRST place where the 40-case test file is opened.
    if _digest(test)!=TEST_SHA:
        raise RuntimeError("Pinned 40-case test digest does not match")
    rows=[json.loads(s) for s in test.read_text(encoding="utf8").splitlines() if s.strip()]
    if len(rows)!=40 or len({x["id"] for x in rows})!=40:
        raise RuntimeError("Invalid 40-case holdout")
    def ids(path):
        return {json.loads(x)["id"] for x in path.read_text(encoding="utf8").splitlines() if x.strip()}
    if ({r["id"] for r in rows}&(ids(train)|ids(val))):
        raise RuntimeError("Training/validation/test ID leakage")
    return rows


def judge(row:dict,raw:str)->tuple[bool,bool]:
    """(correct, tool_format_error). Pure and independent of a model."""
    if not isinstance(raw,str) or not raw.strip() or len(raw.encode("utf8"))>16000:
        return False,True
    try:
        obj=json.loads(raw)
    except (ValueError,TypeError):
        return False,True
    if (type(obj) is not dict or set(obj)!={"tool","spec","answer"} or
        obj.get("tool") not in ("sympy","networkx","z3") or
        type(obj.get("spec")) is not dict):
        return False,True
    expected={"tool":row["task"]["tool"],"spec":row["task"]["spec"],
              "answer":row["answer"]}
    canon=lambda x:json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(",",":"))
    return canon(obj)==canon(expected),False


def run_model(label:str, adapter:Path,checkpoint:Path,rows:list[dict],tokenizer):
    import torch
    from transformers import Qwen3_5ForCausalLM
    from peft import PeftModel
    t0=time.monotonic()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base=Qwen3_5ForCausalLM.from_pretrained(
        str(checkpoint),local_files_only=True,
        quantization_config=create_config(BASE_MODEL),dtype=dtype,
        device_map={"":0},low_cpu_mem_usage=True)
    model=PeftModel.from_pretrained(base,str(adapter),is_trainable=False)
    model.eval()
    good=0
    errors=0
    latencies=[]
    tool_stats={t:{"correct":0,"total":0,"format_errors":0}
                for t in ("sympy","networkx","z3")}
    print("QLORA_V2_COMPARISON_MODEL_LOADED "+label,flush=True)
    for index,row in enumerate(rows):
        turns=[{"role":"system","content":SYSTEM},
               {"role":"user","content":row["prompt"]}]
        ids=tokenizer.apply_chat_template(
            turns,tokenize=True,add_generation_prompt=True,
            enable_thinking=False)
        if isinstance(ids,dict):ids=ids["input_ids"]
        if type(ids) is not list or len(ids)>CONTEXT-MAX_NEW_TOKENS:
            raise RuntimeError("Prompt exceeds fixed comparison token budget")
        inputs=torch.tensor([ids],dtype=torch.long,device="cuda")
        torch.cuda.synchronize()
        st=time.monotonic()
        with torch.inference_mode():
            output=model.generate(
                input_ids=inputs,max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,use_cache=True)
        torch.cuda.synchronize()
        sec=time.monotonic()-st
        raw=tokenizer.decode(output[0,len(ids):],skip_special_tokens=True).strip()
        correct,invalid=judge(row,raw)
        good+=int(correct);errors+=int(invalid);latencies.append(sec)
        family=row["task"]["tool"]
        tool_stats[family]["correct"]+=int(correct)
        tool_stats[family]["total"]+=1
        tool_stats[family]["format_errors"]+=int(invalid)
        # Never write a predicted response or a held-out prompt to a log.
        print("QLORA_V2_COMPARISON_PROGRESS "+label+" "+str(index+1)+"/40",flush=True)
        del inputs,output
    peak=torch.cuda.max_memory_reserved()/1024**2
    allocated=torch.cuda.max_memory_allocated()/1024**2
    total_s=time.monotonic()-t0
    result={"label":label,"correct":good,"total":len(rows),
            "accuracy":good/len(rows),
            "format_errors":errors,
            "median_latency_s":round(statistics.median(latencies),3),
            "peak_reserved_mib":round(peak,1),
            "peak_allocated_mib":round(allocated,1),
            "by_tool":tool_stats,
            "elapsed_s":round(total_s,1)}
    del model,base
    gc.collect()
    torch.cuda.empty_cache()
    print("QLORA_V2_COMPARISON_SCALAR "+json.dumps(result,sort_keys=True),flush=True)
    return result


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--trained",type=Path,required=True)
    p.add_argument("--v1",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True)
    p.add_argument("--test",type=Path,required=True)
    p.add_argument("--train",type=Path,required=True)
    p.add_argument("--val",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    args=p.parse_args()
    rows=frozen_inputs(trained=args.trained,v1=args.v1,
         test=args.test,train=args.train,val=args.val,output=args.out)
    import torch
    from transformers import AutoTokenizer
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("Actual RTX GPU required")
    torch.manual_seed(SEED)
    tokenizer=AutoTokenizer.from_pretrained(
        str(args.checkpoint),use_fast=True,local_files_only=True)
    before=run_model("QLoRA-v1",args.v1,args.checkpoint,rows,tokenizer)
    after=run_model("QLoRA-v2-pilot",args.trained/"adapter",
                    args.checkpoint,rows,tokenizer)
    report={"status":"real_40_case_comparison","test_sha256":TEST_SHA,
            "same_base_model":BASE_MODEL,"same_4bit_nf4":True,
            "same_prompt_and_decoding":True,"holdout_used_in_training":False,
            "max_new_tokens":MAX_NEW_TOKENS,"seed":SEED,
            "v1":before,"v2":after,
            "delta_correct":after["correct"]-before["correct"],
            "delta_median_latency_s":round(after["median_latency_s"]-before["median_latency_s"],3)}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print("QLORA_V2_COMPARISON_COMPLETE "+json.dumps({
        "v1_correct":before["correct"],"v2_correct":after["correct"],
        "v1_format_errors":before["format_errors"],
        "v2_format_errors":after["format_errors"],
        "v1_median_s":before["median_latency_s"],
        "v2_median_s":after["median_latency_s"]},sort_keys=True),flush=True)


if __name__=="__main__":main()
