"""V4.3 owner-only raw Qwen3.5-9B vs frozen NERAL1.0 matched reference50.

NEVER writes prompts, keys or answers to CI logs. Two sequential, mutually
exclusive GPU phases. Separate owner orchestration must ensure raw runs only
with port 9690 stopped; the frozen NERAL phase requires HTTP200.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import statistics
import subprocess
import time
import gc
from typing import Any

from scripts.neral_v41_blind200_live import OwnerClient, grade_response, IMAGE
from scripts.neral_v42_resumable import wait_until_available, sample_gpu_mb
from scripts.neral_v43_reference50 import (
    FROZEN, ENGINES, generate_cases, seal_cases, verify_seal, aggregate,
)

CHECKPOINT="D:/neral-qlora/checkpoint"


def validate_engine_identity(engine, *, checkpoint, adapter_loaded):
    if checkpoint.replace("\\","/").rstrip("/").lower()!=CHECKPOINT.lower():
        raise ValueError("Raw and tuned reference must share the pinned 9B HF base")
    if engine not in ENGINES or type(adapter_loaded) is not bool:
        raise ValueError("Unexpected reference mode")
    if (engine=="raw_qwen35_9b" and adapter_loaded
            or engine=="neral10_obsidian" and not adapter_loaded):
        raise ValueError("Adapter identity invalid")
    return engine


@contextmanager
def store_private(path:Path):
    """Exclusive creation only; no resume of interrupted or duplicate results."""
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x",encoding="utf-8") as sink:
        yield sink


def verify_docker_grader(*,runner=subprocess.run):
    """Fail BEFORE loading any model if offline private code grader unavailable."""
    try:
        check=runner(["docker","image","inspect",IMAGE],
                     stdout=subprocess.PIPE,stderr=subprocess.PIPE,
                     text=True,timeout=12,check=False)
    except (OSError,subprocess.TimeoutExpired) as exc:
        raise RuntimeError("Offline grading image not available") from exc
    if check.returncode != 0:
        raise RuntimeError("Preinstalled offline coding grader image absent")


def make_record(case,engine,response,*,grade,reason,elapsed,
                total_peak_mib,cuda_reserved_mib=None):
    """The private reply is intentionally excluded from the safe record."""
    return {
        "id":case["id"],"category":case["category"],"engine":engine,
        "prompt_sha256":sha256(case["prompt"].encode("utf-8")).hexdigest(),
        "grade":grade,"reason":reason,
        "latency_ms":round(elapsed*1000),
        "vram_total_peak_mib":total_peak_mib,
        "cuda_reserved_peak_mib":cuda_reserved_mib,
        "benchmark_isolated":True,
    }


def _root(outdir):
    if outdir.exists() and outdir.is_symlink():
        raise ValueError("Private baseline path cannot be a symlink")
    if not outdir.is_dir():
        raise FileNotFoundError("Owner-local sealed baseline directory is required")
    sealed=outdir/"sealed50.jsonl"; manifest=outdir/"manifest.json"
    return verify_seal(sealed,manifest),json.loads(manifest.read_text("utf-8"))


def prepare(outdir):
    if outdir.exists():
        raise FileExistsError("Run-id baseline directory already exists")
    outdir.mkdir(parents=True,exist_ok=False)
    try:
        digest=seal_cases(generate_cases(secrets.token_hex(32)),
                          outdir/"sealed50.jsonl",outdir/"manifest.json")
    except BaseException:
        raise
    print("NERAL_V43_REF50_SEALED_SHA256="+digest+" cases=50",flush=True)


def _emit(rows,path):
    with store_private(path) as sink:
        for record in rows:
            sink.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n")
            sink.flush()


def _score(case,response):
    if case["grader"]=="blind_pairwise":
        return "ungraded","needs_independent_blinded_human_review"
    return grade_response(case,response)


def neral(outdir):
    cases,_=_root(outdir)
    validate_engine_identity("neral10_obsidian",checkpoint=CHECKPOINT,adapter_loaded=True)
    verify_docker_grader()
    output=outdir/"neral10-50.jsonl"
    if output.exists():raise FileExistsError("NERAL 1.0 baseline already attempted")
    client=OwnerClient()
    client.connect()
    rows=[]
    with store_private(output) as sink:
        for i,case in enumerate(cases,1):
            if not wait_until_available(client,max_wait_seconds=90):
                raise RuntimeError("NERAL GPU owner busy; stop without counting task")
            start=time.monotonic()
            code,result,elapsed=client.post("Obsidian",case["prompt"],[],diagnostic_telemetry=True)
            if code==409: raise RuntimeError("HTTP409; no unsanctioned retry")
            if code==200 and result.get("benchmark_isolated") is not True:
                raise RuntimeError("No training/memory isolation contract")
            if code==200 and result.get("adapter_loaded") is not True:
                # May be purely deterministic source/tool response; not a
                # model win; annotate response as tool-only in private result.
                if result.get("model_generated") is True:
                    raise RuntimeError("Unexpected unadapted 9B model")
            if code != 200:
                grade,reason="fail","http_failure"
            else:
                grade,reason=_score(case,result)
            diagnostic=result.get("telemetry") if type(result) is dict else {}
            status=make_record(case,"neral10_obsidian",result,grade=grade,reason=reason,
                               elapsed=elapsed,total_peak_mib=sample_gpu_mb(),
                               cuda_reserved_mib=(diagnostic or {}).get("peak_cuda_reserved_mib"))
            status["http_status"]=code
            status["adapter_loaded"]=result.get("adapter_loaded")
            status["model_generated"]=result.get("model_generated")
            status["source_commit"]=FROZEN
            sink.write(json.dumps(status,sort_keys=True)+"\n")
            sink.flush()
            rows.append(status)
            print("NERAL_V43_REF50_NERAL_PROGRESS "+str(i)+"/50",flush=True)
    print("NERAL_V43_REF50_NERAL_DONE n="+str(len(rows)),flush=True)


def _raw_model():
    # No cloud/model downloads. Quantization and generation follow frozen
    # NERAL FrozenQwen35HF NF4 settings, but NEVER load the PEFT adapter.
    import torch
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM
    os.environ.update({"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1",
                       "HF_HUB_DISABLE_TELEMETRY":"1","HF_DATASETS_OFFLINE":"1",
                       "TOKENIZERS_PARALLELISM":"false","WANDB_DISABLED":"true"})
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:
        raise RuntimeError("One owned local CUDA GPU required")
    free,total=torch.cuda.mem_get_info(0)
    if free/(1024**2)<8500:
        raise RuntimeError("Raw 9B preload free memory <8500 MiB")
    torch.cuda.set_per_process_memory_fraction(min(1.0,10752*1024**2/total),device=0)
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base=Path(CHECKPOINT)
    if not base.is_dir() or base.is_symlink() or not (base/"config.json").is_file():
        raise RuntimeError("Frozen 9B source snapshot unavailable")
    q=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type="nf4",
                         bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=dtype)
    tokenizer=AutoTokenizer.from_pretrained(str(base),local_files_only=True,
                                             trust_remote_code=False,use_fast=True)
    model=Qwen3_5ForCausalLM.from_pretrained(str(base),local_files_only=True,
                                            trust_remote_code=False,
                                            quantization_config=q,dtype=dtype,
                                            device_map={"":0},low_cpu_mem_usage=True)
    model.eval()
    return torch,tokenizer,model


def _raw_answer(torch,tokenizer,model,prompt):
    encoded=tokenizer.apply_chat_template(
        [{"role":"user","content":prompt}],tokenize=True,
        add_generation_prompt=True,enable_thinking=False)
    if isinstance(encoded,dict):encoded=encoded["input_ids"]
    if type(encoded) is not list or len(encoded)>3200:
        raise RuntimeError("Raw input context invalid")
    ids=torch.tensor([encoded],device="cuda",dtype=torch.long)
    try:
        torch.cuda.reset_peak_memory_stats(0)
        with torch.inference_mode():
            out=model.generate(input_ids=ids,do_sample=False,
                               max_new_tokens=768,max_time=55.0,use_cache=True,
                               pad_token_id=tokenizer.pad_token_id,
                               eos_token_id=tokenizer.eos_token_id)
        produced=out[0,len(encoded):].tolist()
        if not produced or produced[-1] not in (
                tokenizer.eos_token_id,tokenizer.convert_tokens_to_ids("<|im_end|>")):
            raise RuntimeError("Raw output incomplete or exceeded fixed budget")
        text=tokenizer.decode(produced,skip_special_tokens=True).strip()
        if "<think>" in text and "</think>" in text:
            text=text.rsplit("</think>",1)[-1].strip()
        if not text or len(text)>16000:raise RuntimeError("Raw response invalid")
        free,total=torch.cuda.mem_get_info(0)
        used=(total-free)/(1024**2)
        if used>10752 or free/(1024**2)<1000:
            raise RuntimeError("Postload GPU total/headroom safety exceeded")
        return text,round(torch.cuda.max_memory_reserved(0)/(1024**2)),round(used)
    finally:
        del ids


def raw(outdir):
    cases,_=_root(outdir)
    validate_engine_identity("raw_qwen35_9b",checkpoint=CHECKPOINT,adapter_loaded=False)
    verify_docker_grader()
    # Caller must have stopped the authenticated owner: no overlapping 9B.
    try:
        OwnerClient().get("/")
    except (OSError,RuntimeError):
        pass
    else:
        raise RuntimeError("Port 9690 still active; refuse concurrent raw 9B")
    output=outdir/"raw9b-50.jsonl"
    if output.exists():raise FileExistsError("Raw baseline already attempted")
    torch,tokenizer,model=_raw_model()
    try:
        with store_private(output) as sink:
            for i,case in enumerate(cases,1):
                started=time.monotonic()
                try:
                    answer,reserved,total=_raw_answer(torch,tokenizer,model,case["prompt"])
                    grade,reason=_score(case,{"text":answer})
                    response={"text":answer}
                except RuntimeError:
                    # A GPU pressure failure is never converted to a scored
                    # answer; do not continue running another GPU question.
                    raise
                elapsed=time.monotonic()-started
                row=make_record(case,"raw_qwen35_9b",response,
                                grade=grade,reason=reason,elapsed=elapsed,
                                total_peak_mib=total,cuda_reserved_mib=reserved)
                sink.write(json.dumps(row,sort_keys=True)+"\n")
                sink.flush()
                print("NERAL_V43_REF50_RAW_PROGRESS "+str(i)+"/50",flush=True)
    finally:
        del model,tokenizer
        gc.collect()
        torch.cuda.empty_cache()
    print("NERAL_V43_REF50_RAW_DONE n=50",flush=True)


def finish(outdir):
    cases,manifest=_root(outdir)
    records=[]
    for name in ("neral10-50.jsonl","raw9b-50.jsonl"):
        path=outdir/name
        if not path.is_file():raise ValueError("Incomplete baseline, cannot publish")
        records.extend(json.loads(l) for l in path.read_text("utf-8").splitlines() if l)
    summary=aggregate(cases,records)
    summary["sealed_sha256"]=manifest["sha256"]
    summary["raw_model_id"]="Qwen3.5-9B HF NF4 without PEFT adapter"
    summary["neral_model_id"]="NERAL 1.0 Obsidian frozen source "+FROZEN
    summary["human_review_pending"]=True
    summary_path=outdir/"summary.json"
    with summary_path.open("x",encoding="utf-8") as sink:
        json.dump(summary,sink,sort_keys=True,indent=2)
    print("NERAL_V43_REF50_100_RESULTS_SEALED "+manifest["sha256"],flush=True)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase",choices=("prepare","neral","raw","finish"),required=True)
    parser.add_argument("--private-dir",type=Path,required=True)
    args=parser.parse_args()
    {"prepare":prepare,"neral":neral,"raw":raw,"finish":finish}[args.phase](args.private_dir)


if __name__=="__main__":
    main()
