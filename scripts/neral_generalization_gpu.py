#!/usr/bin/env python3
"""Read-only, paired local NF4 Qwen / NERAL rank32 32-case development comparison.

The same verified model is loaded ONCE. Base inference runs with the LoRA
disabled; NERAL inference runs with that existing LoRA enabled. Only aggregate
counts and sealed identity hashes may be exported; raw output is never saved,
executed, returned by an API or uploaded. No training, tool use or promotion.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import re
import sys

from scripts.neral_generalization_probe import CASES, compare_responses

PINNED_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
REPORT_KEYS = frozenset((
    "schema", "status", "question_count", "base_correct", "adapter_correct",
    "delta_adapter_vs_base", "by_family", "independent_external_holdout",
    "model_quality_improvement_proven", "training_performed",
    "raw_prompts_uploaded", "base_revision", "adapter_sha256",
    "max_new_tokens", "adapter_modified", "source_training_changed",
    "data_sent_to_external_model_provider", "model_origin",
))


def build_aggregate(base: dict, adapter: dict, *, base_revision: str,
                    adapter_sha256: str, max_new_tokens: int) -> dict:
    if base_revision != PINNED_REVISION or not re.fullmatch(r"[0-9a-f]{64}",adapter_sha256):
        raise ValueError("Unverified foundation or adapter identity")
    if max_new_tokens not in (96,160):
        raise ValueError("Unreviewed generation budget")
    result = compare_responses(base, adapter)
    result.update(base_revision=base_revision, adapter_sha256=adapter_sha256,
                  max_new_tokens=max_new_tokens, adapter_modified=False,
                  source_training_changed=False,
                  data_sent_to_external_model_provider=False,
                  model_origin="local_qwen25_coder7b_plus_experimental_neral_rank32")
    validate_report(result)
    return result


def validate_report(report: object) -> bool:
    if not isinstance(report,dict) or set(report) != REPORT_KEYS:
        raise ValueError("Unexpected scalar report fields")
    if (report.get("schema") != 1 or
        report.get("status") != "PROJECT_INTERNAL_DEV_COMPARISON_NOT_RELEASE_VALIDATION" or
        type(report.get("question_count")) is not int or report["question_count"] != 32 or
        report.get("base_revision") != PINNED_REVISION or
        report.get("model_origin") != "local_qwen25_coder7b_plus_experimental_neral_rank32" or
        not isinstance(report.get("adapter_sha256"),str) or
        not re.fullmatch(r"[0-9a-f]{64}",report["adapter_sha256"]) or
        report.get("max_new_tokens") not in (96,160)):
        raise ValueError("Wrong report identity")
    for key in ("independent_external_holdout","model_quality_improvement_proven",
                "training_performed","raw_prompts_uploaded","adapter_modified",
                "source_training_changed","data_sent_to_external_model_provider"):
        if report.get(key) is not False:
            raise ValueError("Invalid training/privacy or quality claim")
    for key in ("base_correct","adapter_correct"):
        value=report.get(key)
        if type(value) is not int or not 0<=value<=32:
            raise ValueError("Invalid benchmark score")
    if type(report.get("delta_adapter_vs_base")) is not int or (
        report["delta_adapter_vs_base"] != report["adapter_correct"]-report["base_correct"]):
        raise ValueError("Inconsistent benchmark difference")
    families = report.get("by_family")
    if not isinstance(families,dict) or set(families)!= {"math","sql","coding_ast","agent"}:
        raise ValueError("Unexpected benchmark families")
    for family in families.values():
        if not isinstance(family,dict) or set(family)!= {"base_correct","adapter_correct","total"}:
            raise ValueError("Invalid family metrics")
        if family["total"] != 8 or type(family["base_correct"]) is not int or (
            type(family["adapter_correct"]) is not int) or not (
            0<=family["base_correct"]<=8 and 0<=family["adapter_correct"]<=8):
            raise ValueError("Invalid family counts")
    if sum(x["base_correct"] for x in families.values())!=report["base_correct"] or (
        sum(x["adapter_correct"] for x in families.values())!=report["adapter_correct"]):
        raise ValueError("Per-family and aggregate scores differ")
    if len(json.dumps(report,ensure_ascii=True)) > 12000:
        raise ValueError("Report unexpectedly large")
    return True


def _offline() -> None:
    os.environ.update({
        "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
        "HF_DATASETS_OFFLINE":"1", "HF_HUB_DISABLE_TELEMETRY":"1",
        "WANDB_DISABLED":"true", "WANDB_MODE":"disabled",
    })


def _generate(model, tokenizer, *, max_new_tokens: int) -> dict[str,str]:
    import torch
    from qwen_local_runtime import CHAT_TEMPLATE
    model.eval()
    answers={}
    for index, case in enumerate(CASES,1):
        # Model outputs are never executed as code or treated as tool calls.
        instruction=("Du bist NERAL im lokalen Prüfmodus. Folge exakt dem "
                     "geforderten Ausgabeformat. Keine Werkzeugausführung, "
                     "keine Markdown-Markierung.")
        msg=[{"role":"system","content":instruction},
             {"role":"user","content":case["prompt"]}]
        tokens=tokenizer.apply_chat_template(msg,chat_template=CHAT_TEMPLATE,
                     tokenize=True,add_generation_prompt=True,
                     return_tensors="pt").to("cuda:0")
        if tokens.shape[-1]>3072:
            raise ValueError("Unbounded model prompt")
        with torch.inference_mode():
            generated=model.generate(tokens,attention_mask=torch.ones_like(tokens),
                                     do_sample=False,max_new_tokens=max_new_tokens,
                                     pad_token_id=tokenizer.pad_token_id)
        text=tokenizer.decode(generated[0,tokens.shape[-1]:],skip_special_tokens=True)
        answers[case["id"]]=text
        # Do not log text, paths, model contents or private user prompts.
        print(f"Case {index}/32: local inference complete",flush=True)
    return answers


def _run(args) -> dict:
    _offline()
    from scripts.neral_math_v2_eval import ensure_pinned_model_runtime
    ensure_pinned_model_runtime(args.trainer_root)
    from model_catalog import validate_hf_snapshot
    from neral_adapter_candidate_r32 import validate_adapter
    from scripts.neral_math_eval import _load_model
    if any(path.is_symlink() or not path.is_dir() for path in
           (args.base,args.adapter,args.trainer_root)):
        raise ValueError("Trusted, existing local model and runtime directories required")
    snapshot=validate_hf_snapshot(args.base,verify_hashes=True)
    if snapshot.get("revision") != PINNED_REVISION:
        raise ValueError("Wrong Qwen foundation revision")
    sealed=validate_adapter(args.adapter,snapshot)
    if sealed.get("release_validated"):
        raise ValueError("Only experimental, unpromoted NERAL adapter allowed")
    adapter_sha=sealed.get("files",{}).get("adapter_model.safetensors")
    if not isinstance(adapter_sha,str):
        raise ValueError("No verified adapter checksum")
    import torch
    from peft import PeftModel
    from transformers import Qwen2TokenizerFast
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise ValueError("Local CUDA GPU with BF16 required")
    free,_=torch.cuda.mem_get_info(0)
    if free < 10_000*1024*1024:
        raise ValueError("Under 10000 MiB free GPU memory; no loading allowed")
    tokenizer=Qwen2TokenizerFast.from_pretrained(str(args.base),local_files_only=True,
                                                 trust_remote_code=False,token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token=tokenizer.eos_token
    model=_load_model(args.base)
    try:
        model=PeftModel.from_pretrained(model,str(args.adapter),
                                       is_trainable=False,local_files_only=True)
        with model.disable_adapter():
            base=_generate(model,tokenizer,max_new_tokens=args.max_new_tokens)
        custom=_generate(model,tokenizer,max_new_tokens=args.max_new_tokens)
        result=build_aggregate(base,custom,base_revision=snapshot["revision"],
                               adapter_sha256=adapter_sha,
                               max_new_tokens=args.max_new_tokens)
        return result
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()


def main(argv=None) -> int:
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--base",required=True,type=Path)
    p.add_argument("--adapter",required=True,type=Path)
    p.add_argument("--trainer-root",required=True,type=Path)
    p.add_argument("--summary",required=True,type=Path)
    p.add_argument("--max-new-tokens",required=False,type=int,default=160,
                   choices=(96,160))
    args=p.parse_args(argv)
    if args.summary.exists() or args.summary.is_symlink() or not args.summary.parent.is_dir():
        p.error("A new summary filename inside an existing local directory is required")
    report=_run(args)
    with args.summary.open("x",encoding="utf8",newline="\n") as output:
        json.dump(report,output,indent=2,sort_keys=True,ensure_ascii=True)
        output.write("\n")
    print("NERAL_PAIRED_DEV_COMPARISON_COMPLETE; NO_TRAINING; NO_PROMPT_EXPORT",flush=True)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
