"""Read-only paired local 7B+NERAL adapter evaluation. NO training or promotion.

Uses one PEFT 4-bit model. Qwen base is evaluated with adapter disabled,
then the same verified adapter enabled. All generated text remains on the
local Windows runner; only aggregate counts/hashes enter GitHub artifacts.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.neral_generalization_probe import CASES, compare_responses

LOCAL_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "WANDB_MODE": "disabled",
    "WANDB_DISABLED": "true",
    "HF_HUB_DISABLE_TELEMETRY": "1",
}
SYSTEM = ("Du bist ein lokal betriebenes Coding- und Analysemodell. "
          "Antworte exakt im vom Nutzer verlangten Format. "
          "Erfinde keine überprüften Werkzeugergebnisse oder Tests. "
          "Führe keine Aktionen aus.")


def _generate(backend, label: str) -> dict[str,str]:
    responses = {}
    for i, task in enumerate(CASES, 1):
        print(f"LOCAL_{label.upper()}_CASE={i:02d}/32", flush=True)
        reply = backend.respond([
            {"role":"system","content":SYSTEM},
            {"role":"user","content":task["prompt"]},
        ])
        if not isinstance(reply,str):
            raise ValueError("Model returned an invalid response")
        responses[task["id"]] = reply[:10000]
    return responses


def evaluate(base: Path, adapter: Path, out: Path, summary: Path) -> dict:
    for key,value in LOCAL_ENV.items():
        os.environ[key] = value
    if out.exists() or out.is_symlink() or not out.parent.is_dir():
        raise ValueError("Local result directory must be new")
    if summary.exists() or summary.is_symlink() or not summary.parent.is_dir():
        raise ValueError("Summary destination must be new")
    for folder in (base,adapter):
        if folder.is_symlink() or not folder.is_dir():
            raise ValueError("Verified existing local model directories required")
    from neral_agent_hf import HFAdapterLocalBackend
    backend=HFAdapterLocalBackend(base,adapter,max_new_tokens=160)
    backend._load()
    if getattr(backend,"_model",None) is None or not hasattr(backend._model,"disable_adapter"):
        raise ValueError("Verified PEFT model with disable_adapter required")
    start=time.monotonic()
    # Identical local snapshot, token budgets and decoding for both arms.
    with backend._model.disable_adapter():
        base_out=_generate(backend,"base")
    adapter_out=_generate(backend,"adapter")
    metrics=compare_responses(base_out,adapter_out)
    metrics.update({
        "model_backend":"offline_local_hf_peft",
        "base_model":"Qwen/Qwen2.5-Coder-7B-Instruct",
        "adapter_sha256":backend.adapter_sha256,
        "base_adapter_disabled":True,
        "paired_settings_identical":True,
        "max_new_tokens":160,
        "elapsed_seconds":round(time.monotonic()-start,2),
        "no_cloud_inference":True,
        "adapter_promoted":False,
        "weights_modified":False,
        "training_performed":False,
    })
    # Raw generations retained ONLY locally for diagnosis and never as Actions artifacts.
    out.mkdir(mode=0o700)
    (out/"local_only_raw_answers.json").write_text(
        json.dumps({"base":base_out,"adapter":adapter_out},ensure_ascii=False,indent=2)+"\n",
        encoding="utf-8")
    with summary.open("x",encoding="utf-8") as stream:
        json.dump(metrics,stream,ensure_ascii=True,sort_keys=True,indent=2,allow_nan=False)
        stream.write("\n")
    print(f"NERAL_LOCAL_DEV_COMPARE_COMPLETE base={metrics['base_correct']}/32 adapter={metrics['adapter_correct']}/32",flush=True)
    return metrics


def main(argv=None) -> int:
    parser=argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--base",required=True,type=Path)
    parser.add_argument("--adapter",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--summary",required=True,type=Path)
    args=parser.parse_args(argv)
    evaluate(args.base,args.adapter,args.out,args.summary)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
