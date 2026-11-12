"""Opt-in local GGUF export; never silently replace production Ollama models.

Qwen3.5 multimodal GGUF -> Ollama compatibility depends on runtime/version.
This script writes a Modelfile only after successful local GGUF creation; it
does not call ollama create or claim the imported model works.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--adapter",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--quant",choices=("q4_k_m","q8_0"),default="q4_k_m")
    args=p.parse_args()
    if not (args.adapter/"adapter_config.json").is_file():
        raise FileNotFoundError("A completed PEFT adapter folder is required")
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Refuse to overwrite existing GGUF export")
    # Load the exact fine-tuned checkpoint; no fresh base/training run.
    from unsloth import FastModel
    model,processor=FastModel.from_pretrained(
        model_name=str(args.adapter),max_seq_length=2048,
        load_in_4bit=True,full_finetuning=False)
    if not callable(getattr(model,"save_pretrained_gguf",None)):
        raise RuntimeError("Installed Unsloth version cannot export this Qwen3.5 model")
    args.out.mkdir(parents=True,exist_ok=False)
    model.save_pretrained_gguf(
        str(args.out),processor,quantization_method=args.quant)
    gguf=list(args.out.rglob("*.gguf"))
    if len(gguf)!=1 or gguf[0].stat().st_size<100*1024*1024:
        raise RuntimeError("Missing, split or unexpectedly small GGUF; no Ollama import")
    # Do NOT implicitly overwrite user's existing qwen3.5:9b.
    modelfile=args.out/"Modelfile.neral-qlora-dev"
    modelfile.write_text(
        "FROM "+str(gguf[0].resolve()).replace("\\","/")+"\n"
        "PARAMETER num_ctx 4096\n"
        "PARAMETER temperature 0\n",
        encoding="utf-8")
    print("GGUF_EXPORTED_UNVALIDATED")
    print("Ollama candidate Modelfile:",modelfile)
    print("Suggested manual import after verifying model architecture support:")
    print("  ollama create neral-qwen35-qlora:dev -f",modelfile)
    print("Then perform true A/B tests with a matching quantized baseline.")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
