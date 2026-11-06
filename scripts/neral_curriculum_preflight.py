#!/usr/bin/env python3
"""Read-only Qwen tokenizer/completion-mask preflight for 1,000 NERAL examples.

No training, adapter modifications, internet calls or model generation. The
real model verification is possible ONLY through the --model-dir CLI path;
pure test tokenizers are never reported as verified Qwen.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from scripts.validate_neral_curriculum import validate

SOURCE_PINS = {
    "train": "48c50127de7396c77195a59a11c94740ab5cb83f",
    "val": "caa8ebbad4079df9928ddd3b4ba79b1df3442551",
    "test": "82f9def4ee01ff491644eb577ad9b533e765762f",
    "manifest": "15eee32d3ca5577def6e9b7476f84541ac385406",
}
EXPECTED = {"train":800,"val":100,"test":100}
MAX_FILE_BYTES = 1000000


def _git_blob_digest(raw: bytes) -> str:
    header = b"blob " + str(len(raw)).encode("ascii") + b"\x00"
    return hashlib.sha1(header + raw).hexdigest()


def verify_pinned_files(root: Path) -> dict[str,str]:
    root = Path(root)
    observed = {}
    for split, blob in SOURCE_PINS.items():
        filename = root / (split + ".jsonl" if split != "manifest" else "manifest.json")
        if filename.is_symlink() or not filename.is_file() or filename.stat().st_size > MAX_FILE_BYTES:
            raise ValueError("Missing/oversized/symlinked curriculum source pin")
        raw = filename.read_bytes()
        if _git_blob_digest(raw) != blob:
            raise ValueError("Curriculum source git blob pin mismatch; refuse unchecked edits")
        observed[split] = hashlib.sha256(raw).hexdigest()
    return observed


def _example_token_lengths(tokenizer, template: str, messages: list[dict], max_length: int) -> tuple[int,int]:
    prefix = tokenizer.apply_chat_template(messages[:2], chat_template=template,
                                           tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(messages, chat_template=template,
                                         tokenize=True, add_generation_prompt=False)
    if (not isinstance(prefix,list) or not isinstance(full,list) or not prefix
            or full[:len(prefix)] != prefix or len(full) <= len(prefix)):
        raise ValueError("Tokenizer completion boundary mismatch")
    if len(full) > max_length:
        raise ValueError("Tokenizer sequence length exceeds reviewed max_length")
    # Explicitly prove the loss mask is 0 on system/user and 1 ONLY on
    # assistant completion, without building or uploading tensor data.
    mask = [0] * len(prefix) + [1] * (len(full)-len(prefix))
    if len(mask) != len(full) or sum(mask) != len(full)-len(prefix):
        raise ValueError("Completion-only training mask invalid")
    if mask[:len(prefix)] != [0]*len(prefix) or mask[len(prefix):] != [1]*(len(full)-len(prefix)):
        raise ValueError("Completion mask crosses assistant boundary")
    return len(prefix), len(full)-len(prefix)


def prepare_report(root: Path, tokenizer, template: str, *, max_length: int = 512) -> dict:
    if max_length not in (512,1024,2048):
        raise ValueError("Unsupported reviewed maximum sequence length")
    if not isinstance(template,str) or not template:
        raise ValueError("A pinned model chat template is required")
    source_sha = verify_pinned_files(root)
    checked = validate(root)
    if checked["total_rows"] != 1000 or checked["unique_prompts"] != 1000:
        raise ValueError("Curriculum contamination or missing rows")
    manifest = json.loads((Path(root) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("training_allowed") is not False or manifest.get("quality_improvement_proven") is not False:
        raise ValueError("Expected review-only training data; no implicit promotion")
    stats = {}
    for split, expected in EXPECTED.items():
        path = Path(root) / (split + ".jsonl")
        prompts, completions = [], []
        with path.open(encoding="utf-8",newline="\n") as stream:
            for line_no, line in enumerate(stream, 1):
                row = json.loads(line)
                prompt_len, completion_len = _example_token_lengths(
                    tokenizer, template, row["messages"], max_length)
                prompts.append(prompt_len)
                completions.append(completion_len)
        if len(prompts) != expected:
            raise ValueError("Unexpected number of tokenized rows")
        stats[split] = {
            "examples": len(prompts),
            "prompt_tokens_min": min(prompts),
            "prompt_tokens_max": max(prompts),
            "completion_tokens_min": min(completions),
            "completion_tokens_max": max(completions),
            "total_tokens_max": max(a+b for a,b in zip(prompts,completions)),
        }
    return {
        "schema":1,
        "status":"TOKENIZER_INTERFACE_PASSED_NOT_MODEL_VERIFIED",
        "rows":EXPECTED,
        "total_rows":1000,
        "trainable_examples":800,
        "max_length":max_length,
        "source_git_blob_pins":SOURCE_PINS,
        "files_sha256":source_sha,
        "token_stats":stats,
        "tokenizer_contract_checked":True,
        "completion_only_mask_checked":True,
        "local_qwen_tokenizer_verified":False,
        "training_allowed":False,
        "training_performed":False,
        "adapter_modified":False,
        "new_adapter_quality_proven":False,
        "uploaded_user_data":False,
    }


def _load_local_tokenizer(model_dir: Path, trainer_root: Path):
    # Enforce offline behavior BEFORE importing transformers or tokenizers.
    os.environ.update({
        "HF_HUB_OFFLINE":"1", "TRANSFORMERS_OFFLINE":"1",
        "HF_DATASETS_OFFLINE":"1", "WANDB_DISABLED":"true",
        "WANDB_MODE":"disabled", "HF_HUB_DISABLE_TELEMETRY":"1",
    })
    original=Path(trainer_root)
    root=original.resolve(strict=True)
    if original.is_symlink() or not root.is_dir():
        raise ValueError("Pinned original trainer root missing")
    for name in ("model_catalog.py","qwen_local_runtime.py"):
        file=root/name
        if not file.is_file() or file.is_symlink():
            raise ValueError("Pinned runtime modules missing")
    sys.path.insert(0,str(root))
    from model_catalog import validate_hf_snapshot
    from qwen_local_runtime import CHAT_TEMPLATE
    from transformers import Qwen2TokenizerFast
    snapshot=validate_hf_snapshot(Path(model_dir).resolve(strict=True),verify_hashes=True)
    if snapshot["base_model"] != "Qwen/Qwen2.5-Coder-7B-Instruct":
        raise ValueError("Wrong verified local Qwen base")
    tokenizer=Qwen2TokenizerFast.from_pretrained(str(snapshot["folder"]),
        local_files_only=True,trust_remote_code=False,token=False)
    return tokenizer,CHAT_TEMPLATE,snapshot["revision"]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--model-dir",type=Path,required=True)
    parser.add_argument("--trainer-root",type=Path,required=True)
    parser.add_argument("--summary",type=Path,required=True)
    args=parser.parse_args(argv)
    if args.summary.is_symlink() or args.summary.exists() or not args.summary.parent.is_dir():
        parser.error("Summary output must be a new file in an existing folder")
    tokenizer,template,revision=_load_local_tokenizer(args.model_dir,args.trainer_root)
    result=prepare_report(args.data,tokenizer,template)
    result["status"]="LOCAL_QWEN_TOKENIZER_PREFLIGHT_PASSED"
    result["local_qwen_tokenizer_verified"]=True
    result["base_revision"]=revision
    with args.summary.open("x",encoding="utf-8",newline="\n") as output:
        json.dump(result,output,indent=2,sort_keys=True,allow_nan=False)
        output.write("\n")
    print("NERAL_CURRICULUM_TOKENIZER_PREFLIGHT_COMPLETE; NO TRAINING")

if __name__=="__main__":
    main()
