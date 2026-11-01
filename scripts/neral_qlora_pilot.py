"""NERAL QLoRA pilot. Offline preflight by default; explicit GPU opt-in required."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time
import platform
import math
import threading
from importlib.metadata import version

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
ROLES = ("system", "user", "assistant")
PRIVATE_DATA = re.compile(r"(?i)(?:sk-[A-Za-z0-9_-]{16,}|ghp_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")


def verify_review(dataset, manifest):
    """Integrity of a code-reviewed corpus; format checks alone cannot prove truth."""
    manifest = Path(manifest)
    if not manifest.is_file() or manifest.stat().st_size > 1024*1024:
        raise ValueError("Review missing or too large.")
    review = json.loads(Path(manifest).read_text(encoding="utf-8"))
    if (not isinstance(review, dict) or review.get("schema") != 1
            or not isinstance(review.get("sources"), dict) or not review["sources"]
            or not isinstance(review.get("rows"), list) or not review["rows"]):
        raise ValueError("Review requires source evidence and complete row coverage.")
    if review["dataset_sha256"] != hashlib.sha256(Path(dataset).read_bytes()).hexdigest():
        raise ValueError("Dataset differs from reviewed corpus; review it before training.")
    read_examples(dataset)
    lines = Path(dataset).read_text(encoding="utf-8").splitlines()
    if (len(review["rows"]) != len(lines)
            or any(not isinstance(row, dict) or type(row.get("line")) is not int for row in review["rows"])
            or {row["line"] for row in review["rows"]} != set(range(1, len(lines)+1))):
        raise ValueError("Review must cover every example exactly once.")
    for row in review["rows"]:
        digest = hashlib.sha256(lines[row["line"]-1].encode()).hexdigest()
        if row.get("row_sha256") != digest or not row.get("evidence") or not row.get("finding"):
            raise ValueError("Reviewed row differs or lacks evidence.")
    root = Path(__file__).resolve().parents[1]
    for source, digest in review["sources"].items():
        path = (root / source).resolve()
        if root not in path.parents or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Reviewed source changed; refresh the factual review.")
    return review


def tokenize_examples(rows, tokenizer, max_length):
    # Share the audited inference template, never execute an imported Jinja template.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from qwen_local_runtime import CHAT_TEMPLATE
    result = []
    for number, row in enumerate(rows, 1):
        prompt = tokenizer.apply_chat_template(row["prompt"], chat_template=CHAT_TEMPLATE,
                                               tokenize=True, add_generation_prompt=True)
        full = tokenizer.apply_chat_template(row["prompt"] + row["completion"],
                                             chat_template=CHAT_TEMPLATE, tokenize=True,
                                             add_generation_prompt=False)
        if full[:len(prompt)] != prompt or len(full) <= len(prompt):
            raise ValueError(f"Row {number}: invalid completion boundary.")
        if len(full) > max_length:
            raise ValueError(f"Row {number}: {len(full)} tokens exceeds {max_length}; no silent truncation.")
        result.append({"input_ids": full, "completion_mask": [0]*len(prompt) + [1]*(len(full)-len(prompt))})
    return result


def read_examples(path):
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 1024 * 1024:
        raise ValueError("Dataset missing or too large for a pilot (1 MiB cap).")
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"Empty line {number}.")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {number}.") from exc
        if not isinstance(row, dict) or set(row) != {"messages"}:
            raise ValueError(f"Line {number}: expected only 'messages'.")
        messages = row["messages"]
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"Line {number}: exactly system/user/assistant required.")
        for role, message in zip(ROLES, messages):
            if not isinstance(message, dict) or set(message) != {"role", "content"} or message["role"] != role:
                raise ValueError(f"Line {number}: invalid role or fields.")
            value = message["content"]
            if not isinstance(value, str) or not value.strip() or len(value) > 8192:
                raise ValueError(f"Line {number}: empty or oversized content.")
            if PRIVATE_DATA.search(value) or "<think>" in value or "</think>" in value:
                raise ValueError(f"Line {number}: secrets or invented thinking tags forbidden.")
        rows.append({"prompt": messages[:2], "completion": [messages[2]]})
    if not rows or len(rows) > 2000:
        raise ValueError("Pilot expects 1..2000 examples.")
    return rows


def ensure_output_dir(path):
    path = Path(path).expanduser().resolve()
    repo = Path(__file__).resolve().parents[1]
    if path == repo or repo in path.parents:
        raise ValueError("Adapter output must be outside the source repository.")
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Output directory must be empty or not yet exist.")
    return path


def run_training(args, rows):
    if not args.model_dir:
        raise ValueError("--model-dir is required for GPU training.")
    output = ensure_output_dir(args.output_dir)
    model_folder = Path(args.model_dir).expanduser().resolve()
    if output == model_folder or model_folder in output.parents or output in model_folder.parents:
        raise ValueError("Output must be separate from the base snapshot.")
    verify_review(args.dataset, args.review_manifest)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from model_catalog import validate_hf_snapshot
    snapshot = validate_hf_snapshot(Path(args.model_dir).expanduser().resolve(), verify_hashes=True)
    if snapshot["base_model"] != MODEL_ID:
        raise ValueError("Wrong base model.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA device with BF16 support required.")
    from transformers import Qwen2TokenizerFast, TrainerCallback
    from neral_training_preflight import check_stack
    compatibility = check_stack(cuda=True)

    model_path = str(snapshot["folder"])
    tokenizer = Qwen2TokenizerFast.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoded = tokenize_examples(rows, tokenizer, args.max_length)
    output.mkdir(parents=True, exist_ok=True)
    report = {"status": "started", "platform": platform.platform(), "python": platform.python_version(),
              "packages": {name: version(name) for name in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "bitsandbytes", "safetensors")},
              "gpu": torch.cuda.get_device_name(0), "torch_cuda": torch.version.cuda,
              "steps": args.steps, "max_length": args.max_length, "logs": [],
              "compatibility": compatibility,
              "total_tokens_per_epoch": sum(len(r["input_ids"]) for r in encoded)}
    class Metrics(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            if any(isinstance(v, float) and not math.isfinite(v) for v in (logs or {}).values()):
                raise FloatingPointError("Non-finite training metric; pilot stopped.")
            report["logs"].append({"step": state.global_step, **(logs or {})})
    samples = []
    stopped = threading.Event()
    def sample():
        try:
            while not stopped.is_set():
                free, total = torch.cuda.mem_get_info(0)
                samples.append(total-free)
                stopped.wait(0.1)
        except Exception as exc:
            report["device_sampling_error"] = type(exc).__name__
    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    try:
        _train(args, encoded, tokenizer, snapshot, output, report, Metrics())
        report["status"] = "completed"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        stopped.set()
        sampler.join(timeout=2)
        report.update(elapsed_seconds=time.perf_counter()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated(0),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(0),
                      device_used_peak_sampled_bytes=max(samples, default=None),
                      device_sample_count=len(samples), device_sample_interval_seconds=0.1)
        (output / "training-report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")


def _train(args, encoded, tokenizer, snapshot, output, report, metrics):
    import torch
    from datasets import Dataset
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import Qwen2ForCausalLM, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    model = Qwen2ForCausalLM.from_pretrained(
        str(snapshot["folder"]), local_files_only=True, trust_remote_code=False, token=False,
        use_safetensors=True, attn_implementation="sdpa", device_map={"": 0},
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16),
    )
    model.config.use_cache = False
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("NF4 loading was not confirmed.")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                           gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, lora_dropout=0, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    ))
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable or any("lora_" not in name for name in trainable):
        raise RuntimeError("Only LoRA parameters may be trainable; base weights must stay frozen.")
    config = SFTConfig(
        output_dir=str(output), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, max_steps=args.steps, max_length=args.max_length,
        learning_rate=2e-4, bf16=True, fp16=False, optim="paged_adamw_8bit",
        gradient_checkpointing=True, packing=False, completion_only_loss=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        save_strategy="no", eval_strategy="no", report_to="none", logging_steps=1,
        dataloader_num_workers=0, seed=42,
    )
    trainer = SFTTrainer(model=model, args=config, callbacks=[metrics],
                         train_dataset=Dataset.from_list(encoded), processing_class=tokenizer)
    report["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["metrics"] = trainer.train().metrics
    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output), safe_serialization=True)
    metadata = {
        "name": "NERAL QLoRA pilot (not a validated release)",
        "base_model": MODEL_ID,
        "base_revision": snapshot["revision"],
        "base_snapshot_integrity": snapshot["sha256"],
        "dataset_sha256": hashlib.sha256(Path(args.dataset).read_bytes()).hexdigest(),
        "steps": args.steps, "max_length": args.max_length,
        "format": "PEFT LoRA adapter", "inference_in_helios_verified": False,
    }
    (output / "neral-training-manifest.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    from neral_adapter_candidate import seal_adapter
    seal_adapter(output, snapshot, metadata["dataset_sha256"], args.adapter_version)
    print(f"Adapter saved: {output}; NOT integrated into Helios.")


def main(argv=None):
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=str(root / "datasets" / "neral_qlora_pilot.jsonl"))
    parser.add_argument("--model-dir", help="Previously verified, local Qwen HF snapshot directory")
    parser.add_argument("--review-manifest", default=str(root / "datasets" / "neral_qlora_pilot.review.json"))
    parser.add_argument("--check-review", action="store_true", help="Check source review for a custom corpus without training")
    parser.add_argument("--adapter-version", default="0.1.0-pilot.1")
    parser.add_argument("--output-dir", default=str(Path.home() / "neral-qlora-adapter-pilot"))
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--run-local-gpu", action="store_true", help="Explicitly opt into local training")
    args = parser.parse_args(argv)
    if not 1 <= args.steps <= 200 or args.max_length not in (256, 512, 1024, 2048):
        parser.error("steps must be 1..200; max-length must be 256, 512, 1024 or 2048")
    rows = read_examples(args.dataset)
    if not re.fullmatch(r"\d+\.\d+\.\d+(?:-pilot\.\d+)?", args.adapter_version):
        parser.error("adapter-version must be x.y.z or x.y.z-pilot.n")
    print(f"NERAL examples format-checked: {len(rows)}; model={MODEL_ID}; steps={args.steps}; max_length={args.max_length}")
    if args.check_review or Path(args.dataset).resolve() == (root / "datasets" / "neral_qlora_pilot.jsonl").resolve():
        verify_review(args.dataset, args.review_manifest)
        print("Reviewed corpus/source integrity: passed (semantic review recorded in manifest).")
    if not args.run_local_gpu:
        print("PLAN ONLY: no imports of GPU libraries, no model load, no training, no adapter files.")
        return
    run_training(args, rows)


if __name__ == "__main__":
    main()
