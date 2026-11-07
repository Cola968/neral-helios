"""NERAL v5 research-only A100 QLoRA on rented Linux GPUs.

Explicit modes; no cloud creation/billing, no automatic deployment or upload.
Only the pinned Qwen commit and deterministic v5 TRAIN split are consumed.
An A100 40 GB, PyTorch 2.8.0+cu128, and Python >=3.10 are required.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
import time

MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
ROOT = Path("/workspace/neral-cloud")
BASE = ROOT / "base"
DATA = ROOT / "data"
OUTPUT = ROOT / "output"
REQUIRED_FILES = frozenset({"config.json", "tokenizer.json", "tokenizer_config.json"})
EXTRAS = frozenset({
    "config.json", "generation_config.json", "tokenizer.json",
    "tokenizer_config.json", "special_tokens_map.json", "vocab.json",
    "merges.txt", "model.safetensors.index.json",
})
WEIGHTS = re.compile(r"model(?:-\d{5}-of-\d{5})?\.safetensors\Z")


def select_file(name: str) -> bool:
    return isinstance(name, str) and Path(name).name == name and (
        name in EXTRAS or bool(WEIGHTS.fullmatch(name))
    )


def validate_experiment(mode: str, rank: int, steps: int, max_length: int):
    allowed = {"smoke": (16, 3, 512), "train": (32, 480, 512)}
    if (mode not in allowed or (rank, steps, max_length) != allowed[mode]
            or any(type(x) is not int for x in (rank, steps, max_length))):
        raise ValueError("Only smoke r16/3/512 or isolated train r32/480/512 is approved")
    return rank, steps, max_length


def check_device(name: str, total_bytes: int, free_bytes: int) -> bool:
    if (not isinstance(name, str) or "A100" not in name or
            total_bytes < 38 * 1024**3 or free_bytes < 32 * 1024**3 or
            free_bytes > total_bytes):
        raise ValueError("Require a mostly idle single A100 with at least 40 GB VRAM")
    return True


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _validate_snapshot(base: Path):
    # The original catalog validator lives in the pinned trainer commit.
    pinned = Path("/workspace/neral-cloud-pinned")
    module = pinned / "model_catalog.py"
    if not module.is_file():
        raise RuntimeError("Checkout pinned original trainer at a082738f9145722dec443141e5d7801a3ead93af")
    spec = importlib.util.spec_from_file_location("neral_vast_pinned_catalog", module)
    if spec is None or spec.loader is None:
        raise ValueError("Pinned catalog cannot be imported")
    original = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(original)
    checked = original.validate_hf_snapshot(base, verify_hashes=True)
    if checked["revision"] != BASE_REVISION or checked["base_model"] != MODEL_ID:
        raise ValueError("Pinned model provenance mismatch")
    return checked


def prepare_model(base: Path = BASE):
    if base.is_dir():
        return _validate_snapshot(base)
    if base.exists() or not base.parent.is_dir():
        raise ValueError("Model destination must be new, inside prepared workspace")
    from huggingface_hub import snapshot_download
    downloaded = Path(snapshot_download(
        repo_id=MODEL_ID, revision=BASE_REVISION,
        allow_patterns=sorted(EXTRAS) + ["model.safetensors", "model-*.safetensors"],
        cache_dir=str(ROOT / "hf-cache"), max_workers=4, token=False,
    ))
    if downloaded.name != BASE_REVISION:
        raise ValueError("Hugging Face returned unpinned model revision")
    names = {x.name for x in downloaded.iterdir() if x.is_file() and select_file(x.name)}
    if (not REQUIRED_FILES <= names or
            not any(WEIGHTS.fullmatch(name) for name in names) or
            len(names) > 64):
        raise ValueError("Incomplete or unexpected Qwen snapshot")
    tmp = base.with_name(base.name + "-staging")
    if tmp.exists():
        raise ValueError("Unreviewed partial snapshot staging directory exists")
    tmp.mkdir(mode=0o700)
    try:
        hashes = {}
        for name in sorted(names):
            file = downloaded / name
            if not file.resolve().is_file():
                raise ValueError("Snapshot file missing")
            shutil.copyfile(file, tmp / name)
            hashes[name] = _sha(tmp / name)
        (tmp / "helios-manifest.json").write_text(
            json.dumps({"base_model": MODEL_ID, "revision": BASE_REVISION,
                        "files": hashes}, indent=2, sort_keys=True) + "\n",
            encoding="utf8",
        )
        checked = _validate_snapshot(tmp)
        tmp.rename(base)
        return checked
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise


def prepare_data():
    from scripts.neral_v5_transfer_curriculum import build, verify
    if DATA.exists():
        if not verify(DATA)["valid"]:
            raise ValueError("Cloud data differs from the approved v5 generator")
    else:
        build(DATA)
    return verify(DATA)


def gpu_preflight():
    import torch
    import bitsandbytes as bnb
    if torch.__version__ != "2.8.0+cu128" or torch.version.cuda != "12.8":
        raise RuntimeError("Unexpected PyTorch/CUDA wheel; require torch==2.8.0+cu128")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA device unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Require exactly one GPU")
    total = torch.cuda.get_device_properties(0).total_memory
    free, _ = torch.cuda.mem_get_info(0)
    check_device(torch.cuda.get_device_name(0), total, free)
    matrix = torch.randn(64, 64, device="cuda:0", dtype=torch.bfloat16)
    quant, state = bnb.functional.quantize_4bit(
        matrix, quant_type="nf4", compress_statistics=True
    )
    back = bnb.functional.dequantize_4bit(quant, state)
    if not torch.isfinite(back).all():
        raise RuntimeError("NF4 CUDA self-test failed")
    return {"gpu": torch.cuda.get_device_name(0), "total_vram_bytes": total,
            "free_vram_bytes": free, "nf4_test": "passed",
            "torch": torch.__version__, "cuda": torch.version.cuda}


def _tokenize(rows, tokenizer):
    from qwen_local_runtime import CHAT_TEMPLATE
    enc = []
    for index, item in enumerate(rows):
        if (not isinstance(item, dict) or
                set(item) != {"prompt", "completion"} or
                len(item["prompt"]) != 2 or len(item["completion"]) != 1):
            raise ValueError(f"Invalid data record {index}")
        prompt = tokenizer.apply_chat_template(
            item["prompt"], chat_template=CHAT_TEMPLATE, tokenize=True,
            add_generation_prompt=True,
        )
        full = tokenizer.apply_chat_template(
            item["prompt"] + item["completion"], chat_template=CHAT_TEMPLATE,
            tokenize=True, add_generation_prompt=False,
        )
        if full[:len(prompt)] != prompt or not len(prompt) < len(full) <= 512:
            raise ValueError(f"Training completion boundary/truncation at row {index}")
        enc.append({"input_ids": full,
                    "completion_mask": [0]*len(prompt) + [1]*(len(full)-len(prompt))})
    if len(enc) != 1920:
        raise ValueError("Expected exactly 1,920 v5 training examples")
    return enc


def _train(mode: str, rank: int, steps: int, max_length: int):
    validate_experiment(mode, rank, steps, max_length)
    if not (DATA.is_dir() and BASE.is_dir()):
        raise RuntimeError("Run prepare first")
    from scripts.neral_v5_transfer_curriculum import verify
    if not verify(DATA)["valid"]:
        raise RuntimeError("The exact v5 corpus failed verification")
    snapshot = _validate_snapshot(BASE)
    gpu = gpu_preflight()
    import torch
    from transformers import Qwen2TokenizerFast, Qwen2ForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer
    from datasets import Dataset

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        str(BASE), local_files_only=True, trust_remote_code=False, token=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = [json.loads(line) for line in (DATA / "train.jsonl").read_text(
        encoding="utf8").splitlines()]
    encoded = _tokenize(records, tokenizer)

    dest = OUTPUT / f"neral-v5-cloud-{mode}-r{rank}-s{steps}"
    if dest.exists():
        raise ValueError("Refuse overwrite of existing cloud adapter")
    if not OUTPUT.is_dir():
        OUTPUT.mkdir(parents=True, exist_ok=True)

    model = Qwen2ForCausalLM.from_pretrained(
        str(BASE), local_files_only=True, trust_remote_code=False, token=False,
        use_safetensors=True, attn_implementation="sdpa", device_map={"": 0},
        torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
        ),
    )
    if not getattr(model, "is_loaded_in_4bit", False):
        raise RuntimeError("4-bit model loading was not verified")
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model, use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=rank, lora_dropout=0, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    ))
    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_count != (40_370_176 if rank == 16 else 80_740_352):
        raise RuntimeError(f"Unexpected trainable LoRA parameter count: {parameter_count}")
    if any("lora_" not in n for n, p in model.named_parameters() if p.requires_grad):
        raise RuntimeError("Base model must remain frozen")

    config = SFTConfig(
        output_dir=str(dest), per_device_train_batch_size=1,
        gradient_accumulation_steps=4, max_steps=steps, max_length=512,
        learning_rate=5e-5, bf16=True, fp16=False, optim="paged_adamw_8bit",
        gradient_checkpointing=True, packing=False, completion_only_loss=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        save_strategy="no", eval_strategy="no", report_to="none", logging_steps=10,
        dataloader_num_workers=0, seed=42,
    )
    trainer = SFTTrainer(
        model=model, args=config, processing_class=tokenizer,
        train_dataset=Dataset.from_list(encoded),
    )
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    training = trainer.train()
    model.save_pretrained(str(dest), safe_serialization=True)
    weight = dest / "adapter_model.safetensors"
    cfg = dest / "adapter_config.json"
    if not weight.is_file() or not cfg.is_file():
        raise RuntimeError("Training completed without safetensors output")
    record = {
        "status": "NERAL_CLOUD_EXPERIMENTAL_TRAINED_NOT_DEPLOYED",
        "base_model": MODEL_ID, "base_revision": BASE_REVISION,
        "source_split": "v5 train only: 1920 examples; val and test excluded",
        "train_sha256": _sha(DATA / "train.jsonl"),
        "rank": rank, "steps": steps, "max_length": max_length,
        "trainable_parameters": parameter_count,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "elapsed_seconds": round(time.monotonic() - started, 1),
        "train_loss": training.metrics.get("train_loss"),
        "adapter_sha256": _sha(weight),
        "adapter_config_sha256": _sha(cfg),
        "release_validated": False,
        "general_improvement_proven": False,
        "weights_promoted": False,
        "cloud_gpu": gpu["gpu"],
        "original_trainer_identical": False,
        "readme": "Separate research trainer: results need exact paired evaluation.",
    }
    with (dest / "neral-cloud-research.json").open("x", encoding="utf8") as f:
        json.dump(record, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps(record, sort_keys=True))
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("mode", choices=("preflight", "prepare", "smoke", "train"))
    p.add_argument("--rank", type=int, default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--max-length", type=int, default=512)
    p.add_argument("--approve", default="")
    args = p.parse_args(argv)

    if args.mode == "preflight":
        print(json.dumps(gpu_preflight(), sort_keys=True))
        return
    if args.mode == "prepare":
        ROOT.mkdir(parents=True, exist_ok=True)
        print(json.dumps(prepare_data(), sort_keys=True))
        info = prepare_model()
        print(json.dumps({"base_model": info["base_model"],
                          "base_revision": info["revision"],
                          "snapshot_sha256": info["sha256"],
                          "training_performed": False}, sort_keys=True))
        return
    expected = {"smoke": (16, 3), "train": (32, 480)}[args.mode]
    rank = args.rank if args.rank is not None else expected[0]
    steps = args.steps if args.steps is not None else expected[1]
    validate_experiment(args.mode, rank, steps, args.max_length)
    if args.approve != "RUN_NERAL_RESEARCH_ONLY":
        p.error("Training requires --approve RUN_NERAL_RESEARCH_ONLY")
    _train(args.mode, rank, steps, args.max_length)


if __name__ == "__main__":
    main()
