"""Prepare Arctus350 (own checkpoint) for one rented Vast.ai RTX 4090.

Default invocation is an offline read-only plan. Explicit --prepare only STAGES
a private working copy and a human-run shell command; it never starts GPUs,
rents machines, uploads weights, installs packages or starts training.

This file lives beside the EXISTING Arctus350 checkpoint-native trainer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

from frontier350.chat350_contract import EXPECTED_CONFIG, EXPECTED_TOKENIZER_SHA256

MIN_TOKENS = 2_000_000_000
SHA_PATTERN = re.compile(r"[0-9a-f]{64}")
DEFAULT_PRICE_USD = 0.366
DEFAULT_INCREMENT = 131_072


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or not SHA_PATTERN.fullmatch(value):
        raise ValueError(f"{label} requires an exact 64-digit lowercase SHA256")


def validate_host(gpus: list[dict], ram_gb: float):
    """Guardrail, not a performance prediction."""
    if len(gpus) != 1:
        raise ValueError("Vast trial requires exactly one GPU")
    gpu = gpus[0]
    if "4090" not in str(gpu.get("name", "")) or "rtx" not in str(gpu.get("name", "")).lower():
        raise ValueError("Expected one RTX 4090, not another GPU")
    if float(gpu.get("memory_mib", 0)) < 22_000:
        raise ValueError("RTX 4090 GPU memory too low for this profile")
    if not math.isfinite(ram_gb) or ram_gb < 28:
        raise ValueError("At least 28 GiB system RAM required for safe first trial")
    return {"gpu_name": gpu["name"], "gpu_vram_mib": gpu["memory_mib"],
            "system_ram_gib": ram_gb}


def probe_host():
    """Only reads the visible host hardware; never allocates a model."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise ValueError("nvidia-smi unavailable; refusing to prepare cloud training") from exc
    devices = []
    for line in out.stdout.splitlines():
        parts = [s.strip() for s in line.split(",", 1)]
        if len(parts) != 2:
            raise ValueError("Unexpected nvidia-smi GPU report")
        devices.append({"name": parts[0], "memory_mib": int(parts[1])})
    try:
        meminfo = Path("/proc/meminfo").read_text()
        kib = int(re.search(r"^MemTotal:\s+(\d+) kB", meminfo, re.M).group(1))
        ram_gb = kib / (1024 * 1024)
        # cgroup memory limit, when configured, can be lower than host RAM.
        for limit in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
            p = Path(limit)
            if not p.exists():
                continue
            s = p.read_text().strip()
            if s.isdigit() and int(s) < 1 << 60:
                ram_gb = min(ram_gb, int(s) / (1024 ** 3))
    except (OSError, AttributeError, ValueError) as exc:
        raise ValueError("Cannot validate RAM for RTX 4090 host") from exc
    return validate_host(devices, ram_gb)


def make_plan(metadata: dict, *, extra_tokens: int = DEFAULT_INCREMENT,
              price_usd: float = DEFAULT_PRICE_USD, cap_usd: float = 5.0) -> dict:
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata required; do not start from fresh weights")
    if metadata.get("config") != EXPECTED_CONFIG:
        raise ValueError("Arctus350 architecture mismatch")
    if metadata.get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("Original Arctus350 tokenizer mismatch")
    if not isinstance(metadata.get("model"), dict) or not isinstance(metadata.get("optimizer"), dict):
        raise ValueError("Resume checkpoint must have original model AND optimizer weights")
    if not metadata["model"] or not metadata["optimizer"]:
        raise ValueError("Missing model/optimizer state; fresh initialization forbidden")
    if type(metadata.get("processed_tokens")) is not int or metadata["processed_tokens"] < MIN_TOKENS:
        raise ValueError("Expected the verified 2B+ original native checkpoint")
    if type(metadata.get("target_tokens")) is not int or metadata["target_tokens"] <= metadata["processed_tokens"]:
        raise ValueError("Original checkpoint LR schedule already exhausted; do not silently reset target")
    if type(metadata.get("step")) is not int or metadata["step"] <= 0:
        raise ValueError("Invalid resume training step")
    if type(extra_tokens) is not int or not 8192 <= extra_tokens <= 1_000_000:
        raise ValueError("Pilot must use 8192..1,000,000 additional tokens")
    if not isinstance(price_usd, (int, float)) or not math.isfinite(price_usd) or not 0 < price_usd <= 0.40:
        raise ValueError("Hourly GPU price exceeds 0.40 USD guardrail")
    if not isinstance(cap_usd, (int, float)) or not math.isfinite(cap_usd) or not 0 < cap_usd <= 20:
        raise ValueError("Invalid planning budget (up to 20 USD)")
    begin = metadata["processed_tokens"]
    finish = min(begin + extra_tokens, metadata["target_tokens"])
    if finish <= begin:
        raise ValueError("No training tokens available in the unchanged schedule")
    return {
        "checkpoint_schema": metadata.get("checkpoint_schema"),
        "starting_step": metadata["step"],
        "start_processed_tokens": begin,
        "schedule_target_tokens": metadata["target_tokens"],
        "run_until_tokens": finish,
        "new_token_budget": finish - begin,
        "gpu_count": 1,
        "recommended_gpu": "NVIDIA RTX 4090 / 24 GB",
        "expected_hourly_gpu_usd": price_usd,
        "user_budget_usd": cap_usd,
        "nominal_gpu_hours_at_budget": round(cap_usd / price_usd, 2),
        "cost_hard_limit": False,  # Vast.ai billing must be controlled separately.
        "gpu_rented": False,
        "training_started": False,
        "code": "frontier350/train350_compiled.py (existing, compile-mode none)",
        "note": "The price is a snapshot, not a guaranteed rate or spending cap; "
                "disk/egress are additional. No time/throughput claim before a real test.",
    }


def validate_data_manifest(manifest_path: Path, *, seq: int = 512) -> dict:
    """Require original Frontier BPE .bin shards (not PR #205 JSONL data)."""
    path = Path(manifest_path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("A real BPE corpus manifest.json is required")
    if path.stat().st_size > 2 * 1024 * 1024:
        raise ValueError("BPE corpus manifest unexpectedly large")
    data = json.loads(path.read_text(encoding="utf8"))
    if (data.get("schema") != 2 or data.get("architecture") != "frontier350-v1"
            or data.get("benchmark_content_included") is not False
            or (data.get("tokenizer") or {}).get("tokenizer_sha256") != EXPECTED_TOKENIZER_SHA256):
        raise ValueError("Unverified/contaminated/non-BPE Frontier data corpus")
    result = {"manifest_sha256": sha256_file(path)}
    for name, field in (("train", "train_files"), ("valid", "valid_files")):
        entries = data.get(field)
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"{field} not declared in corpus manifest")
        expected = set()
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError("Invalid shard manifest row")
            rel = entry.get("path")
            if (not isinstance(rel, str) or not rel.endswith(".bin")
                    or Path(rel).name != rel or rel in expected):
                raise ValueError("Unsafe/duplicate shard path in manifest")
            expected.add(rel)
            shard = path.parent / name / rel
            if not shard.is_file() or shard.is_symlink():
                raise ValueError("Missing or symlinked BPE shard")
            _sha(entry.get("sha256"), "shard")
            if sha256_file(shard) != entry["sha256"]:
                raise ValueError("BPE shard SHA256 mismatch")
            if (shard.stat().st_size != int(entry.get("tokens", -1)) * 2
                    or shard.stat().st_size < 2 * (seq + 1)):
                raise ValueError("Bad uint16 token length for BPE shard")
        observed = {p.name for p in (path.parent / name).glob("*.bin")}
        if observed != expected:
            raise ValueError("Shard inventory differs from validated corpus manifest")
        result[f"{name}_shards"] = len(entries)
    return result


def _load_verified_checkpoint(path: Path) -> dict:
    # Explicit safe-global allowlist for the NumPy RNG primitive in legacy
    # checkpoints; refuses unrestricted pickle. See secure_checkpoint350.py.
    from frontier350.secure_checkpoint350 import load_checkpoint
    return load_checkpoint(path)


def prepare_run(*, source_checkpoint: Path, checkpoint_sha256: str,
                tokenizer: Path, tokenizer_sha256: str, data_manifest: Path,
                run_directory: Path, extra_tokens: int = DEFAULT_INCREMENT,
                price_usd: float = DEFAULT_PRICE_USD, cap_usd: float = 5.0,
                checkpoint_loader=_load_verified_checkpoint,
                hardware_check=probe_host, allow_test_tokenizer: bool = False):
    """Prepare only: output dir must not exist; never invoke a trainer or network."""
    checkpoint = Path(source_checkpoint)
    tokenizer = Path(tokenizer)
    output = Path(run_directory)
    if output.exists():
        raise FileExistsError("Refuse to overwrite an existing or completed trial directory")
    if not checkpoint.is_file() or checkpoint.is_symlink():
        raise ValueError("Trusted original checkpoint missing or symlinked")
    _sha(checkpoint_sha256, "checkpoint")
    if sha256_file(checkpoint) != checkpoint_sha256:
        raise ValueError("Original checkpoint SHA256 does not match verified receipt")
    if not tokenizer.is_file() or tokenizer.is_symlink():
        raise ValueError("Original locked tokenizer missing or symlinked")
    _sha(tokenizer_sha256, "tokenizer")
    if sha256_file(tokenizer) != tokenizer_sha256:
        raise ValueError("Tokenizer file SHA256 mismatch")
    if not allow_test_tokenizer and tokenizer_sha256 != EXPECTED_TOKENIZER_SHA256:
        raise ValueError("Unexpected tokenizer: original 16K Arctus BPE required")
    corpus = validate_data_manifest(data_manifest)
    host = hardware_check()
    meta = checkpoint_loader(checkpoint)
    plan = make_plan(meta, extra_tokens=extra_tokens, price_usd=price_usd, cap_usd=cap_usd)
    if any(name not in meta or meta[name] is None for name in ("torch_rng", "python_rng", "numpy_rng")):
        raise ValueError("Source checkpoint lacks required RNG state")
    repo = Path(__file__).resolve().parents[1]
    trainer = repo / "frontier350" / "train350_compiled.py"
    if not trainer.is_file():
        raise ValueError("Original resume trainer not found")
    # Never change the original checkpoint; a standalone working copy can be
    # overwritten by the native trainer and later backed up separately.
    output.mkdir(parents=True, exist_ok=False)
    copy_path = output / "resume.pt"
    try:
        with checkpoint.open("rb") as src, copy_path.open("xb") as dest:
            shutil.copyfileobj(src, dest, length=4 * 1024 * 1024)
        if sha256_file(copy_path) != checkpoint_sha256:
            raise ValueError("Working checkpoint copy corrupt")
        command = [
            "python", "-u", str(trainer),
            "--train-data", str(Path(data_manifest).parent / "train"),
            "--valid-data", str(Path(data_manifest).parent / "valid"),
            "--out", str(output),
            "--resume", str(copy_path),
            "--target-tokens", str(plan["schedule_target_tokens"]),
            "--stop-after-tokens", str(plan["run_until_tokens"]),
            "--tokenizer-sha256", EXPECTED_TOKENIZER_SHA256,
            "--seq", "512", "--microbatch", "2", "--grad-accum", "8",
            "--compile-mode", "none", "--resume-only",
            "--log-every", "1", "--eval-every", "4",
            "--save-every", "4", "--eval-batches", "2",
        ]
        script = "#!/usr/bin/env bash\nset -euo pipefail\n" + (
            "export TORCH_FORCE_WEIGHTS_ONLY_LOAD=1\n"
            "echo 'STARTING EXPLICIT USER-APPROVED ARCTUS350 GPU TRIAL; "
            "VAST.AI BILLING IS NOT AUTOMATICALLY CAPPED'\n"
            + shlex.join(command) + "\n"
        )
        (output / "start-vast4090.sh").write_text(script, encoding="utf8")
        report = {
            **plan, **corpus,
            "checkpoint_original_sha256": checkpoint_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "working_checkpoint_file": str(copy_path),
            "run_command_file": str(output / "start-vast4090.sh"),
            "training_started": False,
            "gpu_rented": False,  # script never calls Vast API
            "host_check": host,
            "checkpoint_copy_is_not_source": True,
            "privacy": "No model uploaded or exposed by this script.",
            "warning": "Never put PAT, SSH private key or billing credentials on untrusted Vast hosts.",
        }
        (output / "prepare-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf8")
        return report
    except BaseException:
        # Do not silently publish a partial or unsafe run directory.
        shutil.rmtree(output)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true", help="explicit staging only; NO training")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--extra-tokens", type=int, default=DEFAULT_INCREMENT)
    parser.add_argument("--gpu-hourly-usd", type=float, default=DEFAULT_PRICE_USD)
    parser.add_argument("--budget-usd", type=float, default=5.0)
    a = parser.parse_args(argv)
    if not a.prepare:
        print(json.dumps({
            "state": "READ_ONLY_PLAN",
            "gpu": "1x RTX 4090, 24GB VRAM",
            "offer_price_usd_per_hour": DEFAULT_PRICE_USD,
            "intended_additional_tokens": DEFAULT_INCREMENT,
            "checkpoint_required": "verified native Arctus350 full optimizer resume.pt",
            "tokenizer_sha256": EXPECTED_TOKENIZER_SHA256,
            "dataset_required": "original BPE train+valid .bin shards with schema-2 manifest",
            "requires": "explicit --prepare plus verified files ON the rented host",
            "training_started": False, "rental_started": False,
            "cost_cap_automated": False,
        }, indent=2))
        return 0
    if not all([a.checkpoint, a.checkpoint_sha256, a.tokenizer, a.data_manifest, a.out]):
        parser.error("--prepare requires --checkpoint, --checkpoint-sha256, --tokenizer, --data-manifest, --out")
    report = prepare_run(
        source_checkpoint=a.checkpoint, checkpoint_sha256=a.checkpoint_sha256,
        tokenizer=a.tokenizer, tokenizer_sha256=EXPECTED_TOKENIZER_SHA256,
        data_manifest=a.data_manifest, run_directory=a.out,
        extra_tokens=a.extra_tokens, price_usd=a.gpu_hourly_usd,
        cap_usd=a.budget_usd)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("NOT TRAINING. To START after separately confirming rate and backups: "
          f"bash {shlex.quote(str(a.out/'start-vast4090.sh'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
