"""Read-only admission preflight for a future *local* Neral RTX 4070 experiment.

No torch.load, checkpoint writes, training, cloud compute, or inference. All
expected hashes must be pinned independently of the files being inspected.
A pass is ONLY environmental and byte-level readiness, never model provenance,
quality, exact resume or 44k tokens/s evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess

SHA = re.compile(r"^[0-9a-f]{64}$")
MIN_FREE_DISK_BYTES = 16 * 1024 ** 3
MIN_FREE_GPU_MIB = 9000
MAX_GPU_TEMP_C = 78


def require_sha(value: str, label: str) -> str:
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError(f"{label}: independently pinned lowercase SHA-256 required")
    return value


def hash_file(path: Path) -> str:
    """Hash a regular, non-symlink file and catch ordinary concurrent rewrites."""
    if path.is_symlink():
        raise ValueError("symlink input rejected")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError("input must be a nonempty regular file")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(fd), "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
            after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns
        ):
            raise ValueError("input changed while hashing")
        return digest.hexdigest()
    finally:
        os.close(fd)


def json_manifest(path: Path) -> dict:
    def pairs(entries):
        output = {}
        for key, value in entries:
            if key in output:
                raise ValueError("duplicate JSON manifest field")
            output[key] = value
        return output
    result = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                        parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid JSON constant")))
    if not isinstance(result, dict):
        raise ValueError("manifest must be an object")
    return result


def verify_shards(directory: Path, records: object) -> int:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("training shards directory missing or symlinked")
    if not isinstance(records, list) or not records:
        raise ValueError("nonempty recorded shards required")
    seen = set()
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            raise ValueError("invalid shard record")
        name = record["path"]
        if name in (".", "..") or Path(name).name != name or not name or "\\" in name or "/" in name:
            raise ValueError("shard path must be a basename")
        if not name.endswith(".bin"):
            raise ValueError("training shard must be a *.bin file")
        if name in seen:
            raise ValueError("duplicate shard")
        seen.add(name)
        expected = require_sha(record.get("sha256"), "shard")
        if not hmac.compare_digest(hash_file(directory / name), expected):
            raise ValueError("training shard SHA-256 mismatch")
    # PersistentMMapPool opens every *.bin file, not merely manifest-listed files.
    # Detect extras before allocating GPU/model memory (also checks missing files).
    discovered = {path.name for path in directory.glob("*.bin")}
    if discovered != seen:
        raise ValueError("unlisted or missing *.bin training shards")
    return len(seen)


def query_gpu() -> dict:
    command = ["nvidia-smi", "--query-gpu=name,memory.free,temperature.gpu", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, timeout=10, check=True)
    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise ValueError("exactly one GPU required")
    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) != 3 or "RTX 4070" not in parts[0]:
        raise ValueError("unexpected GPU identity")
    free, temp = int(parts[1]), int(parts[2])
    apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, timeout=10, check=True)
    running = [line.strip() for line in apps.stdout.splitlines() if line.strip()]
    if running == ["No running processes found"]:
        running = []
    if free < MIN_FREE_GPU_MIB or temp > MAX_GPU_TEMP_C or running:
        raise ValueError(
            f"GPU not ready: free_mib={free} (min {MIN_FREE_GPU_MIB}); "
            f"temperature_c={temp} (max {MAX_GPU_TEMP_C}); "
            f"compute_processes={len(running)} (required 0)"
        )
    return {"gpu": "RTX 4070", "free_mib": free, "temperature_c": temp,
            "compute_processes": 0}


def verify(anchor: Path, anchor_sha256: str, main_data: Path, replay: Path,
           tokenizer_sha256: str, arm_a: Path, arm_b: Path, *, gpu_probe=query_gpu) -> dict:
    expected_anchor = require_sha(anchor_sha256, "safe anchor")
    expected_tokenizer = require_sha(tokenizer_sha256, "tokenizer")
    # On a busy shared GPU, reject before rehashing multi-GB local assets.
    gpu = gpu_probe()
    if (gpu.get("gpu") != "RTX 4070" or gpu.get("free_mib", 0) < MIN_FREE_GPU_MIB
            or gpu.get("temperature_c", 999) > MAX_GPU_TEMP_C
            or gpu.get("compute_processes") != 0):
        raise ValueError("GPU readiness verification failed: " + json.dumps(gpu, sort_keys=True))
    if not hmac.compare_digest(hash_file(anchor), expected_anchor):
        raise ValueError("safe anchor bytes differ from independently pinned SHA-256")
    main = json_manifest(main_data.parent / "manifest.json")
    if main.get("benchmark_content_included") is not False:
        raise ValueError("training and benchmark separation not attested")
    if not isinstance(main.get("tokenizer"), dict) or main["tokenizer"].get("tokenizer_sha256") != expected_tokenizer:
        raise ValueError("main tokenizer identity differs")
    main_files = verify_shards(main_data, main.get("train_files"))
    replay_manifest = json_manifest(replay / "manifest.json")
    if replay_manifest.get("tokenizer_sha256") != expected_tokenizer:
        raise ValueError("replay tokenizer identity differs")
    domains = replay_manifest.get("domains")
    if not isinstance(domains, dict) or set(domains) != {"education", "general", "math_reasoning"}:
        raise ValueError("expected exact replay domains")
    replay_files = sum(verify_shards(replay / domain, domains[domain].get("files"))
                       for domain in sorted(domains))
    outputs = [arm_a.absolute(), arm_b.absolute()]
    if outputs[0] == outputs[1] or outputs[0].resolve() == outputs[1].resolve():
        raise ValueError("A/B experiments require two distinct output directories")
    input_roots = [anchor.parent.resolve(), main_data.resolve(), replay.resolve()]
    for output in outputs:
        if output.exists() or output.is_symlink():
            raise ValueError("experiment output must not already exist")
        if not output.parent.is_dir() or output.parent.is_symlink():
            raise ValueError("output parent must exist and be a real directory")
        resolved = output.resolve()
        if any(resolved == root or resolved in root.parents or root in resolved.parents for root in input_roots):
            raise ValueError("experiment output overlaps input/anchor storage")
        if shutil.disk_usage(output.parent).free < MIN_FREE_DISK_BYTES:
            raise ValueError("insufficient disk for isolated A/B artifacts")
    return {"schema": 1, "status": "preflight_passed_not_training_evidence",
            "anchor_sha256": expected_anchor, "tokenizer_sha256": expected_tokenizer,
            "main_shards_verified": main_files, "replay_shards_verified": replay_files,
            "gpu": gpu, "outputs_created": False, "training_started": False,
            "real_resume_verified": False, "real_tokens_per_second_verified": False,
            "eligible_for_production": False}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("anchor", "anchor-sha256", "main-data", "replay", "tokenizer-sha256", "arm-a", "arm-b"):
        parser.add_argument("--" + flag, required=True)
    args = parser.parse_args()
    try:
        report = verify(Path(args.anchor), args.anchor_sha256, Path(args.main_data), Path(args.replay),
                        args.tokenizer_sha256, Path(args.arm_a), Path(args.arm_b))
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        print(json.dumps({"schema": 1, "status": "blocked", "reason": str(error),
                          "training_started": False, "eligible_for_production": False}))
        return 2
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
