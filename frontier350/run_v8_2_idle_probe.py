from __future__ import annotations

"""Bounded read-only GPU probe AFTER the active training job has released its runner.

Never edits a checkpoint, training state, optimizer or SAFE gate. Uses identical
checkpoint/tokenizer/validation inputs for independent and single-load evaluators.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from recovery_identity import file_sha
from verify_health_parity_v8_2 import compare

ROOT = Path(__file__).resolve().parent


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def call(name: str, script: str, args: list[str], timings: dict[str, float]) -> None:
    started = time.perf_counter()
    print(json.dumps({"event": "probe_start", "name": name, "script": script}), flush=True)
    subprocess.run([sys.executable, "-u", str(ROOT / script), *args], check=True)
    timings[name] = time.perf_counter() - started
    print(json.dumps({"event": "probe_end", "name": name, "seconds": timings[name]}), flush=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--min-tokens", type=int, default=2_000_000_000)
    args = p.parse_args()
    assets, report = args.assets.resolve(), args.report.resolve()
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("probe report directory must be empty")
    report.mkdir(parents=True, exist_ok=True)
    state_file = assets / "frontier350_scale_v5_state.json"
    state_sha = file_sha(state_file)
    state = read(state_file)
    continuity = state.get("continuity") or {}
    checkpoint = Path(str(continuity.get("path", "")))
    claimed_sha = str(continuity.get("sha256", ""))
    tokens = int(continuity.get("tokens", -1))
    if tokens < args.min_tokens:
        raise RuntimeError(f"continuity below minimum: {tokens} < {args.min_tokens}")
    if not checkpoint.is_file() or file_sha(checkpoint) != claimed_sha:
        raise RuntimeError("checkpoint missing or does not match pinned continuity SHA256")
    tokenizer = assets / "frontier350_assets/tokenizer.json"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2" / "valid"
    if not tokenizer.is_file() or not valid.is_dir():
        raise FileNotFoundError("tokenizer or validation shards missing")
    timings: dict[str, float] = {}
    rotations: list[dict] = []
    for rotation in (0, 1):
        prefix = report / f"rotation_{rotation}"
        prefix.mkdir()
        legacy_smoke = prefix / "legacy_smoke.json"
        legacy_validation = prefix / "legacy_validation.json"
        combined_smoke = prefix / "combined_smoke.json"
        combined_validation = prefix / "combined_validation.json"
        common = ["--checkpoint", str(checkpoint)]
        call(f"legacy_smoke_{rotation}", "eval_recovery_smoke_v3.py",
             [*common, "--tokenizer", str(tokenizer), "--out", str(legacy_smoke),
              "--expected-tokens", str(tokens), "--rotation", str(rotation)], timings)
        call(f"legacy_validation_{rotation}", "eval_validation_loss350.py",
             [*common, "--valid-data", str(valid), "--out", str(legacy_validation),
              "--seq", "512", "--batch", "4", "--batches", "4", "--seed", "351337"], timings)
        call(f"combined_{rotation}", "eval_health_v8_2_combined.py",
             [*common, "--tokenizer", str(tokenizer), "--valid-data", str(valid),
              "--smoke-out", str(combined_smoke), "--validation-out", str(combined_validation),
              "--expected-tokens", str(tokens), "--rotation", str(rotation),
              "--seq", "512", "--batch", "4", "--batches", "4", "--seed", "351337"], timings)
        parity = compare(read(legacy_smoke), read(combined_smoke),
                         read(legacy_validation), read(combined_validation))
        (prefix / "parity.json").write_text(json.dumps(parity, indent=2), encoding="utf-8")
        rotations.append({"rotation": rotation, **parity})
        if not parity["parity_pass"]:
            raise RuntimeError(f"read-only numerical/report parity failed for rotation {rotation}")
        if file_sha(checkpoint) != claimed_sha or file_sha(state_file) != state_sha:
            raise RuntimeError("training checkpoint/state changed during parity probe")

    readiness_dir = report / "nis2"
    call("nis2_v2_140", "run_v8_nis2_readiness.py",
         ["--assets", str(assets), "--report", str(readiness_dir),
          "--expected-tokens", str(tokens)], timings)
    if file_sha(checkpoint) != claimed_sha or file_sha(state_file) != state_sha:
        raise RuntimeError("training checkpoint/state changed during read-only probe")
    nis = read(readiness_dir / "summary.json")
    if not nis.get("read_only_verified") or nis.get("checkpoint_sha256") != claimed_sha:
        raise RuntimeError("readiness checkpoint identity not verified")
    result = {
        "schema": 1, "checkpoint_sha256": claimed_sha, "checkpoint_tokens": tokens,
        "state_sha256_unchanged": True, "read_only_verified": True,
        "rotations": rotations, "timings_seconds": timings,
        "legacy_eval_total_seconds": sum(timings[f"legacy_smoke_{r}"] + timings[f"legacy_validation_{r}"] for r in (0, 1)),
        "combined_eval_total_seconds": sum(timings[f"combined_{r}"] for r in (0, 1)),
        "nis2_summary": nis,
        "combined_production_enabled": False, "stable_40k_verified": False,
        "warning": "Parity/speed microprobe and NIS2 readiness do not establish 40k accepted-tokens/s or full chatbot quality.",
    }
    (report / "summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
