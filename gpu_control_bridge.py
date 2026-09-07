"""Restricted self-hosted RTX control bridge for Neral.

The bridge intentionally exposes only a small allow-list of read-only model actions.
It never accepts shell commands, arbitrary paths, training targets, or weight-changing
operations. Private checkpoints/corpora remain below NERAL_ASSETS_ROOT; only reports
are written to the requested report directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ALLOWED_ACTIONS = {"gpu_status", "inspect_b2", "benchmark_b2", "run_tests"}
REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_request(path: Path) -> dict:
    raw = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(raw, dict):
        raise ValueError("request must be a JSON object")
    allowed_keys = {"version", "request_id", "action"}
    extra = sorted(set(raw) - allowed_keys)
    if extra:
        raise ValueError(f"unsupported request fields: {extra}")
    if raw.get("version") != 1:
        raise ValueError("request version must be 1")
    request_id = raw.get("request_id")
    if not isinstance(request_id, str) or not REQUEST_ID_RE.fullmatch(request_id):
        raise ValueError("invalid request_id")
    action = raw.get("action")
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"unsupported action: {action!r}")
    return {"version": 1, "request_id": request_id, "action": action}


def inspect_b2(assets: Path) -> dict:
    run = assets / "runs" / "neral50m_b2_100m"
    summary_path = run / "summary.json"
    out: dict = {
        "run_dir": str(run),
        "summary_exists": summary_path.is_file(),
        "weights_modified": False,
    }
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text(encoding="utf8"))
        keep = (
            "stop", "complete", "stage_b2_tokens", "this_segment_tokens",
            "source_cumulative_tokens", "cumulative_tokens", "target_cumulative_tokens",
            "best_b2_validation", "baseline_old_validation", "elapsed_seconds",
            "optimizer_seconds", "optimizer_tokens_per_second", "peak_allocated",
            "peak_reserved", "best_sha256", "latest_sha256",
            "instruction_tuning_performed", "train_block_replay",
        )
        out["summary"] = {k: summary.get(k) for k in keep if k in summary}
    for name in ("best_validation.pt", "latest.pt"):
        p = run / name
        item = {"exists": p.is_file()}
        if p.is_file():
            item.update({"bytes": p.stat().st_size, "sha256": sha256_file(p)})
        out[name] = item
    return out


def gpu_status() -> dict:
    import torch

    cuda = torch.cuda.is_available()
    status: dict = {
        "torch": str(torch.__version__),
        "cuda_runtime": str(torch.version.cuda),
        "cuda_available": bool(cuda),
        "bf16_supported": bool(cuda and torch.cuda.is_bf16_supported()),
        "weights_modified": False,
    }
    if cuda:
        props = torch.cuda.get_device_properties(0)
        status.update({
            "gpu": torch.cuda.get_device_name(0),
            "vram_bytes": int(props.total_memory),
            "allocated_bytes": int(torch.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch.cuda.memory_reserved(0)),
        })
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,memory.used,memory.free,temperature.gpu,power.draw,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    status["nvidia_smi_returncode"] = proc.returncode
    status["nvidia_smi"] = proc.stdout.strip()
    if proc.stderr.strip():
        status["nvidia_smi_stderr"] = proc.stderr.strip()
    return status


def run_logged(args: list[str], cwd: Path, log_path: Path) -> int:
    with log_path.open("w", encoding="utf8") as log:
        proc = subprocess.run(args, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, text=True)
    return int(proc.returncode)


def execute(request: dict, assets: Path, reports_root: Path) -> Path:
    if not assets.is_dir():
        raise FileNotFoundError(f"assets root does not exist: {assets}")
    run_id = os.environ.get("GITHUB_RUN_ID") or f"local-{int(time.time())}"
    out = reports_root / request["request_id"] / str(run_id)
    out.mkdir(parents=True, exist_ok=False)
    (out / "request.json").write_text(json.dumps(request, indent=2), encoding="utf8")

    started = time.time()
    action = request["action"]
    result: dict = {
        "version": 1,
        "request_id": request["request_id"],
        "action": action,
        "weights_modified": False,
        "started_unix": started,
    }

    try:
        if action == "gpu_status":
            result["gpu"] = gpu_status()
        elif action == "inspect_b2":
            result["b2"] = inspect_b2(assets)
        elif action == "benchmark_b2":
            benchmark_dir = out / "benchmark"
            code = run_logged(
                [sys.executable, "-X", "utf8", "finalize_b2_benchmark50.py",
                 "--assets", str(assets), "--reports", str(benchmark_dir)],
                Path(__file__).resolve().parent,
                out / "benchmark_driver.log",
            )
            result["benchmark_returncode"] = code
            if code != 0:
                raise RuntimeError(f"B2 benchmark handoff failed with exit code {code}")
        elif action == "run_tests":
            code = run_logged(
                [sys.executable, "-m", "pytest", "-q", "--junitxml", str(out / "pytest.xml")],
                Path(__file__).resolve().parent,
                out / "pytest.log",
            )
            result["pytest_returncode"] = code
            if code != 0:
                raise RuntimeError(f"pytest failed with exit code {code}")
        else:  # load_request already fail-closes; keep an execution guard too.
            raise AssertionError(action)
        result["status"] = "success"
        return out
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        result["finished_unix"] = time.time()
        result["elapsed_seconds"] = result["finished_unix"] - started
        (out / "result.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
        print(json.dumps(result, ensure_ascii=False), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--request", required=True)
    ap.add_argument("--assets", required=True)
    ap.add_argument("--reports", default="gpu_control_reports")
    args = ap.parse_args()
    request = load_request(Path(args.request))
    execute(request, Path(args.assets).resolve(), Path(args.reports).resolve())


if __name__ == "__main__":
    main()
