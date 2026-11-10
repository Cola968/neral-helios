"""GPU preflight for NERAL HIGH12: one 45-second Qwen request, never 12 tasks.

Manual-only, localhost-only, no training, no TIR, no untrusted model-code
execution. Only sanitized Ollama timings and VRAM residency go to GitHub.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
import subprocess
import threading

MODEL = "qwen3.5:9b"
LOW_MODEL = "qwen3:8b"
PROFILE = "measured_nonthinking_2048"
AUTHORIZATION = "NERAL_HIGH12_PROBE_GPU_MANUAL_ONLY"
ENDPOINT = "http://127.0.0.1:11434"
SHA = re.compile(r"[0-9a-f]{40}\Z")
OPTIONS = {
    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    "min_p": 0.0, "presence_penalty": 1.5, "repeat_penalty": 1.0,
    "num_predict": 2048, "num_ctx": 4096,
}
PROBE_SEED = 20261110


class PreflightError(RuntimeError):
    """Safe status: never include model text, local paths, or HTTP bodies."""


def measured_chat_payload(messages: list[dict]) -> dict:
    """Exact measured profile from OllamaLocalBackend._respond, raw chat arm."""
    return {
        "model": MODEL, "messages": messages, "think": False,
        "stream": False, "options": dict(OPTIONS), "keep_alive": "5m",
    }


def model_fully_on_gpu(model: dict) -> bool:
    size = model.get("size")
    vram = model.get("size_vram")
    return (type(size) is int and type(vram) is int
            and size > 0 and vram >= size)


def running_model_summary(packet: dict, *, allow_offload: bool = False) -> dict:
    """Validate Ollama /api/ps. Never evict any unexpected local model."""
    if type(packet) is not dict or type(packet.get("models")) is not list:
        raise PreflightError("invalid_ps_response")
    rows = packet["models"]
    if len(rows) > 15:
        raise PreflightError("unexpected_loaded_model_count")
    result = {}
    for row in rows:
        if type(row) is not dict:
            raise PreflightError("invalid_ps_model")
        name, size, vram = row.get("name"), row.get("size"), row.get("size_vram")
        if (name not in (MODEL, LOW_MODEL) or name in result
                or type(size) is not int or type(vram) is not int
                or size <= 0 or vram < 0):
            raise PreflightError("unexpected_model_or_size_in_ps")
        result[name] = {"size": size, "size_vram": vram}
        if name == MODEL and not allow_offload and not model_fully_on_gpu(row):
            raise PreflightError("high_model_cpu_offload_detected")
    return result


def sanitized_timings(packet: dict) -> dict:
    """Extract durations in ns without ever returning response content."""
    if (type(packet) is not dict or packet.get("done") is not True
            or packet.get("done_reason") != "stop"
            or type(packet.get("message")) is not dict
            or not isinstance(packet["message"].get("content"), str)
            or not packet["message"]["content"].strip()):
        raise PreflightError("no_complete_final_response")
    keys = ("load_duration", "prompt_eval_duration", "eval_duration",
            "total_duration")
    durations = {}
    for key in keys:
        value = packet.get(key)
        if type(value) is not int or not 0 <= value < 10**14:
            raise PreflightError("missing_or_invalid_ollama_timings")
        durations[key] = value
    eval_count = packet.get("eval_count")
    prompt_count = packet.get("prompt_eval_count")
    if (type(eval_count) is not int or type(prompt_count) is not int
            or eval_count < 0 or prompt_count < 0):
        raise PreflightError("invalid_ollama_token_counts")
    seconds = {k: round(v / 10**9, 3) for k, v in durations.items()}
    return {
        "load_seconds": seconds["load_duration"],
        "prompt_eval_seconds": seconds["prompt_eval_duration"],
        "eval_seconds": seconds["eval_duration"],
        "total_ollama_seconds": seconds["total_duration"],
        "eval_tokens": eval_count,
        "prompt_tokens": prompt_count,
        "tokens_per_second": round(
            eval_count / (durations["eval_duration"] / 1e9), 3)
            if durations["eval_duration"] else 0.0,
        "done_reason": "stop",
        "content_chars": len(packet["message"]["content"]),
    }



def parse_gpu_telemetry(line: str) -> tuple[int, int] | None:
    """Parse only numeric utilization% and used MiB; no process identifiers."""
    if type(line) is not str or re.fullmatch(r"\s*\d{1,3}\s*,\s*\d{1,6}\s*", line) is None:
        return None
    numbers = [int(part.strip()) for part in line.split(",")]
    if numbers[0] > 100 or numbers[1] > 131072:
        return None
    return numbers[0], numbers[1]


def summarize_gpu_samples(samples: list[tuple[int, int]]) -> dict:
    """Display activity while nonstreaming Ollama is awaiting a response."""
    return {
        "samples_count": len(samples),
        "gpu_util_samples_percent": [row[0] for row in samples],
        "gpu_util_peak_percent": max((row[0] for row in samples), default=None),
        "vram_used_peak_mib": max((row[1] for row in samples), default=None),
        "interpretation": "GPU activity only; cannot prove queue vs generation",
    }


def _watch_gpu_activity(stop: threading.Event, samples: list[tuple[int, int]]) -> None:
    """Nine best-effort local telemetry polls, no web/network or raw logs."""
    while not stop.is_set() and len(samples) < 9:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                 "--format=csv,noheader,nounits"],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, timeout=3, check=False,
            )
            if result.returncode == 0:
                sample = parse_gpu_telemetry(result.stdout.strip())
                if sample is not None:
                    samples.append(sample)
        except (OSError, subprocess.TimeoutExpired):
            pass
        stop.wait(5)


def parse_nvidia_smi_snapshot(stdout: str) -> dict:
    """Strict scalar RTX 4070 inventory; never upload arbitrary driver output."""
    if type(stdout) is not str or len(stdout) > 512 or "\n" in stdout.strip():
        raise PreflightError("invalid_nvidia_inventory")
    fields = [field.strip() for field in stdout.strip().split(",")]
    if len(fields) != 5:
        raise PreflightError("invalid_nvidia_inventory")
    name, total, used, free, driver = fields
    if (re.fullmatch(r"[A-Za-z0-9 .+\-]{1,72}", name) is None
            or "RTX 4070" not in name
            or re.fullmatch(r"\d{2,4}\.\d{2,3}", driver) is None
            or any(re.fullmatch(r"\d{1,6}", v) is None
                   for v in (total, used, free))):
        raise PreflightError("gpu_or_driver_not_expected")
    total, used, free = map(int, (total, used, free))
    if not 0 < total <= 131072 or not 0 <= used <= total or not 0 <= free <= total:
        raise PreflightError("invalid_nvidia_memory")
    return {
        "gpu_name": name,
        "memory_total_mib": total,
        "memory_used_mib": used,
        "memory_free_mib": free,
        "driver_version": driver,
    }


def nvidia_gpu_inventory() -> dict:
    """Read one NVIDIA adapter's name and memory, without user or PID data."""
    try:
        run = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=False, timeout=8,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreflightError("nvidia_smi_unavailable") from exc
    if run.returncode != 0:
        raise PreflightError("nvidia_smi_unavailable")
    return parse_nvidia_smi_snapshot(run.stdout)


def parse_ollama_process_inventory(output: str) -> dict:
    """Validate ONLY process and listener counts: no paths, PIDs, usernames."""
    if type(output) is not str or len(output) > 512:
        raise PreflightError("invalid_ollama_process_inventory")
    try:
        row = json.loads(output)
    except (TypeError, ValueError) as exc:
        raise PreflightError("invalid_ollama_process_inventory") from exc
    fields = {
        "ollama_process_count", "llama_worker_count",
        "listener_count", "unique_listener_owners",
    }
    if type(row) is not dict or set(row) != fields:
        raise PreflightError("invalid_ollama_process_inventory")
    if any(type(row[field]) is not int or not 0 <= row[field] <= 50
           for field in fields):
        raise PreflightError("invalid_ollama_process_inventory")
    if row["unique_listener_owners"] > row["listener_count"]:
        raise PreflightError("invalid_ollama_process_inventory")
    if row["unique_listener_owners"] > 1:
        raise PreflightError("multiple_distinct_ollama_listener_owners")
    return {key: row[key] for key in sorted(fields)}


def windows_ollama_process_inventory() -> dict:
    """Read-only PowerShell/CIM check. Never launch or kill Ollama processes."""
    if os.name != "nt":
        return {"status": "not_windows"}
    command = (
        "$ErrorActionPreference='Stop';"
        "$p=@(Get-CimInstance Win32_Process -Filter "
        '"Name = \'ollama.exe\' OR Name = \'ollama app.exe\' '
        'OR Name = \'ollama_llama_server.exe\'");'
        "$l=@(Get-NetTCPConnection -LocalPort 11434 -State Listen "
        "-ErrorAction SilentlyContinue);"
        "$u=@($l | Select-Object -ExpandProperty OwningProcess -Unique);"
        "[pscustomobject]@{"
        "ollama_process_count=@($p | Where-Object { "
        "$_.Name -eq 'ollama.exe' -or $_.Name -eq 'ollama app.exe' }).Count;"
        "llama_worker_count=@($p | Where-Object { "
        "$_.Name -eq 'ollama_llama_server.exe' }).Count;"
        "listener_count=@($l).Count;"
        "unique_listener_owners=@($u).Count"
        "} | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoLogo", "-NoProfile", "-NonInteractive",
             "-Command", command],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            text=True, check=False, timeout=9,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {"status": "inventory_unavailable"}
    if result.returncode != 0:
        return {"status": "inventory_unavailable"}
    return parse_ollama_process_inventory(result.stdout.strip())



def select_probe_case() -> dict:
    """One reproducible but unseen instance from the SAME HIGH12 generator."""
    from scripts.neral_high12_manual import select_cases
    return next(c for c in select_cases(PROBE_SEED) if c["domain"] == "math")


def _ollama_json(path: str, *, data: dict | None = None,
                 timeout: float = 10.0) -> dict:
    """Allow-listed localhost Ollama requests; no proxy and no redirects."""
    if path not in ("/api/chat", "/api/ps"):
        raise PreflightError("unsupported_local_endpoint")
    if (data is None) != (path == "/api/ps"):
        raise PreflightError("unexpected_request_method")
    from neral_agent_autonomy import local_only_opener
    payload = (None if data is None else
               json.dumps(data, ensure_ascii=False).encode("utf8"))
    req = urllib.request.Request(
        ENDPOINT + path, data=payload,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    with local_only_opener().open(req, timeout=timeout) as response:
        raw = response.read(131073)
    if len(raw) > 131072:
        raise PreflightError("oversized_ollama_response")
    packet = json.loads(raw)
    if type(packet) is not dict:
        raise PreflightError("invalid_ollama_json")
    return packet


def _loaded() -> dict:
    return running_model_summary(_ollama_json("/api/ps", timeout=8),
                                 allow_offload=True)


def model_unload_payload(name: str) -> dict:
    """Unloads only approved model tags. Never stops or restarts the service."""
    if name not in (MODEL, LOW_MODEL):
        raise PreflightError("unapproved_model_eviction")
    return {
        "model": name, "messages": [], "stream": False, "keep_alive": 0,
    }


def _unload(name: str) -> None:
    reply = _ollama_json("/api/chat", data=model_unload_payload(name), timeout=15)
    if reply.get("done") is not True:
        raise PreflightError("model_eviction_not_confirmed")


def _preload() -> None:
    payload = measured_chat_payload([])
    packet = _ollama_json("/api/chat", data=payload, timeout=30)
    if packet.get("done") is not True or packet.get("done_reason") not in (
            "load", "stop", None):
        raise PreflightError("high_model_preload_failed")


def _gpu_summary(row: dict) -> dict:
    return {
        "size_bytes": row["size"],
        "size_vram_bytes": row["size_vram"],
        "gpu_percent_of_reported_size": round(
            min(100, 100 * row["size_vram"] / row["size"]), 3),
        "fully_gpu_per_ps": model_fully_on_gpu(row),
    }



def require_high_gpu_residency(ps: dict, report: dict) -> None:
    """Preserve the exact /api/ps VRAM measurements even when failing closed."""
    loaded = running_model_summary(ps, allow_offload=True)
    report["observed_loaded_models_after_preload"] = sorted(loaded)
    if MODEL in loaded:
        report["gpu_residency_before"] = _gpu_summary(loaded[MODEL])
    if MODEL not in loaded or LOW_MODEL in loaded:
        raise PreflightError("expected_only_high_model_resident")
    if not model_fully_on_gpu(loaded[MODEL]):
        raise PreflightError("high_model_cpu_offload_detected")


def release_high_model_after_probe(report: dict) -> None:
    """After verified inference, free only the known HIGH model for HIGH12.

    The probe and paired benchmark use the same exclusive GPU but the probe
    keeps Qwen resident for five minutes. Releasing it is not a daemon
    restart and must never evict unexpected models or bypass verification.
    """
    if (type(report) is not dict
            or report.get("gpu_residency_after", {}).get("fully_gpu_per_ps") is not True):
        raise PreflightError("probe_gpu_verification_required_before_release")
    _unload(MODEL)
    if _loaded():
        raise PreflightError("probe_left_model_resident")
    report["gpu_released_for_followup"] = True


def run_probe(*, authorize: str, target_sha: str, output: Path,
              restart_authorization: str = "") -> bool:
    if (authorize != AUTHORIZATION or type(target_sha) is not str
            or SHA.fullmatch(target_sha) is None):
        raise PreflightError("invalid_manual_authorization_or_sha")
    if os.environ.get("NERAL_EXPERIMENTAL_TIR") != "0":
        raise PreflightError("tir_must_be_disabled")
    output = Path(output)
    if (output.exists() or output.is_symlink() or output.suffix != ".json"
            or not output.parent.is_dir() or output.parent.is_symlink()):
        raise PreflightError("invalid_output_destination")

    report = {
        "schema": 1, "name": "NERAL_HIGH12_SINGLE_GPU_PREFLIGHT",
        "development_only": True, "model": MODEL, "profile": PROFILE,
        "target_sha": target_sha, "case_count": 1, "client_timeout_seconds": 30,
        "format": "nonstream_chat", "thinking_enabled": False,
        "status": "FAILED", "failure_stage": None, "failure_code": None,
        "no_training": True, "no_large_benchmark": True,
        "no_private_questions_or_answers_uploaded": True,
    }
    stage = "setup"
    try:
        from scripts.neral_foundation_2048_retest import ORIGINAL_DIGESTS
        from scripts.neral_same_base_live import require_model
        from scripts.neral_v7_foundation_tournament import OllamaLocal
        from scripts.neral_low_high_quality import make_prompt
        from neral_identity import build_identity_prompt

        stage = "nvidia_smi"
        report["nvidia_gpu"] = nvidia_gpu_inventory()
        stage = "windows_ollama_process_inventory"
        report["ollama_process_inventory"] = windows_ollama_process_inventory()
        stage = "digest"
        digest = require_model(OllamaLocal(timeout=20).inventory())
        if digest != ORIGINAL_DIGESTS[MODEL]:
            raise PreflightError("model_digest_changed")
        report["model_digest"] = digest

        stage = "ps_initial"
        loaded = _loaded()
        report["initial_loaded_models"] = sorted(loaded)
        report["initial_ps_sizes"] = {
            name: _gpu_summary(info) for name, info in loaded.items()
        }
        if LOW_MODEL in loaded:
            stage = "unload_qwen3_8b"
            _unload(LOW_MODEL)
        if MODEL in loaded:
            # Always evict any already-loaded HIGH model before a fresh
            # preload. This is not an Ollama service restart.
            stage = "fresh_unload_qwen35"
            _unload(MODEL)

        stage = "ps_after_eviction"
        loaded = _loaded()
        if LOW_MODEL in loaded or MODEL in loaded:
            raise PreflightError("model_still_resident_after_eviction")

        stage = "preload"
        _preload()
        stage = "ps_residency"
        try:
            require_high_gpu_residency(
                _ollama_json("/api/ps", timeout=8), report)
        except PreflightError as first_error:
            if str(first_error) != "high_model_cpu_offload_detected":
                raise
            from scripts.neral_ollama_cuda_recover import (
                recover_if_authorized, RecoveryRefused
            )
            stage = "cuda_daemon_recovery"
            report["cuda_fallback_detected"] = True
            report["gpu_residency_on_fallback"] = dict(
                report["gpu_residency_before"])
            recover_if_authorized(
                size_vram=report["gpu_residency_before"]["size_vram_bytes"],
                free_mib=report["nvidia_gpu"]["memory_free_mib"],
                consent=restart_authorization,
                report=report,
            )
            stage = "ps_after_cuda_recovery"
            _preload()
            require_high_gpu_residency(
                _ollama_json("/api/ps", timeout=8), report)
            report["cuda_recovery_verified_full_gpu"] = True

        stage = "one_direct_qwen_request"
        case = select_probe_case()
        prompt = make_prompt(case)
        messages = [
            {"role": "system", "content": build_identity_prompt(
                "web", "local_foundation_not_neral_adapter")},
            {"role": "user", "content": prompt},
        ]
        report["probe_domain"] = "math"
        report["probe_prompt_sha256"] = hashlib.sha256(prompt.encode("utf8")).hexdigest()
        request = measured_chat_payload(messages)
        started = time.monotonic()
        samples = []
        stop = threading.Event()
        watcher = threading.Thread(
            target=_watch_gpu_activity, args=(stop, samples), daemon=True)
        watcher.start()
        try:
            try:
                result = _ollama_json("/api/chat", data=request, timeout=30)
            except (TimeoutError, urllib.error.URLError) as exc:
                report["wall_seconds"] = round(time.monotonic() - started, 3)
                raise PreflightError("direct_qwen_client_timeout_or_transport") from exc
            report["wall_seconds"] = round(time.monotonic() - started, 3)
        finally:
            stop.set()
            watcher.join(timeout=4)
            report["gpu_activity_during_request"] = summarize_gpu_samples(samples)
        if report["wall_seconds"] > 30.0:
            raise PreflightError("direct_qwen_exceeded_30_seconds")
        report["ollama"] = sanitized_timings(result)

        stage = "ps_after_completion"
        loaded = running_model_summary(_ollama_json("/api/ps", timeout=8))
        if MODEL not in loaded or LOW_MODEL in loaded:
            raise PreflightError("gpu_residency_changed")
        report["gpu_residency_after"] = _gpu_summary(loaded[MODEL])
        stage = "release_after_successful_probe"
        release_high_model_after_probe(report)
        report["status"] = "PASS"
    except PreflightError as exc:
        report["failure_stage"] = stage
        report["failure_code"] = str(exc)
    except Exception as exc:
        from scripts.neral_ollama_cuda_recover import RecoveryRefused
        report["failure_stage"] = stage
        report["failure_code"] = (
            str(exc) if isinstance(exc, RecoveryRefused)
            else "local_environment_or_ollama_error"
        )
    with output.open("x", encoding="utf8") as f:
        json.dump(report, f, ensure_ascii=False, sort_keys=True, allow_nan=False)
        f.write("\n")
    print(json.dumps({
        "status": report["status"],
        "failure_stage": report["failure_stage"],
        "failure_code": report["failure_code"],
        "wall_seconds": report.get("wall_seconds"),
        "gpu_residency_before": report.get("gpu_residency_before"),
        "ollama": report.get("ollama"),
    }, sort_keys=True), flush=True)
    return report["status"] == "PASS"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--authorize", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--authorize-restart", default="")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        return 0 if run_probe(
            authorize=args.authorize, target_sha=args.target_sha,
            output=args.out,
            restart_authorization=args.authorize_restart) else 1
    except Exception:
        print("HIGH12_PREFLIGHT_ABORT: Invalid manual control or output setup", flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
