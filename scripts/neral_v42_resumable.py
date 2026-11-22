"""Owner-only resumable, per-tier sealed benchmark with bounded task workers.

Privacy: complete answers and sealed keys remain in owner-local JSONL. Only
aggregates are suitable for CI artifacts. No training/capture is enabled.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

from scripts.neral_v41_blind200 import CATEGORIES, TIERS, validate_suite
from scripts.neral_v41_blind200_live import (
    IMAGE, OwnerClient, execute_case, grade_response,
)

TIMEOUTS = {"Amber": 30, "Quartz": 45, "Obsidian": 75}
MAX_BATCH = 50


def balanced_sample(cases, amount=50):
    """Stable largest-remainder allocation over the seven blind categories."""
    if amount < len(CATEGORIES) or amount > len(cases):
        raise ValueError("Balanced sample size out of bounds")
    buckets = {key: [c for c in cases if c["category"] == key] for key in CATEGORIES}
    capacity = sum(len(x) for x in buckets.values())
    desired = {cat: amount * len(rows) / capacity for cat, rows in buckets.items()}
    take = {cat: int(quota) for cat, quota in desired.items()}
    leftover = amount - sum(take.values())
    order = sorted(CATEGORIES, key=lambda c: (-(desired[c] - take[c]), list(CATEGORIES).index(c)))
    for cat in order[:leftover]:
        take[cat] += 1
    result = [case for case in cases if take[case["category"]] > 0]
    # Preserve the original sealed order while taking exactly each quota.
    seen = Counter()
    selected = []
    for case in result:
        cat = case["category"]
        if seen[cat] < take[cat]:
            selected.append(case)
            seen[cat] += 1
    if len(selected) != amount or set(seen) != set(CATEGORIES):
        raise ValueError("Cannot cover all seven categories")
    return selected


def select_batch(cases, *, index=0, count=50):
    if type(index) is not int or type(count) is not int or index < 0 or count < 1 or count > MAX_BATCH:
        raise ValueError("Invalid batch offset/count; maximum 50 per GPU job")
    selected = cases[index:index + count]
    if len(selected) != count:
        raise ValueError("Batch exceeds sealed suite")
    return selected


def parse_vram_mb(raw):
    match = re.fullmatch(r"\s*(\d+)\s*(?:MiB)?\s*", str(raw), re.I)
    return int(match.group(1)) if match else None


def sample_gpu_mb():
    """Total GPU memory used, not model-attributable memory; may include games."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            check=False, timeout=3,
        )
        if r.returncode == 0:
            samples = [parse_vram_mb(line) for line in r.stdout.splitlines()]
            return max((x for x in samples if x is not None), default=None)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return None


def make_record(case, tier, response, grade, reason, elapsed, vram_peak_mb, error):
    text = str(response.get("text", "")) if type(response) is dict else ""
    return {
        "id": case["id"], "task_id": case["id"], "tier": tier,
        "category": case["category"],
        "prompt_sha256": sha256(case["prompt"].encode("utf-8")).hexdigest(),
        "score": 1 if grade == "pass" else 0 if grade == "fail" else None,
        "grade": grade, "reason": reason,
        "latency_ms": round(1000 * elapsed),
        "latency_s": round(elapsed, 4),
        "vram_peak_mb": vram_peak_mb, "error": error,
        # Private ONLY: needed for independent review, never upload this file.
        "model_text": text[:16000],
        "model_origin": response.get("model_origin"),
        "model": response.get("model"), "effective_tier": response.get("tier"),
        "model_generated": response.get("model_generated"),
        "tool_used": response.get("tool_used"),
        "verification_scope": response.get("verification_scope"),
        "error_code": response.get("error_code"),
        "adapter_loaded": response.get("adapter_loaded"),
        "repair_attempts": response.get("repair_attempts"),
        "repair_success": response.get("repair_success"),
    }


def load_checkpoint(path, all_cases, tiers):
    """Validate every completed (tier, task_id) before it can be skipped."""
    if not path.exists():
        return []
    allowed = {c["id"]: c for c in all_cases}
    seen = set()
    selected = []
    with path.open("r", encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except ValueError as ex:
                raise ValueError(f"Damaged checkpoint JSONL line {number}; restore snapshot") from ex
            if type(row) is not dict or row.get("tier") not in TIERS:
                raise ValueError(f"Invalid checkpoint tier at line {number}")
            tier = row["tier"]
            case_id = row.get("task_id", row.get("id"))
            case = allowed.get(case_id)
            if case is None:
                raise ValueError(f"Unknown checkpoint task at line {number}")
            if row.get("prompt_sha256") != sha256(case["prompt"].encode("utf-8")).hexdigest():
                raise ValueError(f"Sealed task digest mismatch at line {number}")
            key = (tier, case_id)
            if key in seen:
                raise ValueError(f"Duplicate checkpoint tier/task at line {number}")
            seen.add(key)
            if tier in tiers:
                selected.append(row)
    return selected


def _child(conn, tier, case):
    try:
        code, response, elapsed, setup_failure = execute_case(OwnerClient(), tier, case)
        if code != 200:
            grade, reason = "fail", "http_" + str(code)
        elif setup_failure:
            grade, reason = "fail", setup_failure
        else:
            grade, reason = grade_response(case, response)
        conn.send(("ok", code, response, grade, reason))
    except BaseException as exc:
        # Isolation failure is never converted into a scored failure.
        if "isolated, no-training" in str(exc):
            conn.send(("fatal", 0, {}, "fail", "benchmark_isolation_failed"))
        else:
            conn.send(("ok", 0, {}, "fail", "benchmark_exception"))
    finally:
        conn.close()


def bounded_evaluate(tier, case):
    """Hard wall-clock cap on client+grader process, including multistep history."""
    ctx = mp.get_context("spawn")
    receiver, sender = ctx.Pipe(duplex=False)
    worker = ctx.Process(target=_child, args=(sender, tier, case))
    start = time.monotonic()
    peak = sample_gpu_mb()
    worker.start()
    sender.close()
    result = None
    budget = TIMEOUTS[tier]
    try:
        while time.monotonic() - start < budget:
            remaining = budget - (time.monotonic() - start)
            if receiver.poll(min(1.5, max(0.0, remaining))):
                try:
                    result = receiver.recv()
                except EOFError:
                    result = None
                break
            sample = sample_gpu_mb()
            if sample is not None:
                peak = sample if peak is None else max(peak, sample)
            if not worker.is_alive():
                break
        elapsed = time.monotonic() - start
        sample = sample_gpu_mb()
        if sample is not None:
            peak = sample if peak is None else max(peak, sample)
        if result is None:
            reason = "task_timeout" if worker.is_alive() and elapsed >= budget else "worker_failed"
            return make_record(case, tier, {}, "fail", reason, min(elapsed, budget), peak, reason)
        status, code, response, grade, reason = result
        if status == "fatal":
            raise RuntimeError("Benchmark isolation verification failed; abort")
        failure = reason if reason in (
            "benchmark_exception", "runtime_error", "task_timeout",
            "worker_failed",
        ) or str(reason).startswith("http_") else None
        return make_record(case, tier, response, grade, reason, elapsed, peak, failure)
    finally:
        if worker.is_alive():
            worker.terminate()
        worker.join(timeout=4)
        if worker.is_alive():
            worker.kill()
            worker.join(timeout=2)
        receiver.close()


def _aggregate(rows, manifest, tiers, expected):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["tier"], row["category"])].append(row)
    summary = {
        "status": "COMPLETE" if len(rows) == expected else "PARTIAL",
        "sealed_sha256": manifest["sha256"],
        "observations": len(rows), "expected": expected,
        "tiers": list(tiers),
        "no_training_capture": True,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "vram_measurement": "Total GPU memory.used sampled periodically via nvidia-smi; other applications included",
        "measurement_note": "Ungraded writing/explanation requires independent human review",
        "categories": [],
    }
    for tier in tiers:
        for category in CATEGORIES:
            rs = groups[(tier, category)]
            graded = [r for r in rs if r.get("grade") in ("pass", "fail")]
            passed = sum(r.get("grade") == "pass" for r in rs)
            repair = [r for r in rs if isinstance(r.get("repair_attempts"), int) and r["repair_attempts"] > 0]
            rates = [float(r.get("latency_ms", float(r.get("latency_s", 0)) * 1000)) for r in rs]
            vram = [r.get("vram_peak_mb") for r in rs if isinstance(r.get("vram_peak_mb"), (int, float))]
            summary["categories"].append({
                "tier": tier, "category": category, "n": len(rs),
                "graded": len(graded), "passed": passed,
                "failed": sum(r.get("grade") == "fail" for r in rs),
                "ungraded": sum(r.get("grade") == "ungraded" for r in rs),
                "pass_rate_graded_pct": round(passed / len(graded) * 100, 2) if graded else None,
                "mean_latency_ms": round(statistics.mean(rates), 1) if rates else None,
                "max_latency_ms": round(max(rates), 1) if rates else None,
                "vram_peak_mb": max(vram) if vram else None,
                "repair_attempt_count": len(repair),
                "repair_hit_rate_pct": (
                    round(100 * sum(r.get("repair_success") is True for r in repair) / len(repair), 2)
                    if repair else None
                ),
                "errors": dict(Counter(str(r.get("error")) for r in rs if r.get("error"))),
            })
    return summary


def run(sealed, manifest_path, private_details, summary_path, *,
        tier, count=50, batch_offset=0, balanced_50=False):
    if tier not in TIERS:
        raise ValueError("Each job must select exactly one NERAL tier")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sealed_info = validate_suite(sealed)
    if manifest["sha256"] != sealed_info["sha256"]:
        raise ValueError("Sealed SHA256 mismatch")
    all_cases = [json.loads(line) for line in sealed.read_text(encoding="utf-8").splitlines() if line.strip()]
    if balanced_50:
        if count != 50 or batch_offset:
            raise ValueError("Balanced A/B is exactly 50 tasks, no offset")
        selected = balanced_sample(all_cases, 50)
    else:
        selected = select_batch(all_cases, index=batch_offset, count=count)
    # No-training probe before any private task is submitted.
    from neral_docker_runtime import ensure_docker_ready
    if not ensure_docker_ready():
        raise RuntimeError("Offline Docker daemon unavailable")
    check = subprocess.run(["docker", "image", "inspect", IMAGE],
                           capture_output=True, timeout=15, check=False)
    if check.returncode:
        raise RuntimeError("Offline grader Docker image unavailable")
    client = OwnerClient()
    client.connect()
    code, data, _ = client.post("Amber", "Hallo!", [])
    if code != 200 or data.get("benchmark_isolated") is not True:
        raise RuntimeError("Three-tier API does not enforce no-training isolation")
    existing = load_checkpoint(private_details, all_cases, (tier,))
    keys = {r.get("task_id", r.get("id")) for r in existing}
    selected_ids = {r["id"] for r in selected}
    rows = [r for r in existing if r.get("task_id", r.get("id")) in selected_ids]
    private_details.parent.mkdir(parents=True, exist_ok=True)
    # Append per-task; never truncate existing private history.
    with private_details.open("a", encoding="utf-8") as out:
        for case in selected:
            if case["id"] in keys:
                continue
            record = bounded_evaluate(tier, case)
            out.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
            rows.append(record)
            keys.add(case["id"])
            print(f"NERAL_V42_CHECKPOINT tier={tier} {len(rows)}/{len(selected)} "
                  f"category={case['category']} grade={record['grade']} "
                  f"latency_ms={record['latency_ms']} error={record['error']}", flush=True)
    OwnerClient().get("/")
    summary = _aggregate(rows, manifest, (tier,), len(selected))
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
    print(f"NERAL_V42_TIER_COMPLETE tier={tier} observations={len(rows)} "
          f"expected={len(selected)} HTTP200_PORT9690", flush=True)
    return summary


def main():
    p = argparse.ArgumentParser()
    for name in ("sealed", "manifest", "private-details", "summary"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--tier", choices=TIERS, required=True)
    p.add_argument("--count", type=int, default=50)
    p.add_argument("--batch-offset", type=int, default=0)
    p.add_argument("--balanced-50", action="store_true")
    args = p.parse_args()
    run(args.sealed, args.manifest, args.private_details, args.summary,
        tier=args.tier, count=args.count, batch_offset=args.batch_offset,
        balanced_50=args.balanced_50)


if __name__ == "__main__":
    main()
