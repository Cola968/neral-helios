"""Bounded 1M-token validation canary for the Neral 350M fast replay path.

This does not promote model weights. It verifies that the r4-winning execution geometry
(fast replay-only, main_pack=4, torch.compile default) can complete a real 1M-token
continuation from the accepted 1.009B anchor while preserving the existing validation,
development-category and bilingual floors. Benchmarks remain evaluation-only.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

from recovery_identity import file_sha

ANCHOR_SHA = "b780d089a72a54dea94b4669b696924259c24add909b010ee62670fe0c066894"
ANCHOR_TOKENS = 1_009_003_584
NEW_TOKENS = 1_000_000
TARGET_TOKENS = ANCHOR_TOKENS + NEW_TOKENS
TOKENS_PER_FULL_UPDATE = 24_576
WARMUP_UPDATES = 4
MIN_MEDIAN_TPS = 40_000.0
MIN_P10_TPS = 39_000.0
MAX_RESERVED = int(11.0 * 1024**3)


def execute(script: str, *args: object) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).parent / script), *map(str, args)]
    print(json.dumps({"event": "exec", "argv": cmd}), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def percentile10(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.floor((len(ordered) - 1) * 0.10))]


def resolve_anchor(assets: Path) -> Path:
    runs = assets / "runs"
    likely = sorted(runs.glob("neral350_autolearn_*/resume.pt"))
    fallback = sorted(runs.glob("neral350_*_selected/resume.pt"))
    seen: set[str] = set()
    evidence = []
    matches = []
    for path in [*likely, *fallback]:
        key = str(path.resolve())
        if key in seen or not path.is_file():
            continue
        seen.add(key)
        sha = file_sha(path)
        row = {"path": str(path), "sha256": sha, "match": sha == ANCHOR_SHA}
        evidence.append(row)
        print(json.dumps({"event": "anchor_candidate", **row}), flush=True)
        if sha == ANCHOR_SHA:
            matches.append(path)
    if not matches:
        raise ValueError("accepted 1.009B anchor not found by SHA-256")
    chosen = sorted(matches, key=lambda p: str(p))[0]
    print(json.dumps({"event": "anchor_resolved", "path": str(chosen), "sha256": ANCHOR_SHA}), flush=True)
    return chosen


def strict_bilingual(base: dict, candidate: dict, dev: dict) -> bool:
    for obj in (base, candidate):
        if obj.get("cases") != 400 or set(obj.get("directions", {})) != {"de_to_en", "en_to_de"}:
            raise ValueError("incomplete bilingual evidence")
        total = correct = 0
        for direction in obj["directions"].values():
            n, k = direction["total"], direction["correct"]
            if type(n) is not int or type(k) is not int or n != 200 or not 0 <= k <= n:
                raise ValueError("invalid bilingual counts")
            pct = direction["accuracy_pct"]
            if isinstance(pct, bool) or not math.isfinite(pct) or abs(pct - 100 * k / n) > 1e-9:
                raise ValueError("inconsistent bilingual direction score")
            total += n
            correct += k
        pct = obj["overall_accuracy_pct"]
        if isinstance(pct, bool) or not math.isfinite(pct) or abs(pct - 100 * correct / total) > 1e-9:
            raise ValueError("inconsistent bilingual total score")
    for key in ("cases_sha256", "tokenizer_sha256", "protocol_sha256"):
        if not base.get(key) or base[key] != candidate.get(key):
            raise ValueError("bilingual protocol/data changed")
    for key in ("checkpoint_sha256", "checkpoint_processed_tokens", "tokenizer_sha256"):
        if candidate[key] != dev[key]:
            raise ValueError("mixed checkpoint evidence")
    return (
        candidate["overall_accuracy_pct"] >= base["overall_accuracy_pct"]
        and all(
            candidate["directions"][d]["accuracy_pct"] >= base["directions"][d]["accuracy_pct"]
            for d in base["directions"]
        )
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("assets", "report", "scratch", "request-id"):
        ap.add_argument("--" + name, required=True)
    a = ap.parse_args()
    if not a.request_id or len(a.request_id) > 64 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in a.request_id):
        raise ValueError("unsafe request id")

    assets = Path(a.assets).resolve()
    report = Path(a.report).resolve()
    scratch = Path(a.scratch).resolve()
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report directory is not empty")
    if scratch.exists():
        raise FileExistsError("scratch directory already exists")
    report.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True)

    anchor = resolve_anchor(assets)
    if file_sha(anchor) != ANCHOR_SHA:
        raise ValueError("anchor identity changed")

    tokenizer = assets / "frontier350_assets/tokenizer.json"
    main_data = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05"
    replay = assets / "frontier350_replay_v9_20260912"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid"
    cases = main_data / "bilingual_holdout.json"
    for path in (tokenizer, main_data / "manifest.json", replay / "manifest.json", cases):
        if not path.exists():
            raise FileNotFoundError(path)
    shutil.copyfile(main_data / "manifest.json", report / "main_manifest.json")
    shutil.copyfile(replay / "manifest.json", report / "replay_manifest.json")

    def evaluate(checkpoint: Path, tag: str, expected_tokens: int) -> dict:
        validation = report / f"{tag}_validation.json"
        dev = report / f"{tag}_dev.json"
        bilingual = report / f"{tag}_bilingual.json"
        execute(
            "eval_validation_loss350.py",
            "--checkpoint", checkpoint,
            "--valid-data", valid,
            "--out", validation,
            "--seq", 512,
            "--batch", 4,
            "--batches", 32,
            "--seed", 351337,
        )
        execute(
            "eval_recovery_dev_v2.py",
            "--checkpoint", checkpoint,
            "--tokenizer", tokenizer,
            "--out", dev,
            "--expected-tokens", expected_tokens,
        )
        execute(
            "eval_bilingual_holdout_v1.py",
            "--checkpoint", checkpoint,
            "--tokenizer", tokenizer,
            "--cases", cases,
            "--out", bilingual,
            "--expected-tokens", expected_tokens,
        )
        result = {"validation": load_json(validation), "dev": load_json(dev), "bilingual": load_json(bilingual)}
        sha = file_sha(checkpoint)
        for obj in result.values():
            if obj.get("checkpoint_sha256") != sha or obj.get("checkpoint_processed_tokens") != expected_tokens:
                raise ValueError("evaluation identity mismatch")
        return result

    print(json.dumps({"event": "baseline_eval_start", "tokens": ANCHOR_TOKENS}), flush=True)
    baseline = evaluate(anchor, "anchor", ANCHOR_TOKENS)

    out = scratch / "fast_pack4_canary"
    execute(
        "train_guarded_v9.py",
        "--anchor", anchor,
        "--anchor-sha", ANCHOR_SHA,
        "--main-data", main_data / "train",
        "--replay", replay,
        "--out", out,
        "--mode", "replay_only",
        "--new-tokens", NEW_TOKENS,
        "--batch-size", 4,
        "--microbatches", 12,
        "--main-pack", 4,
        "--fast-replay-only",
        "--compile-mode", "default",
    )
    summary = load_json(out / "summary.json")
    if not summary.get("complete") or summary.get("new_tokens") != NEW_TOKENS or summary.get("processed_tokens") != TARGET_TOKENS:
        raise ValueError("canary did not complete exact 1M-token budget")
    if summary.get("main_pack") != 4 or summary.get("fast_replay_only") is not True:
        raise ValueError("fast-path execution contract changed")
    if summary.get("frozen_weights_unchanged") is not True:
        raise RuntimeError("frozen weights changed")
    shutil.copyfile(out / "summary.json", report / "train_summary.json")
    shutil.copyfile(out / "metrics.jsonl", report / "train_metrics.jsonl")

    rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines() if line.strip()]
    train_rows = [r for r in rows if r.get("event") == "train" and int(r.get("tokens_per_update", 0)) == TOKENS_PER_FULL_UPDATE]
    stable_rows = train_rows[WARMUP_UPDATES:]
    rates = [float(r["tokens_per_second"]) for r in stable_rows]
    if len(rates) < 10 or not all(math.isfinite(x) and x > 0 for x in rates):
        raise ValueError("insufficient valid stable throughput evidence")
    median_tps = statistics.median(rates)
    mean_tps = statistics.fmean(rates)
    p10_tps = percentile10(rates)
    peak_tps = max(rates)
    peak_reserved = int(summary.get("peak_reserved_bytes", 2**63))
    throughput_gate = median_tps >= MIN_MEDIAN_TPS and p10_tps >= MIN_P10_TPS and peak_reserved <= MAX_RESERVED

    candidate_ckpt = out / "resume.pt"
    candidate_sha = file_sha(candidate_ckpt)
    print(json.dumps({"event": "candidate_eval_start", "sha256": candidate_sha, "tokens": TARGET_TOKENS}), flush=True)
    candidate = evaluate(candidate_ckpt, "candidate", TARGET_TOKENS)

    gate_path = report / "legacy_gate.json"
    execute(
        "select_recovery_lr_pilot.py",
        "--baseline-dev", report / "anchor_dev.json",
        "--baseline-validation", report / "anchor_validation.json",
        "--candidate", "fast_pack4_canary", 7.5e-5,
        report / "candidate_dev.json",
        report / "candidate_validation.json",
        "--out", gate_path,
    )
    legacy = load_json(gate_path)
    legacy_gate = bool(legacy.get("selection_allowed"))
    bilingual_gate = strict_bilingual(baseline["bilingual"], candidate["bilingual"], candidate["dev"])
    pass_all = legacy_gate and bilingual_gate and throughput_gate

    result = {
        "schema": 1,
        "canary": "neral350-fastpath-canary-v1",
        "request_id": a.request_id,
        "anchor_sha256": ANCHOR_SHA,
        "anchor_tokens": ANCHOR_TOKENS,
        "candidate_sha256": candidate_sha,
        "candidate_tokens": TARGET_TOKENS,
        "training": {
            "new_tokens": NEW_TOKENS,
            "mode": "replay_only",
            "main_pack": 4,
            "fast_replay_only": True,
            "compile_mode": "default",
            "replay_fraction": 0.25,
            "checkpoint_promoted": False,
        },
        "throughput": {
            "median_tps": median_tps,
            "mean_tps": mean_tps,
            "p10_tps": p10_tps,
            "peak_tps": peak_tps,
            "stable_full_updates": len(rates),
            "min_median_tps": MIN_MEDIAN_TPS,
            "min_p10_tps": MIN_P10_TPS,
            "peak_allocated_gib": int(summary["peak_allocated_bytes"]) / 1024**3,
            "peak_reserved_gib": peak_reserved / 1024**3,
            "pass": throughput_gate,
        },
        "retention": {
            "legacy_gate": legacy_gate,
            "legacy_failures": legacy.get("candidates", [{}])[0].get("failures", []),
            "bilingual_gate": bilingual_gate,
            "baseline_validation_loss": baseline["validation"]["validation_loss"],
            "candidate_validation_loss": candidate["validation"]["validation_loss"],
            "baseline_overall_pct": baseline["dev"]["overall_accuracy_pct"],
            "candidate_overall_pct": candidate["dev"]["overall_accuracy_pct"],
            "baseline_categories": {k: v["accuracy_pct"] for k, v in baseline["dev"]["categories"].items()},
            "candidate_categories": {k: v["accuracy_pct"] for k, v in candidate["dev"]["categories"].items()},
            "baseline_bilingual_pct": baseline["bilingual"]["overall_accuracy_pct"],
            "candidate_bilingual_pct": candidate["bilingual"]["overall_accuracy_pct"],
            "baseline_de_to_en_pct": baseline["bilingual"]["directions"]["de_to_en"]["accuracy_pct"],
            "candidate_de_to_en_pct": candidate["bilingual"]["directions"]["de_to_en"]["accuracy_pct"],
            "baseline_en_to_de_pct": baseline["bilingual"]["directions"]["en_to_de"]["accuracy_pct"],
            "candidate_en_to_de_pct": candidate["bilingual"]["directions"]["en_to_de"]["accuracy_pct"],
        },
        "canary_pass": pass_all,
        "adopt_fastpath_recommended": pass_all,
        "benchmarks_used_for_training": False,
        "checkpoint_promoted": False,
        "next_step": "If pass, adopt fast_pack4_default as the guarded trainer execution path; if retention fails, keep legacy execution and diagnose with an A/B semantic-equivalence canary.",
    }
    (report / "canary_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"event": "canary_result", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
