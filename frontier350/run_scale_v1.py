from __future__ import annotations

import argparse
import json
import math
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from recovery_identity import file_sha

ANCHOR_SHA = "ef3ee045203f1bfb7b9f501db800408b095aa5ef29da5d22a8a9486e115e9f7c"
ANCHOR_TOKENS = 1_010_003_584
TRAIN_TOKENS = 2_000_000
TARGET_TOKENS = ANCHOR_TOKENS + TRAIN_TOKENS
PACKS = (4, 5, 6, 8)
ALPHAS = (0.025, 0.0125, 0.00625, 0.003125)
SEQ = 512
BATCH = 4
MICROBATCHES = 12
TOKENS_PER_UPDATE = SEQ * BATCH * MICROBATCHES
SWEEP_UPDATES = 12
SWEEP_WARMUP = 3
MAX_ALLOCATED = int(10.25 * 1024**3)
MAX_RESERVED = int(11.0 * 1024**3)


def run(script: str, *args: object) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).parent / script), *map(str, args)]
    print(json.dumps({"event": "exec", "argv": cmd}), flush=True)
    subprocess.run(cmd, check=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip() and json.loads(x).get("event") == "train"]


def p10(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.floor((len(ordered) - 1) * 0.10))]


def strict_bilingual(base: dict, cand: dict, dev: dict) -> bool:
    for obj in (base, cand):
        if obj.get("cases") != 400 or set(obj.get("directions", {})) != {"de_to_en", "en_to_de"}:
            raise ValueError("invalid bilingual evidence")
    for key in ("cases_sha256", "tokenizer_sha256", "protocol_sha256"):
        if base.get(key) != cand.get(key):
            raise ValueError("bilingual protocol changed")
    for key in ("checkpoint_sha256", "checkpoint_processed_tokens", "tokenizer_sha256"):
        if cand.get(key) != dev.get(key):
            raise ValueError("mixed checkpoint evidence")
    return cand["overall_accuracy_pct"] >= base["overall_accuracy_pct"] and all(
        cand["directions"][d]["accuracy_pct"] >= base["directions"][d]["accuracy_pct"]
        for d in base["directions"]
    )


def resolve_anchor(assets: Path) -> Path:
    preferred = assets / "runs/neral350_autorecovery_neral-autorecovery-v1-20260912-r1_selected/resume.pt"
    if preferred.is_file() and file_sha(preferred) == ANCHOR_SHA:
        return preferred
    matches = []
    for p in sorted((assets / "runs").glob("neral350_*_selected/resume.pt")):
        if p.is_file() and file_sha(p) == ANCHOR_SHA:
            matches.append(p)
    if not matches:
        raise ValueError("safe 1.010B anchor not found by SHA")
    return matches[0]


def main() -> None:
    ap = argparse.ArgumentParser()
    for n in ("assets", "report", "scratch", "request-id"):
        ap.add_argument("--" + n, required=True)
    a = ap.parse_args()
    assets, report, scratch = map(lambda p: Path(p).resolve(), (a.assets, a.report, a.scratch))
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report not empty")
    if scratch.exists():
        raise FileExistsError("scratch already exists")
    report.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True)

    anchor = resolve_anchor(assets)
    if file_sha(anchor) != ANCHOR_SHA:
        raise ValueError("anchor identity changed")

    tokenizer = assets / "frontier350_assets/tokenizer.json"
    main_root = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05"
    main_data = main_root / "train"
    replay = assets / "frontier350_replay_v9_20260912"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid"
    cases = main_root / "bilingual_holdout.json"

    def evaluate(checkpoint: Path, tag: str, tokens: int) -> dict:
        val = report / f"{tag}_validation.json"
        dev = report / f"{tag}_dev.json"
        bi = report / f"{tag}_bilingual.json"
        run("eval_validation_loss350.py", "--checkpoint", checkpoint, "--valid-data", valid, "--out", val,
            "--seq", 512, "--batch", 4, "--batches", 32, "--seed", 351337)
        run("eval_recovery_dev_v2.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer, "--out", dev,
            "--expected-tokens", tokens)
        run("eval_bilingual_holdout_v1.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer,
            "--cases", cases, "--out", bi, "--expected-tokens", tokens)
        out = {"validation": load(val), "dev": load(dev), "bilingual": load(bi)}
        sha = file_sha(checkpoint)
        for obj in out.values():
            if obj.get("checkpoint_sha256") != sha or obj.get("checkpoint_processed_tokens") != tokens:
                raise ValueError("evaluation identity mismatch")
        return out

    baseline = evaluate(anchor, "anchor", ANCHOR_TOKENS)

    def train_once(tag: str, pack: int, tokens: int, checked: bool = True) -> tuple[Path, int]:
        out = scratch / tag
        cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_guarded_v9.py"),
               "--anchor", str(anchor), "--anchor-sha", ANCHOR_SHA,
               "--main-data", str(main_data), "--replay", str(replay), "--out", str(out),
               "--mode", "replay_only", "--new-tokens", str(tokens), "--batch-size", "4",
               "--microbatches", "12", "--main-pack", str(pack), "--compile-mode", "default",
               "--fast-replay-only"]
        print(json.dumps({"event": "train_start", "tag": tag, "pack": pack, "tokens": tokens}), flush=True)
        proc = subprocess.run(cmd)
        if checked and proc.returncode != 0:
            raise RuntimeError(f"training failed for {tag}")
        return out / "resume.pt", proc.returncode

    # Phase 1: bounded geometry sweep. No sweep checkpoint can be promoted.
    sweep = []
    sweep_budget = TOKENS_PER_UPDATE * SWEEP_UPDATES
    for pack in PACKS:
        tag = f"sweep_pack{pack}"
        raw, code = train_once(tag, pack, sweep_budget, checked=False)
        row = {"pack": pack, "returncode": code}
        out = raw.parent
        try:
            if code != 0:
                row.update(stable=False, pass_memory=False, error="trainer returned nonzero")
            else:
                s = load(out / "summary.json")
                train_rows = rows(out / "metrics.jsonl")
                stable_rows = train_rows[SWEEP_WARMUP:]
                rates = [float(r["tokens_per_second"]) for r in stable_rows if r.get("tokens_per_update") == TOKENS_PER_UPDATE]
                if len(rates) < 5 or not all(math.isfinite(x) and x > 0 for x in rates):
                    raise ValueError("insufficient stable throughput samples")
                alloc = max(int(r["peak_allocated_bytes"]) for r in train_rows)
                reserved = max(int(r.get("peak_reserved_bytes", 0)) for r in train_rows)
                med = statistics.median(rates)
                low = p10(rates)
                row.update(median_tps=med, mean_tps=statistics.fmean(rates), p10_tps=low, peak_tps=max(rates),
                           peak_allocated_bytes=alloc, peak_reserved_bytes=reserved,
                           peak_allocated_gib=alloc / 1024**3, peak_reserved_gib=reserved / 1024**3,
                           pass_memory=alloc <= MAX_ALLOCATED and reserved <= MAX_RESERVED,
                           stable=low >= 0.90 * med,
                           frozen_weights_unchanged=bool(s.get("frozen_weights_unchanged")))
                shutil.copy2(out / "summary.json", report / f"{tag}_summary.json")
                shutil.copy2(out / "metrics.jsonl", report / f"{tag}_metrics.jsonl")
        except Exception as exc:
            row.update(stable=False, pass_memory=False, error=str(exc))
        sweep.append(row)
        (report / "sweep_progress.json").write_text(json.dumps(sweep, indent=2), encoding="utf-8")
        print(json.dumps({"event": "sweep_trial", **row}), flush=True)
        shutil.rmtree(out, ignore_errors=True)

    eligible = [r for r in sweep if r.get("returncode") == 0 and r.get("stable") and r.get("pass_memory") and r.get("frozen_weights_unchanged") and "median_tps" in r]
    if not eligible:
        raise RuntimeError("no stable pack geometry survived")
    best = max(eligible, key=lambda r: r["median_tps"])
    pack4 = next((r for r in eligible if r["pack"] == 4), None)
    speedup_vs_pack4 = best["median_tps"] / pack4["median_tps"] if pack4 else None
    print(json.dumps({"event": "sweep_selection", "best": best, "speedup_vs_pack4": speedup_vs_pack4}), flush=True)

    # Phase 2: real 2M guarded continuation with the fastest stable geometry.
    selected_pack = int(best["pack"])
    raw, code = train_once(f"train2m_pack{selected_pack}", selected_pack, TRAIN_TOKENS, checked=True)
    summary = load(raw.parent / "summary.json")
    if not summary.get("complete") or summary.get("processed_tokens") != TARGET_TOKENS or not summary.get("frozen_weights_unchanged"):
        raise ValueError("2M training contract failed")
    shutil.copy2(raw.parent / "summary.json", report / "train2m_summary.json")
    shutil.copy2(raw.parent / "metrics.jsonl", report / "train2m_metrics.jsonl")
    full_rates = [float(r["tokens_per_second"]) for r in rows(raw.parent / "metrics.jsonl")[5:] if r.get("tokens_per_update") == TOKENS_PER_UPDATE]
    train_throughput = {"median_tps": statistics.median(full_rates), "p10_tps": p10(full_rates), "peak_tps": max(full_rates)} if full_rates else {}

    candidates = []
    selected = None
    for alpha in ALPHAS:
        tag = f"pack{selected_pack}_2m_a{str(alpha).replace('.', 'p')}"
        snap = scratch / f"{tag}.pt"
        run("interpolate_recovery_checkpoint.py", "--anchor", anchor, "--candidate", raw, "--out", snap, "--alpha", alpha)
        measured = evaluate(snap, tag, TARGET_TOKENS)
        gate = report / f"{tag}_legacy_gate.json"
        run("select_recovery_lr_pilot.py", "--baseline-dev", report / "anchor_dev.json",
            "--baseline-validation", report / "anchor_validation.json", "--candidate", tag, 7.5e-5,
            report / f"{tag}_dev.json", report / f"{tag}_validation.json", "--out", gate)
        legacy = load(gate)
        bi_ok = strict_bilingual(baseline["bilingual"], measured["bilingual"], measured["dev"])
        improved = measured["validation"]["validation_loss"] < baseline["validation"]["validation_loss"] or measured["bilingual"]["overall_accuracy_pct"] > baseline["bilingual"]["overall_accuracy_pct"] or any(
            measured["dev"]["categories"][k]["accuracy_pct"] > baseline["dev"]["categories"][k]["accuracy_pct"]
            for k in baseline["dev"]["categories"])
        passed = bool(legacy.get("selection_allowed")) and bi_ok and improved
        row = {"pack": selected_pack, "alpha": alpha, "pass": passed,
               "legacy_gate": bool(legacy.get("selection_allowed")),
               "legacy_failures": legacy.get("candidates", [{}])[0].get("failures", []),
               "bilingual_gate": bi_ok, "strict_improvement": improved,
               "validation_loss": measured["validation"]["validation_loss"],
               "overall_pct": measured["dev"]["overall_accuracy_pct"],
               "categories": {k: v["accuracy_pct"] for k, v in measured["dev"]["categories"].items()},
               "bilingual_pct": measured["bilingual"]["overall_accuracy_pct"],
               "de_to_en_pct": measured["bilingual"]["directions"]["de_to_en"]["accuracy_pct"],
               "en_to_de_pct": measured["bilingual"]["directions"]["en_to_de"]["accuracy_pct"],
               "checkpoint_sha256": file_sha(snap), "path": str(snap), "throughput": train_throughput}
        candidates.append(row)
        (report / "recovery_progress.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
        print(json.dumps({"event": "candidate", **row}), flush=True)
        if not passed:
            snap.unlink(missing_ok=True)

    passing = [c for c in candidates if c["pass"]]
    if passing:
        selected = min(passing, key=lambda c: (c["validation_loss"], -c["bilingual_pct"], -c["alpha"]))

    promoted = None
    if selected:
        destdir = assets / "runs" / f"neral350_scale_{a.request_id}_selected"
        if destdir.exists():
            raise FileExistsError("promotion destination exists")
        destdir.mkdir(parents=True)
        dest = destdir / "resume.pt"
        shutil.copy2(selected["path"], dest)
        if file_sha(dest) != selected["checkpoint_sha256"]:
            raise RuntimeError("promotion hash mismatch")
        promoted = str(dest)

    result = {"schema": 1, "controller": "neral350-scale-v1", "request_id": a.request_id,
              "anchor_sha256": ANCHOR_SHA, "anchor_tokens": ANCHOR_TOKENS,
              "target_tokens": TARGET_TOKENS, "train_tokens": TRAIN_TOKENS,
              "sweep": sweep, "best_pack": best, "speedup_vs_pack4": speedup_vs_pack4,
              "train_throughput": train_throughput, "candidates": candidates,
              "success": selected is not None, "selected": selected,
              "promoted_checkpoint": promoted, "safe_anchor_preserved_if_failure": True,
              "benchmarks_used_for_training": False, "final_audit_pending": True,
              "next_if_success": "Run 5M Scale V2 with smoke gates before 10M/25M blocks."}
    (report / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"event": "final_selection", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
