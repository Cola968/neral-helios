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

ANCHOR_SHA = "ef3ee045203f1bfb7b9f501db800408b095aa5ef29da5d22a8a9486e115e9f7c"
ANCHOR_TOKENS = 1_010_003_584
PACK = 6
LRS = (5e-5, 3.75e-5, 2.5e-5)
ALPHAS = (0.05, 0.0375, 0.025, 0.01875, 0.0125, 0.00625)
PRIMARY_TOKENS = 2_000_000
FALLBACK_TOKENS = 1_000_000
SEQ = 512
BATCH = 4
MICROBATCHES = 12
TOKENS_PER_UPDATE = SEQ * BATCH * MICROBATCHES


def run(script: str, *args: object) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).parent / script), *map(str, args)]
    print(json.dumps({"event": "exec", "argv": cmd}), flush=True)
    subprocess.run(cmd, check=True)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def train_rows(path: Path) -> list[dict]:
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            obj = json.loads(line)
            if obj.get("event") == "train":
                out.append(obj)
    return out


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
    matches = [p for p in sorted((assets / "runs").glob("neral350_*_selected/resume.pt")) if p.is_file() and file_sha(p) == ANCHOR_SHA]
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
    tokenizer = assets / "frontier350_assets/tokenizer.json"
    main_root = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05"
    main_data = main_root / "train"
    replay = assets / "frontier350_replay_v9_20260912"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid"
    cases = main_root / "bilingual_holdout.json"

    def eval_core(checkpoint: Path, tag: str, tokens: int) -> dict:
        val = report / f"{tag}_validation.json"
        dev = report / f"{tag}_dev.json"
        run("eval_validation_loss350.py", "--checkpoint", checkpoint, "--valid-data", valid, "--out", val,
            "--seq", 512, "--batch", 4, "--batches", 32, "--seed", 351337)
        run("eval_recovery_dev_v2.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer, "--out", dev,
            "--expected-tokens", tokens)
        out = {"validation": load(val), "dev": load(dev)}
        sha = file_sha(checkpoint)
        for obj in out.values():
            if obj.get("checkpoint_sha256") != sha or obj.get("checkpoint_processed_tokens") != tokens:
                raise ValueError("core evaluation identity mismatch")
        return out

    def eval_bilingual(checkpoint: Path, tag: str, tokens: int, dev: dict) -> dict:
        bi = report / f"{tag}_bilingual.json"
        run("eval_bilingual_holdout_v1.py", "--checkpoint", checkpoint, "--tokenizer", tokenizer,
            "--cases", cases, "--out", bi, "--expected-tokens", tokens)
        obj = load(bi)
        if obj.get("checkpoint_sha256") != file_sha(checkpoint) or obj.get("checkpoint_processed_tokens") != tokens:
            raise ValueError("bilingual evaluation identity mismatch")
        if obj.get("tokenizer_sha256") != dev.get("tokenizer_sha256"):
            raise ValueError("tokenizer identity mismatch")
        return obj

    baseline_core = eval_core(anchor, "anchor", ANCHOR_TOKENS)
    baseline_bi = eval_bilingual(anchor, "anchor", ANCHOR_TOKENS, baseline_core["dev"])
    baseline = {**baseline_core, "bilingual": baseline_bi}

    def train(tag: str, lr: float, tokens: int) -> tuple[Path, dict, dict]:
        out = scratch / tag
        cmd = [sys.executable, "-u", str(Path(__file__).parent / "train_guarded_v9.py"),
               "--anchor", str(anchor), "--anchor-sha", ANCHOR_SHA,
               "--main-data", str(main_data), "--replay", str(replay), "--out", str(out),
               "--mode", "replay_only", "--new-tokens", str(tokens), "--batch-size", "4",
               "--microbatches", "12", "--main-pack", str(PACK), "--compile-mode", "default",
               "--lr", str(lr), "--fast-replay-only"]
        print(json.dumps({"event": "train_start", "tag": tag, "pack": PACK, "lr": lr, "tokens": tokens}), flush=True)
        subprocess.run(cmd, check=True)
        summary = load(out / "summary.json")
        target = ANCHOR_TOKENS + tokens
        if not summary.get("complete") or summary.get("processed_tokens") != target or not summary.get("frozen_weights_unchanged"):
            raise ValueError("training contract failed")
        rows = [r for r in train_rows(out / "metrics.jsonl")[5:] if r.get("tokens_per_update") == TOKENS_PER_UPDATE]
        rates = [float(r["tokens_per_second"]) for r in rows]
        throughput = {"median_tps": statistics.median(rates), "mean_tps": statistics.fmean(rates),
                      "p10_tps": p10(rates), "peak_tps": max(rates)} if rates else {}
        shutil.copy2(out / "summary.json", report / f"{tag}_summary.json")
        shutil.copy2(out / "metrics.jsonl", report / f"{tag}_metrics.jsonl")
        return out / "resume.pt", summary, throughput

    candidates: list[dict] = []
    selected = None

    def evaluate_family(raw: Path, lr: float, tokens: int, stage: str, throughput: dict) -> list[dict]:
        target = ANCHOR_TOKENS + tokens
        family = []
        for alpha in ALPHAS:
            tag = f"{stage}_lr{str(lr).replace('.', 'p')}_a{str(alpha).replace('.', 'p')}"
            snap = scratch / f"{tag}.pt"
            run("interpolate_recovery_checkpoint.py", "--anchor", anchor, "--candidate", raw, "--out", snap, "--alpha", alpha)
            core = eval_core(snap, tag, target)
            gate = report / f"{tag}_legacy_gate.json"
            run("select_recovery_lr_pilot.py", "--baseline-dev", report / "anchor_dev.json",
                "--baseline-validation", report / "anchor_validation.json", "--candidate", tag, lr,
                report / f"{tag}_dev.json", report / f"{tag}_validation.json", "--out", gate)
            legacy = load(gate)
            legacy_ok = bool(legacy.get("selection_allowed"))
            bi = None
            bi_ok = False
            if legacy_ok:
                bi = eval_bilingual(snap, tag, target, core["dev"])
                bi_ok = strict_bilingual(baseline_bi, bi, core["dev"])
            improved = core["validation"]["validation_loss"] < baseline["validation"]["validation_loss"] or any(
                core["dev"]["categories"][k]["accuracy_pct"] > baseline["dev"]["categories"][k]["accuracy_pct"]
                for k in baseline["dev"]["categories"])
            if bi is not None:
                improved = improved or bi["overall_accuracy_pct"] > baseline_bi["overall_accuracy_pct"]
            passed = legacy_ok and bi_ok and improved
            row = {"stage": stage, "tokens": tokens, "lr": lr, "alpha": alpha, "effective_lr": lr * alpha,
                   "pass": passed, "legacy_gate": legacy_ok,
                   "legacy_failures": legacy.get("candidates", [{}])[0].get("failures", []),
                   "bilingual_gate": bi_ok if bi is not None else None, "strict_improvement": improved,
                   "validation_loss": core["validation"]["validation_loss"],
                   "overall_pct": core["dev"]["overall_accuracy_pct"],
                   "categories": {k: v["accuracy_pct"] for k, v in core["dev"]["categories"].items()},
                   "bilingual_pct": bi["overall_accuracy_pct"] if bi else None,
                   "de_to_en_pct": bi["directions"]["de_to_en"]["accuracy_pct"] if bi else None,
                   "en_to_de_pct": bi["directions"]["en_to_de"]["accuracy_pct"] if bi else None,
                   "checkpoint_sha256": file_sha(snap), "path": str(snap), "throughput": throughput}
            family.append(row); candidates.append(row)
            (report / "recovery_progress.json").write_text(json.dumps(candidates, indent=2), encoding="utf-8")
            print(json.dumps({"event": "candidate", **row}), flush=True)
            if not passed:
                snap.unlink(missing_ok=True)
        return family

    # Primary search: same data/order and Pack 6, only LR changes. Core gates are evaluated
    # before the expensive bilingual holdout, so failed candidates are rejected quickly.
    primary_passing = []
    raw_runs = []
    for lr in LRS:
        tag = f"primary2m_lr{str(lr).replace('.', 'p')}"
        raw, summary, throughput = train(tag, lr, PRIMARY_TOKENS)
        raw_runs.append({"stage": "primary2m", "lr": lr, "tokens": PRIMARY_TOKENS,
                         "checkpoint_sha256": file_sha(raw), "throughput": throughput})
        family = evaluate_family(raw, lr, PRIMARY_TOKENS, "primary2m", throughput)
        primary_passing.extend([c for c in family if c["pass"]])
        if primary_passing:
            break

    if primary_passing:
        selected = min(primary_passing, key=lambda c: (c["validation_loss"], -c["bilingual_pct"], -c["overall_pct"], -c["effective_lr"]))
    else:
        # Autonomous fallback: reduce the continuation horizon to 1M rather than corrupting
        # the anchor. This still makes safe forward progress if a 2M block is too aggressive.
        fallback_passing = []
        for lr in LRS:
            tag = f"fallback1m_lr{str(lr).replace('.', 'p')}"
            raw, summary, throughput = train(tag, lr, FALLBACK_TOKENS)
            raw_runs.append({"stage": "fallback1m", "lr": lr, "tokens": FALLBACK_TOKENS,
                             "checkpoint_sha256": file_sha(raw), "throughput": throughput})
            family = evaluate_family(raw, lr, FALLBACK_TOKENS, "fallback1m", throughput)
            fallback_passing.extend([c for c in family if c["pass"]])
            if fallback_passing:
                break
        if fallback_passing:
            selected = min(fallback_passing, key=lambda c: (c["validation_loss"], -c["bilingual_pct"], -c["overall_pct"], -c["effective_lr"]))

    promoted = None
    if selected:
        destdir = assets / "runs" / f"neral350_scalev2_{a.request_id}_selected"
        if destdir.exists():
            raise FileExistsError("promotion destination exists")
        destdir.mkdir(parents=True)
        dest = destdir / "resume.pt"
        shutil.copy2(selected["path"], dest)
        if file_sha(dest) != selected["checkpoint_sha256"]:
            raise RuntimeError("promotion hash mismatch")
        promoted = str(dest)

    result = {"schema": 2, "controller": "neral350-scale-v2-lr-recovery", "request_id": a.request_id,
              "anchor_sha256": ANCHOR_SHA, "anchor_tokens": ANCHOR_TOKENS, "pack": PACK,
              "learning_rates": list(LRS), "alphas": list(ALPHAS), "raw_runs": raw_runs,
              "candidates": candidates, "success": selected is not None, "selected": selected,
              "final_safe_tokens": ANCHOR_TOKENS + int(selected["tokens"]) if selected else ANCHOR_TOKENS,
              "promoted_checkpoint": promoted, "safe_anchor_preserved_if_failure": True,
              "benchmarks_used_for_training": False, "final_audit_pending": True,
              "next_if_success": "Use the selected LR/effective step for a guarded 5M block, then grow to 10M/25M/50M/100M blocks."}
    (report / "selection.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"event": "final_selection", **result}), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
