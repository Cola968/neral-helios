"""Bounded autonomous continuation for Neral 350M.

Auto-Learn V1 performs a small continuation segment, evaluates several conservative
anchor interpolations, promotes only a candidate that passes all immutable development
floors, and then optionally repeats from the newly accepted anchor. Training never sees
benchmark examples. This is deliberately bounded rather than an unguarded endless loop.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

from recovery_identity import file_sha

INITIAL_ANCHOR_SHA = "d471dfd0d4f866e18571f2d52cdfdc99abbc533bfbb33973642b8b1f7ad3e8af"
INITIAL_ANCHOR_TOKENS = 1_008_003_584
ALPHAS = (0.10, 0.075, 0.05)


def execute(script: str, *args: object) -> None:
    cmd = [sys.executable, "-u", str(Path(__file__).parent / script), *map(str, args)]
    print(json.dumps({"event": "exec", "argv": cmd}), flush=True)
    subprocess.run(cmd, check=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
    ap.add_argument("--max-rounds", type=int, required=True)
    ap.add_argument("--new-tokens-per-round", type=int, required=True)
    a = ap.parse_args()

    if not 1 <= a.max_rounds <= 3:
        raise ValueError("Auto-Learn V1 is capped at 3 rounds")
    if a.new_tokens_per_round != 1_000_000:
        raise ValueError("Auto-Learn V1 currently permits exactly 1,000,000 tokens per round")
    if not a.request_id or len(a.request_id) > 64 or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-" for c in a.request_id):
        raise ValueError("unsafe request id")

    assets = Path(a.assets).resolve()
    report = Path(a.report).resolve()
    scratch = Path(a.scratch).resolve()
    if report.exists() and any(report.iterdir()):
        raise FileExistsError("report directory is not empty")
    if scratch.exists():
        raise FileExistsError("scratch directory already exists; refusing overwrite")
    report.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True)

    initial_anchor = assets / "runs/neral350_v10_arctus-guarded-v10-salvage-20260912-r5_selected/resume.pt"
    if not initial_anchor.is_file() or file_sha(initial_anchor) != INITIAL_ANCHOR_SHA:
        raise ValueError("promoted V10 anchor missing or checksum changed")

    tokenizer = assets / "frontier350_assets/tokenizer.json"
    main_data = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05"
    replay = assets / "frontier350_replay_v9_20260912"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2/valid"
    cases = main_data / "bilingual_holdout.json"
    for name, path in (("main_manifest", main_data / "manifest.json"), ("replay_manifest", replay / "manifest.json")):
        shutil.copyfile(path, report / f"{name}.json")

    def evaluate(checkpoint: Path, tag: str, tokens: int) -> dict:
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
            "--expected-tokens", tokens,
        )
        execute(
            "eval_bilingual_holdout_v1.py",
            "--checkpoint", checkpoint,
            "--tokenizer", tokenizer,
            "--cases", cases,
            "--out", bilingual,
            "--expected-tokens", tokens,
        )
        result = {"validation": load_json(validation), "dev": load_json(dev), "bilingual": load_json(bilingual)}
        sha = file_sha(checkpoint)
        for key in result.values():
            if key.get("checkpoint_sha256") != sha or key.get("checkpoint_processed_tokens") != tokens:
                raise ValueError("evaluation identity mismatch")
        return result

    current_anchor = initial_anchor
    current_tokens = INITIAL_ANCHOR_TOKENS
    base_tag = "anchor_r00"
    baseline = evaluate(current_anchor, base_tag, current_tokens)
    history: list[dict] = []
    stopped_reason = "max_rounds_reached"

    for round_number in range(1, a.max_rounds + 1):
        round_tag = f"r{round_number:02d}"
        target_tokens = current_tokens + a.new_tokens_per_round
        round_root = scratch / round_tag
        raw_out = round_root / "raw"
        round_root.mkdir(parents=True)

        train_common = [
            "--anchor", current_anchor,
            "--anchor-sha", file_sha(current_anchor),
            "--main-data", main_data / "train",
            "--replay", replay,
            "--out", raw_out,
            "--mode", "replay_only",
            "--new-tokens", a.new_tokens_per_round,
        ]
        # Two-update probe first. The exact same run is then resumed, so probe work is not discarded.
        execute("train_guarded_v9.py", *train_common, "--max-updates", 2)
        smoke = load_json(raw_out / "summary.json")
        shutil.copyfile(raw_out / "summary.json", report / f"{round_tag}_gpu_probe.json")
        if smoke.get("new_tokens") != 49_152 or smoke.get("peak_allocated_bytes", 2**63) > 10 * 1024**3:
            raise RuntimeError("GPU probe outside validated memory/token budget")
        execute("train_guarded_v9.py", *train_common, "--resume", raw_out / "resume.pt")
        summary = load_json(raw_out / "summary.json")
        if not summary.get("complete") or summary.get("new_tokens") != a.new_tokens_per_round or summary.get("processed_tokens") != target_tokens:
            raise ValueError("training round did not complete the bounded budget")
        shutil.copyfile(raw_out / "summary.json", report / f"{round_tag}_train_summary.json")
        shutil.copyfile(raw_out / "metrics.jsonl", report / f"{round_tag}_train_metrics.jsonl")

        raw_checkpoint = raw_out / "resume.pt"
        candidates: list[dict] = []
        round_records: list[dict] = []
        for alpha in ALPHAS:
            alpha_name = str(alpha).replace(".", "p")
            tag = f"{round_tag}_a{alpha_name}"
            snapshot = round_root / f"{tag}.pt"
            execute(
                "interpolate_recovery_checkpoint.py",
                "--anchor", current_anchor,
                "--candidate", raw_checkpoint,
                "--out", snapshot,
                "--alpha", alpha,
            )
            measured = evaluate(snapshot, tag, target_tokens)
            gate_path = report / f"{tag}_legacy_gate.json"
            execute(
                "select_recovery_lr_pilot.py",
                "--baseline-dev", report / f"{base_tag}_dev.json",
                "--baseline-validation", report / f"{base_tag}_validation.json",
                "--candidate", tag, 7.5e-5,
                report / f"{tag}_dev.json",
                report / f"{tag}_validation.json",
                "--out", gate_path,
            )
            legacy = load_json(gate_path)
            bilingual_ok = strict_bilingual(baseline["bilingual"], measured["bilingual"], measured["dev"])
            improved = (
                measured["validation"]["validation_loss"] < baseline["validation"]["validation_loss"]
                or measured["bilingual"]["overall_accuracy_pct"] > baseline["bilingual"]["overall_accuracy_pct"]
                or any(
                    measured["dev"]["categories"][category]["accuracy_pct"]
                    > baseline["dev"]["categories"][category]["accuracy_pct"]
                    for category in baseline["dev"]["categories"]
                )
            )
            passed = bool(legacy.get("selection_allowed")) and bilingual_ok and improved
            row = {
                "round": round_number,
                "tag": tag,
                "alpha": alpha,
                "checkpoint_sha256": file_sha(snapshot),
                "processed_tokens": target_tokens,
                "legacy_gate": bool(legacy.get("selection_allowed")),
                "legacy_failures": legacy.get("candidates", [{}])[0].get("failures", []),
                "bilingual_gate": bilingual_ok,
                "strict_improvement": improved,
                "pass": passed,
                "validation_loss": measured["validation"]["validation_loss"],
                "overall_pct": measured["dev"]["overall_accuracy_pct"],
                "categories": {k: v["accuracy_pct"] for k, v in measured["dev"]["categories"].items()},
                "bilingual_pct": measured["bilingual"]["overall_accuracy_pct"],
                "de_to_en_pct": measured["bilingual"]["directions"]["de_to_en"]["accuracy_pct"],
                "en_to_de_pct": measured["bilingual"]["directions"]["en_to_de"]["accuracy_pct"],
            }
            round_records.append(row)
            if passed:
                candidates.append({"row": row, "path": snapshot, "measured": measured, "tag": tag})
            else:
                snapshot.unlink()
            (report / "progress.json").write_text(json.dumps(history + [{"round": round_number, "candidates": round_records}], indent=2))

        if not candidates:
            stopped_reason = f"round_{round_number}_no_candidate_passed"
            history.append({"round": round_number, "accepted": False, "candidates": round_records})
            shutil.rmtree(round_root, ignore_errors=True)
            break

        selected = min(
            candidates,
            key=lambda c: (
                c["row"]["validation_loss"],
                -c["row"]["bilingual_pct"],
                -c["row"]["alpha"],
            ),
        )
        destination_dir = assets / "runs" / f"neral350_autolearn_{a.request_id}_{round_tag}_selected"
        if destination_dir.exists():
            raise FileExistsError("promotion destination already exists")
        destination_dir.mkdir(parents=True)
        destination = destination_dir / "resume.pt"
        shutil.copy2(selected["path"], destination)
        if file_sha(destination) != selected["row"]["checkpoint_sha256"]:
            raise RuntimeError("promoted checkpoint hash mismatch")

        history.append(
            {
                "round": round_number,
                "accepted": True,
                "selected": selected["row"],
                "promoted_checkpoint": str(destination),
                "candidates": round_records,
            }
        )
        baseline = selected["measured"]
        base_tag = selected["tag"]
        current_anchor = destination
        current_tokens = target_tokens
        shutil.rmtree(round_root, ignore_errors=True)
        (report / "progress.json").write_text(json.dumps(history, indent=2))

    result = {
        "schema": 1,
        "controller": "neral350-auto-learn-v1",
        "request_id": a.request_id,
        "initial_checkpoint_sha256": INITIAL_ANCHOR_SHA,
        "start_tokens": INITIAL_ANCHOR_TOKENS,
        "final_tokens": current_tokens,
        "rounds_requested": a.max_rounds,
        "rounds_accepted": sum(1 for r in history if r.get("accepted")),
        "stopped_reason": stopped_reason,
        "final_checkpoint": str(current_anchor),
        "final_checkpoint_sha256": file_sha(current_anchor),
        "history": history,
        "benchmarks_used_for_training": False,
        "final_audit_pending": True,
        "note": "Development gates are repeatedly used for bounded selection; keep a separate untouched release audit.",
    }
    (report / "selection.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    main()
