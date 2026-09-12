"""V10 salvage audit for the retained 1.008B V9 confirmation.

No new training occurs. The already-measured 1.006B anchor evidence is reused only
after binding it to the immutable anchor SHA. Evaluation snapshots live on a large
scratch filesystem (normally D:) so WSL/Windows transient disk pressure on C: cannot
invalidate the experiment. Benchmark material remains evaluation-only.
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

from interpolate_recovery_checkpoint import interpolate
from recovery_identity import file_sha

ANCHOR_SHA = "16656c86ff7846a39c3f75d7ddc8a629efe31245f4ed3d6fa58f1867f63b059e"
ANCHOR_TOKENS = 1_006_003_584
CANDIDATE_SHA = "72e423eae5026c26eac82a2aed8e9c0b38ad8bcb1c83e1dadd52a34fd6960a2c"
CANDIDATE_TOKENS = 1_008_003_584
ALPHAS = (0.075, 0.05, 0.025, 0.0125)
MIN_SCRATCH_FREE_BYTES = 8 * 1024**3
BASELINE_DIR = Path(__file__).parent / "v10_baseline_evidence"


def read_json(path):
    obj = json.loads(Path(path).read_text(encoding="utf8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected object: {path}")
    return obj


def strict_bilingual(base, candidate, dev):
    for obj in (base, candidate):
        if obj.get("cases") != 400 or set(obj.get("directions", {})) != {"de_to_en", "en_to_de"}:
            raise ValueError("incomplete bilingual evidence")
        total = correct = 0
        for direction in obj["directions"].values():
            n, k = direction["total"], direction["correct"]
            if type(n) is not int or type(k) is not int or n != 200 or not 0 <= k <= n:
                raise ValueError("invalid bilingual counts")
            if isinstance(direction["accuracy_pct"], bool) or not math.isfinite(direction["accuracy_pct"]):
                raise ValueError("invalid bilingual percentage")
            if abs(direction["accuracy_pct"] - 100 * k / n) > 1e-9:
                raise ValueError("inconsistent direction score")
            total += n
            correct += k
        if (
            isinstance(obj["overall_accuracy_pct"], bool)
            or not math.isfinite(obj["overall_accuracy_pct"])
            or abs(obj["overall_accuracy_pct"] - 100 * correct / total) > 1e-9
        ):
            raise ValueError("inconsistent total score")
    for key in ("cases_sha256", "tokenizer_sha256", "protocol_sha256"):
        if not base.get(key) or base[key] != candidate.get(key):
            raise ValueError("bilingual protocol/data changed")
    for key in ("checkpoint_sha256", "checkpoint_processed_tokens", "tokenizer_sha256"):
        if candidate[key] != dev[key]:
            raise ValueError("mixed candidate checkpoint evidence")
    return candidate["overall_accuracy_pct"] >= base["overall_accuracy_pct"] and all(
        candidate["directions"][d]["accuracy_pct"] >= base["directions"][d]["accuracy_pct"]
        for d in base["directions"]
    )


def better(row, incumbent):
    if incumbent is None:
        return True
    return (
        row["validation_loss"],
        -row["bilingual_pct"],
        -row["alpha"],
    ) < (
        incumbent["validation_loss"],
        -incumbent["bilingual_pct"],
        -incumbent["alpha"],
    )


def main():
    ap = argparse.ArgumentParser()
    for name in ("assets", "report", "request-id", "v9-request-id", "scratch"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    assets = Path(args.assets).resolve()
    report = Path(args.report).resolve()
    scratch = Path(args.scratch).resolve()
    report.mkdir(parents=True, exist_ok=True)
    if scratch.exists() and any(scratch.iterdir()):
        raise FileExistsError("V10 scratch is non-empty; never overwrite staged checkpoints")
    scratch.mkdir(parents=True, exist_ok=True)
    scratch_free_start = shutil.disk_usage(scratch).free
    if scratch_free_start < MIN_SCRATCH_FREE_BYTES:
        raise RuntimeError(f"V10 scratch has only {scratch_free_start} free bytes")

    v9_root = assets / "runs" / ("neral350_v9_" + args.v9_request_id)
    anchor = v9_root / "replay_only_a010.pt"
    candidate = v9_root / "confirmation" / "resume.pt"
    if file_sha(anchor) != ANCHOR_SHA:
        raise ValueError("V9 1.006B anchor identity changed")
    if file_sha(candidate) != CANDIDATE_SHA:
        raise ValueError("V9 full 1.008B confirmation candidate identity changed")

    baseline_paths = {
        "dev": BASELINE_DIR / "v9_anchor_dev.json",
        "validation": BASELINE_DIR / "v9_anchor_validation.json",
        "bilingual": BASELINE_DIR / "v9_anchor_bilingual.json",
    }
    baseline = {key: read_json(path) for key, path in baseline_paths.items()}
    for key, obj in baseline.items():
        if obj.get("checkpoint_sha256") != ANCHOR_SHA or int(obj.get("checkpoint_processed_tokens", -1)) != ANCHOR_TOKENS:
            raise ValueError(f"stored {key} baseline is not bound to the accepted V9 anchor")
    if baseline["dev"].get("tokenizer_sha256") != baseline["validation"].get("tokenizer_sha256"):
        raise ValueError("stored anchor evidence tokenizer mismatch")

    tok = assets / "frontier350_assets" / "tokenizer.json"
    main_data = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2" / "valid"
    cases = main_data / "bilingual_holdout.json"
    if file_sha(tok) != baseline["dev"]["tokenizer_sha256"]:
        raise ValueError("runtime tokenizer no longer matches stored anchor evidence")
    if file_sha(cases) != baseline["bilingual"]["cases_sha256"]:
        raise ValueError("bilingual cases changed since V9 anchor measurement")

    temp_path = scratch / "candidate_eval.pt"
    selected_path = scratch / "selected.pt"

    def execute(script, *cmd_args):
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        env.setdefault("CUDA_MODULE_LOADING", "LAZY")
        subprocess.run(
            [sys.executable, "-u", str(Path(__file__).parent / script), *map(str, cmd_args)],
            check=True,
            env=env,
        )

    def evaluate(ck, tag):
        execute(
            "eval_validation_loss350.py",
            "--checkpoint", ck,
            "--valid-data", valid,
            "--out", report / f"{tag}_validation.json",
            "--seq", 512,
            "--batch", 4,
            "--batches", 32,
            "--seed", 351337,
        )
        execute(
            "eval_recovery_dev_v2.py",
            "--checkpoint", ck,
            "--tokenizer", tok,
            "--out", report / f"{tag}_dev.json",
            "--expected-tokens", CANDIDATE_TOKENS,
        )
        execute(
            "eval_bilingual_holdout_v1.py",
            "--checkpoint", ck,
            "--tokenizer", tok,
            "--cases", cases,
            "--out", report / f"{tag}_bilingual.json",
            "--expected-tokens", CANDIDATE_TOKENS,
        )
        return {key: read_json(report / f"{tag}_{key}.json") for key in ("dev", "validation", "bilingual")}

    rows = []
    best_row = None
    for alpha in ALPHAS:
        tag = "v10_a" + str(alpha).replace(".", "p")
        if temp_path.exists():
            raise RuntimeError("stale V10 temporary snapshot exists")
        if shutil.disk_usage(scratch).free < MIN_SCRATCH_FREE_BYTES:
            raise RuntimeError("scratch space unexpectedly fell below V10 floor")
        moved_to_selected = False
        try:
            interpolate(anchor, candidate, temp_path, alpha)
            result = evaluate(temp_path, tag)
            gate_path = report / f"{tag}_gate.json"
            execute(
                "select_recovery_lr_pilot.py",
                "--baseline-dev", baseline_paths["dev"],
                "--baseline-validation", baseline_paths["validation"],
                "--candidate", tag, 7.5e-5, report / f"{tag}_dev.json", report / f"{tag}_validation.json",
                "--out", gate_path,
            )
            legacy = read_json(gate_path)
            bilingual = strict_bilingual(baseline["bilingual"], result["bilingual"], result["dev"])
            improved = (
                result["validation"]["validation_loss"] < baseline["validation"]["validation_loss"]
                or result["bilingual"]["overall_accuracy_pct"] > baseline["bilingual"]["overall_accuracy_pct"]
                or any(
                    result["dev"]["categories"][c]["accuracy_pct"]
                    > baseline["dev"]["categories"][c]["accuracy_pct"]
                    for c in baseline["dev"]["categories"]
                )
            )
            passed = legacy["selection_allowed"] and bilingual and improved
            row = {
                "tag": tag,
                "alpha": alpha,
                "legacy_gate": legacy["selection_allowed"],
                "bilingual_gate": bilingual,
                "strict_improvement": improved,
                "pass": passed,
                "legacy_failures": legacy["candidates"][0]["failures"],
                "validation_loss": result["validation"]["validation_loss"],
                "overall_pct": result["dev"]["overall_accuracy_pct"],
                "reasoning_pct": result["dev"]["categories"]["reasoning"]["accuracy_pct"],
                "science_pct": result["dev"]["categories"]["science"]["accuracy_pct"],
                "bilingual_pct": result["bilingual"]["overall_accuracy_pct"],
                "de_to_en_pct": result["bilingual"]["directions"]["de_to_en"]["accuracy_pct"],
                "en_to_de_pct": result["bilingual"]["directions"]["en_to_de"]["accuracy_pct"],
                "checkpoint_sha256": file_sha(temp_path),
                "scratch_free_bytes_after_eval": shutil.disk_usage(scratch).free,
                "assets_free_bytes_after_eval": shutil.disk_usage(assets).free,
            }
            rows.append(row)
            (report / "progress.json").write_text(json.dumps(rows, indent=2), encoding="utf8")
            if passed and better(row, best_row):
                if selected_path.exists():
                    selected_path.unlink()
                temp_path.replace(selected_path)
                moved_to_selected = True
                best_row = row
        finally:
            if temp_path.exists() and not moved_to_selected:
                temp_path.unlink()

    selected_sha = file_sha(selected_path) if best_row is not None else None
    if best_row is not None and selected_sha != best_row["checkpoint_sha256"]:
        raise RuntimeError("staged selected checkpoint hash mismatch")
    summary = {
        "schema": 3,
        "source_anchor_tokens": ANCHOR_TOKENS,
        "candidate_tokens": CANDIDATE_TOKENS,
        "anchor_sha256": ANCHOR_SHA,
        "candidate_sha256": CANDIDATE_SHA,
        "baseline_evidence": {key: str(path.relative_to(Path(__file__).parent)) for key, path in baseline_paths.items()},
        "alphas": list(ALPHAS),
        "records": rows,
        "selection_passed": best_row is not None,
        "selected": best_row,
        "selected_staged_sha256": selected_sha,
        "next_action": "promote_staged_then_bounded_confirmation" if best_row else "reasoning_retention_retrain",
        "promoted_to_chat": False,
        "final_audit_pending": True,
        "benchmarks_used_for_training": False,
        "scratch_free_bytes_start": scratch_free_start,
        "scratch_free_bytes_final": shutil.disk_usage(scratch).free,
        "assets_free_bytes_final": shutil.disk_usage(assets).free,
        "note": "Development gates only; bilingual suite is repeatedly used and is not an untouched release holdout.",
    }
    (report / "selection.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
