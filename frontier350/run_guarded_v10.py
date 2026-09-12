"""V10: conservatively salvage the retained 1.008B V9 confirmation.

No new training occurs here. Only evaluation snapshots are created. At most one
selected snapshot plus one temporary candidate may exist at once; rejected
snapshots are deleted immediately. Benchmark material remains evaluation-only.
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
MIN_FREE_BYTES = 2 * 1024**3


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


def free_bytes(path):
    return shutil.disk_usage(path).free


def require_space(path, stage):
    free = free_bytes(path)
    if free < MIN_FREE_BYTES:
        raise RuntimeError(f"{stage}: only {free} free bytes; V10 requires >= {MIN_FREE_BYTES}")
    return free


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
    for name in ("assets", "report", "request-id", "v9-request-id"):
        ap.add_argument("--" + name, required=True)
    args = ap.parse_args()
    assets = Path(args.assets).resolve()
    report = Path(args.report).resolve()
    report.mkdir(parents=True, exist_ok=True)
    require_space(assets, "startup")

    v9_root = assets / "runs" / ("neral350_v9_" + args.v9_request_id)
    anchor = v9_root / "replay_only_a010.pt"
    candidate = v9_root / "confirmation" / "resume.pt"
    if file_sha(anchor) != ANCHOR_SHA:
        raise ValueError("V9 1.006B anchor identity changed")
    if file_sha(candidate) != CANDIDATE_SHA:
        raise ValueError("V9 full 1.008B confirmation candidate identity changed")

    tok = assets / "frontier350_assets" / "tokenizer.json"
    main_data = assets / "frontier350_safe_v8_arctus-safe-v8-20260912-r1_xling05"
    valid = assets / "frontier350_bpe_recovery_pilot_arctus-recovery-lr-20260911-r2" / "valid"
    cases = main_data / "bilingual_holdout.json"
    run_root = assets / "runs" / ("neral350_v10_" + args.request_id)
    if run_root.exists():
        raise FileExistsError("V10 run already exists; never overwrite measured snapshots")
    run_root.mkdir(parents=True)
    temp_path = run_root / "candidate_eval.pt"
    selected_path = run_root / "selected.pt"

    def execute(script, *cmd_args):
        env = os.environ.copy()
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        subprocess.run(
            [sys.executable, "-u", str(Path(__file__).parent / script), *map(str, cmd_args)],
            check=True,
            env=env,
        )

    def evaluate(ck, tag, tokens):
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
            "--expected-tokens", tokens,
        )
        execute(
            "eval_bilingual_holdout_v1.py",
            "--checkpoint", ck,
            "--tokenizer", tok,
            "--cases", cases,
            "--out", report / f"{tag}_bilingual.json",
            "--expected-tokens", tokens,
        )
        return {
            key: json.loads((report / f"{tag}_{key}.json").read_text())
            for key in ("dev", "validation", "bilingual")
        }

    baseline = evaluate(anchor, "v9_anchor", ANCHOR_TOKENS)
    rows = []
    best_row = None

    for alpha in ALPHAS:
        tag = "v10_a" + str(alpha).replace(".", "p")
        if temp_path.exists():
            raise RuntimeError("stale V10 temporary snapshot exists")
        require_space(assets, f"before {tag}")
        keep_temp = False
        try:
            interpolate(anchor, candidate, temp_path, alpha)
            result = evaluate(temp_path, tag, CANDIDATE_TOKENS)
            gate_path = report / f"{tag}_gate.json"
            execute(
                "select_recovery_lr_pilot.py",
                "--baseline-dev", report / "v9_anchor_dev.json",
                "--baseline-validation", report / "v9_anchor_validation.json",
                "--candidate", tag, 7.5e-5, report / f"{tag}_dev.json", report / f"{tag}_validation.json",
                "--out", gate_path,
            )
            legacy = json.loads(gate_path.read_text())
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
                "validation_loss": result["validation"]["validation_loss"],
                "overall_pct": result["dev"]["overall_accuracy_pct"],
                "reasoning_pct": result["dev"]["categories"]["reasoning"]["accuracy_pct"],
                "science_pct": result["dev"]["categories"]["science"]["accuracy_pct"],
                "bilingual_pct": result["bilingual"]["overall_accuracy_pct"],
                "checkpoint_sha256": file_sha(temp_path),
                "free_bytes_after_eval": free_bytes(assets),
            }
            rows.append(row)
            (report / "progress.json").write_text(json.dumps(rows, indent=2))
            if passed and better(row, best_row):
                if selected_path.exists():
                    selected_path.unlink()
                temp_path.replace(selected_path)
                keep_temp = True
                best_row = row
        finally:
            if temp_path.exists() and not keep_temp:
                temp_path.unlink()

    selected_copy_sha = None
    if best_row is not None:
        if not selected_path.exists():
            raise RuntimeError("selected metadata exists without selected checkpoint")
        promoted_root = assets / "runs" / ("neral350_v10_" + args.request_id + "_selected")
        if promoted_root.exists():
            raise FileExistsError("V10 selected destination already exists")
        promoted_root.mkdir(parents=True)
        promoted = promoted_root / "resume.pt"
        selected_path.replace(promoted)
        selected_copy_sha = file_sha(promoted)
        if selected_copy_sha != best_row["checkpoint_sha256"]:
            raise RuntimeError("selected checkpoint hash changed during promotion")

    summary = {
        "schema": 2,
        "source_anchor_tokens": ANCHOR_TOKENS,
        "candidate_tokens": CANDIDATE_TOKENS,
        "anchor_sha256": ANCHOR_SHA,
        "candidate_sha256": CANDIDATE_SHA,
        "alphas": list(ALPHAS),
        "records": rows,
        "selection_passed": best_row is not None,
        "selected": best_row,
        "selected_copy_sha256": selected_copy_sha,
        "next_action": "bounded_plus_2m_from_selected" if best_row else "reasoning_retention_retrain",
        "promoted_to_chat": False,
        "final_audit_pending": True,
        "benchmarks_used_for_training": False,
        "free_bytes_final": free_bytes(assets),
        "note": "Development gates only; bilingual suite is repeatedly used and is not an untouched release holdout.",
    }
    (report / "selection.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
