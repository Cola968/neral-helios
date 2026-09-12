from __future__ import annotations

"""Fail-closed selector for ARCTUS V6 cross-lingual recovery."""

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-holdout", required=True)
    ap.add_argument("--candidate", action="append", nargs=4, metavar=("NAME", "GATE", "HOLDOUT", "OUT_DIR"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline = load(args.baseline_holdout)
    if baseline.get("suite") != "arctus-bilingual-holdout-v1":
        raise RuntimeError("wrong bilingual baseline suite")
    base_overall = float(baseline["overall_accuracy_pct"])
    base_dirs = {
        name: float(data["accuracy_pct"])
        for name, data in baseline["directions"].items()
    }
    rows = []
    for name, gate_path, holdout_path, out_dir in args.candidate or []:
        gate = load(gate_path)
        holdout = load(holdout_path)
        failures = []
        if gate.get("selection_allowed") is not True:
            failures.append("immutable_dev_or_validation_gate")
        if holdout.get("suite") != baseline.get("suite") or holdout.get("cases_sha256") != baseline.get("cases_sha256"):
            failures.append("holdout_identity_mismatch")
        overall = float(holdout.get("overall_accuracy_pct", -1))
        if overall < base_overall:
            failures.append(f"holdout_overall:{overall}<{base_overall}")
        dirs = {
            key: float(value["accuracy_pct"])
            for key, value in (holdout.get("directions") or {}).items()
        }
        for direction, floor in base_dirs.items():
            value = dirs.get(direction, -1.0)
            if value < floor:
                failures.append(f"{direction}:{value}<{floor}")
        candidate_info = (gate.get("candidates") or [{}])[0]
        rows.append(
            {
                "name": name,
                "pass": not failures,
                "failures": failures,
                "validation_loss": float(candidate_info.get("validation_loss", 1e9)),
                "old_dev_overall_accuracy_pct": float(candidate_info.get("overall_accuracy_pct", -1)),
                "old_dev_bilingual_accuracy_pct": float(
                    (candidate_info.get("categories") or {}).get("bilingual", -1)
                ),
                "holdout_overall_accuracy_pct": overall,
                "holdout_directions": dirs,
                "out_dir": out_dir,
            }
        )

    passing = [row for row in rows if row["pass"]]
    passing.sort(
        key=lambda row: (
            row["validation_loss"],
            -row["holdout_overall_accuracy_pct"],
            row["name"],
        )
    )
    selected = passing[0]["name"] if passing else None
    result = {
        "schema": 1,
        "purpose": "development-only V6 cross-lingual recovery selection",
        "baseline_holdout_overall_accuracy_pct": base_overall,
        "baseline_holdout_directions": base_dirs,
        "selected": selected,
        "selection_allowed": selected is not None,
        "candidates": rows,
        "policy": (
            "Fail closed: immutable 140-case dev categories, overall score, fixed validation "
            "loss, and independent bilingual holdout overall/direction scores may not regress."
        ),
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
