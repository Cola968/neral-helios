from __future__ import annotations

"""Select a recovery LR candidate without touching frozen release benchmarks.

The development selector is intentionally strict: every 140-case category and overall
accuracy must be at least the immutable 1B development baseline, and fixed pilot
validation loss must not worsen. If no candidate satisfies all rules, nothing is selected.
"""

import argparse
import json
import math
from pathlib import Path


def read(path: str) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def finite(x, name: str) -> float:
    y = float(x)
    if not math.isfinite(y):
        raise ValueError(f"non-finite {name}")
    return y


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dev", required=True)
    ap.add_argument("--baseline-validation", required=True)
    ap.add_argument("--candidate", action="append", nargs=4, metavar=("NAME", "LR", "DEV_JSON", "VAL_JSON"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline = read(args.baseline_dev)
    base_val_obj = read(args.baseline_validation)
    if baseline.get("suite") != "arctus-recovery-dev-v1" or int(baseline.get("cases", 0)) != 140:
        raise RuntimeError("wrong recovery development baseline")
    base_overall = finite(baseline["overall_accuracy_pct"], "baseline overall")
    base_categories = baseline.get("categories") or {}
    expected = {"bilingual", "code", "factual", "language", "math", "reasoning", "science"}
    if set(base_categories) != {f"{x}_pct" for x in expected}:
        raise RuntimeError("unexpected baseline category set")
    base_val = finite(base_val_obj["validation_loss"], "baseline validation loss")

    rows = []
    for name, lr_text, dev_path, val_path in args.candidate:
        dev = read(dev_path)
        val = read(val_path)
        lr = finite(lr_text, f"{name} lr")
        if dev.get("suite") != "arctus-recovery-dev-v1" or int(dev.get("cases", 0)) != 140:
            raise RuntimeError(f"wrong dev suite for {name}")
        if dev.get("architecture") != "frontier350-v1" or val.get("architecture") != "frontier350-v1":
            raise RuntimeError(f"architecture mismatch for {name}")

        failures = []
        overall = finite(dev["overall_accuracy_pct"], f"{name} overall")
        if overall < base_overall:
            failures.append(f"overall:{overall}<{base_overall}")
        category_scores = {}
        for cat in sorted(expected):
            score = finite(dev["categories"][cat]["accuracy_pct"], f"{name} {cat}")
            floor = finite(base_categories[f"{cat}_pct"], f"baseline {cat}")
            category_scores[cat] = score
            if score < floor:
                failures.append(f"{cat}:{score}<{floor}")
        validation_loss = finite(val["validation_loss"], f"{name} validation")
        if validation_loss > base_val:
            failures.append(f"validation_loss:{validation_loss}>{base_val}")

        rows.append({
            "name": name,
            "lr": lr,
            "checkpoint_processed_tokens": int(dev["checkpoint_processed_tokens"]),
            "overall_accuracy_pct": overall,
            "categories": category_scores,
            "validation_loss": validation_loss,
            "validation_delta": validation_loss - base_val,
            "pass": not failures,
            "failures": failures,
        })

    eligible = [x for x in rows if x["pass"]]
    eligible.sort(key=lambda x: (x["validation_loss"], -x["overall_accuracy_pct"], x["lr"]))
    selected = eligible[0]["name"] if eligible else None
    result = {
        "schema": 1,
        "purpose": "development-only LR selection; not a release/final-promotion audit",
        "baseline_validation_loss": base_val,
        "baseline_overall_accuracy_pct": base_overall,
        "selected": selected,
        "selection_allowed": selected is not None,
        "candidates": rows,
        "policy": "Fail closed: no development category, overall score, or fixed validation loss may regress versus the immutable 1B development baseline.",
    }
    Path(args.out).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
