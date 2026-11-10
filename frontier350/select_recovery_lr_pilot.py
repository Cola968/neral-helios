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
from recovery_identity import validate_pair


def read(path: str) -> dict:
    obj = json.loads(Path(path).read_text(encoding="utf8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected JSON object: {path}")
    return obj


def finite(x, name: str) -> float:
    if isinstance(x, bool):
        raise ValueError(f"boolean {name}")
    y = float(x)
    if not math.isfinite(y):
        raise ValueError(f"non-finite {name}")
    return y


def percentage(x, name):
    value = finite(x, name)
    if not 0 <= value <= 100:
        raise ValueError(f"out of range {name}")
    return value


def verify_counts(report):
    categories = report['categories']
    total = correct = 0
    for name, entry in categories.items():
        n, k = entry['total'], entry['correct']
        if type(n) is not int or type(k) is not int or n != 20 or not 0 <= k <= n:
            raise ValueError(f'invalid category counts: {name}')
        if abs(percentage(entry['accuracy_pct'], name) - 100*k/n) > 1e-9:
            raise ValueError('percentage disagrees with counts')
        total += n; correct += k
    if total != 140 or report['cases'] != total or abs(percentage(report['overall_accuracy_pct'],'overall')-100*correct/total)>1e-9:
        raise ValueError('overall disagrees with category counts')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-dev", required=True)
    ap.add_argument("--baseline-validation", required=True)
    ap.add_argument("--candidate", action="append", nargs=4, metavar=("NAME", "LR", "DEV_JSON", "VAL_JSON"), required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline = read(args.baseline_dev)
    base_val_obj = read(args.baseline_validation)
    verify_counts(baseline)
    if baseline.get("suite") != "arctus-recovery-dev-v1" or int(baseline.get("cases", 0)) != 140:
        raise RuntimeError("wrong recovery development baseline")
    base_overall = percentage(baseline["overall_accuracy_pct"], "baseline overall")
    base_categories = baseline.get("categories") or {}
    expected = {"bilingual", "code", "factual", "language", "math", "reasoning", "science"}
    if set(base_categories) == expected:
        base_categories = {f'{k}_pct': v['accuracy_pct'] for k, v in base_categories.items()}
    if set(base_categories) != {f"{x}_pct" for x in expected}:
        raise RuntimeError("unexpected baseline category set")
    base_val = finite(base_val_obj["validation_loss"], "baseline validation loss")

    rows = []
    for name, lr_text, dev_path, val_path in args.candidate:
        dev = read(dev_path)
        val = read(val_path)
        verify_counts(dev)
        lr = finite(lr_text, f"{name} lr")
        validate_pair(baseline, base_val_obj, dev, val)
        if lr <= 0 or name in {r['name'] for r in rows}:
            raise ValueError('invalid LR or duplicate candidate')
        if set(dev.get('categories', {})) != expected:
            raise ValueError('candidate category set differs')
        if dev.get("suite") != "arctus-recovery-dev-v1" or int(dev.get("cases", 0)) != 140:
            raise RuntimeError(f"wrong dev suite for {name}")
        if dev.get("architecture") != "frontier350-v1" or val.get("architecture") != "frontier350-v1":
            raise RuntimeError(f"architecture mismatch for {name}")

        failures = []
        overall = percentage(dev["overall_accuracy_pct"], f"{name} overall")
        if overall < base_overall:
            failures.append(f"overall:{overall}<{base_overall}")
        category_scores = {}
        for cat in sorted(expected):
            score = percentage(dev["categories"][cat]["accuracy_pct"], f"{name} {cat}")
            floor = percentage(base_categories[f"{cat}_pct"], f"baseline {cat}")
            category_scores[cat] = score
            if score < floor:
                failures.append(f"{cat}:{score}<{floor}")
        validation_loss = finite(val["validation_loss"], f"{name} validation")
        if min(validation_loss, base_val) < 0:
            raise ValueError('negative validation loss')
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
