from __future__ import annotations

"""Strict Arctus 1B -> 2B capability promotion gate.

The gate does not claim SGD itself is monotonic. It enforces monotonic *promotion*:
a candidate may only replace the immutable 1B baseline (or a later promoted baseline)
when every tracked higher-is-better metric is >= baseline and every lower-is-better
metric is <= baseline. Missing/non-finite values fail closed.

Candidate scorecards must be produced by frozen evaluation code. Benchmark examples
must never be used as training data or exposed to corpus construction.
"""

import argparse
import json
import math
from pathlib import Path


def finite(value, key: str) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"non-numeric metric {key}") from exc
    if not math.isfinite(x):
        raise ValueError(f"non-finite metric {key}")
    return x


def load(path: Path) -> dict:
    obj = json.loads(path.read_text(encoding="utf8"))
    if not isinstance(obj, dict):
        raise ValueError(f"expected object: {path}")
    return obj


def assess(baseline: dict, candidate: dict) -> dict:
    if baseline.get("schema") != 1:
        raise ValueError("unsupported baseline schema")
    if candidate.get("schema") != 1:
        raise ValueError("unsupported candidate schema")
    if candidate.get("architecture") != baseline.get("architecture"):
        raise ValueError("candidate architecture differs from baseline")

    failures: list[str] = []
    comparisons: dict[str, dict] = {}

    for direction in ("higher_is_better", "lower_is_better"):
        base_group = baseline.get(direction)
        cand_group = candidate.get(direction)
        if not isinstance(base_group, dict) or not isinstance(cand_group, dict):
            failures.append(f"missing_group:{direction}")
            continue
        for key, base_value in base_group.items():
            if key not in cand_group:
                failures.append(f"missing:{key}")
                continue
            try:
                b = finite(base_value, key)
                c = finite(cand_group[key], key)
            except ValueError as exc:
                failures.append(str(exc))
                continue
            if direction == "higher_is_better":
                passed = c >= b
                delta = c - b
                rule = ">="
            else:
                passed = c <= b
                delta = b - c  # positive means improvement for lower-is-better metrics
                rule = "<="
            comparisons[key] = {
                "baseline": b,
                "candidate": c,
                "rule": rule,
                "delta_in_improvement_direction": delta,
                "pass": passed,
            }
            if not passed:
                failures.append(f"regression:{key}:baseline={b}:candidate={c}")

    base_tokens = int(baseline.get("checkpoint_processed_tokens", 0))
    cand_tokens = int(candidate.get("checkpoint_processed_tokens", 0))
    if cand_tokens <= base_tokens:
        failures.append(
            f"candidate_not_newer:baseline_tokens={base_tokens}:candidate_tokens={cand_tokens}"
        )

    return {
        "pass": not failures,
        "promotion_allowed": not failures,
        "baseline_tokens": base_tokens,
        "candidate_tokens": cand_tokens,
        "failures": failures,
        "comparisons": comparisons,
        "policy": (
            "Fail closed. A 1B->2B candidate is promotable only if every frozen tracked "
            "higher-is-better metric is at least the baseline and every lower-is-better "
            "metric is no worse. Training may fluctuate; promotion may not regress."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = assess(load(Path(args.baseline)), load(Path(args.candidate)))
    Path(args.out).write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8"
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)
    raise SystemExit(0 if result["pass"] else 2)


if __name__ == "__main__":
    main()
