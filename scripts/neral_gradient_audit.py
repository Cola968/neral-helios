"""NERAL: read-only Karpathy-inspired pre-clipping gradient telemetry.

No GPU, third-party packages, model I/O or network. This is a diagnostic,
NOT evidence of improved generalization or a reason to change clip settings.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Mapping, Sequence


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def summarize_gradient_logs(logs: Sequence[Mapping[str, object]], *,
                            max_grad_norm: float,
                            expected_steps: int | None = None) -> dict:
    """Count *logged* grad norms over the Trainer pre-clipping threshold.

    Transformers Trainer logs ``grad_norm`` BEFORE clipping. A logging interval
    is not a measurement of every optimizer step. Sparse observation therefore
    cannot justify a claim about all the optimizer steps in a run.
    """
    threshold = _finite_nonnegative(max_grad_norm, "max_grad_norm")
    if threshold == 0:
        raise ValueError("max_grad_norm must be positive")
    if (expected_steps is not None and
            (type(expected_steps) is not int or expected_steps <= 0)):
        raise ValueError("expected_steps must be a positive int")
    if not isinstance(logs, (tuple, list)):
        raise ValueError("logs must be a list or tuple")

    samples: list[tuple[int, float]] = []
    for row in logs:
        if not isinstance(row, Mapping):
            raise ValueError("each log entry must be a mapping")
        if "grad_norm" not in row:
            continue
        step = row.get("step")
        if type(step) is not int or step <= 0:
            raise ValueError("gradient sample needs a positive integer step")
        if expected_steps is not None and step > expected_steps:
            raise ValueError("logged step exceeds optimizer steps")
        if samples and step <= samples[-1][0]:
            raise ValueError("gradient samples must have distinct ascending steps")
        samples.append((step, _finite_nonnegative(row["grad_norm"], "grad_norm")))
    if not samples:
        raise ValueError("no usable gradient observations; do not infer zero clipping")

    above = sum(value > threshold for _, value in samples)
    n = len(samples)
    complete = (expected_steps is not None and n == expected_steps and
                samples[0][0] == 1 and samples[-1][0] == expected_steps)
    return {
        "norm_position": "before_clipping",
        "max_grad_norm": threshold,
        "optimizer_steps_in_run": expected_steps,
        "observed_gradient_samples": n,
        "first_observed_step": samples[0][0],
        "last_observed_step": samples[-1][0],
        "logged_above_clip_threshold": above,
        "observed_fraction_above_threshold": above / n,
        "min_logged_grad_norm": min(v for _, v in samples),
        "max_logged_grad_norm": max(v for _, v in samples),
        "all_optimizer_steps_measured": complete,
        "generalization_proven": False,
    }


def summarize_private_report(report: Mapping[str, object], *,
                             max_grad_norm: float = 1.0) -> dict:
    """Read only scalar gradient samples; never emit answers or dataset text."""
    if not isinstance(report, Mapping):
        raise ValueError("expected a JSON object")
    return summarize_gradient_logs(
        report.get("recent_train_logs"),
        max_grad_norm=max_grad_norm,
        expected_steps=report.get("steps_completed"),
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--private-report", type=Path, required=True,
                        help="Local scalar-private.json; do not upload it")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    args = parser.parse_args(argv)
    if (args.private_report.is_symlink() or not args.private_report.is_file() or
            args.private_report.stat().st_size > 2_000_000):
        raise ValueError("Expected bounded local non-symlink JSON report")
    report = json.loads(args.private_report.read_text(encoding="utf8"))
    print(json.dumps(summarize_private_report(report, max_grad_norm=args.max_grad_norm),
                     sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
