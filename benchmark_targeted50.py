"""Frozen targeted diagnostic for Neral-50M math, code, German and reasoning.

This benchmark is model-only. It never calls SOLVE, Helios tools or an external model.
Each case is a two-way continuation preference scored with length-normalized conditional NLL.
The JSONL suite is committed and screened from the new I1 SFT builder. B3 pretraining predates this suite, so it is supplemental rather than a formal release gate.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path

import torch

from benchmark_v3 import _conditional_loss
from experiment import load_model, sha

SUITE_VERSION = 1
DEFAULT_SUITE = Path(__file__).resolve().parent / "benchmarks50" / "targeted_v1.jsonl"
REQUIRED_DOMAINS = {"de_grammar", "math", "code", "reasoning"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_suite(path: Path = DEFAULT_SUITE):
    rows = [json.loads(line) for line in path.read_text(encoding="utf8").splitlines() if line.strip()]
    if len(rows) < 40:
        raise ValueError("targeted suite is unexpectedly small")
    ids = set()
    by_domain = collections.Counter()
    for row in rows:
        missing = {"id", "domain", "language", "prompt", "preferred", "distractor"} - set(row)
        if missing:
            raise ValueError(f"targeted row missing fields: {sorted(missing)}")
        if row["id"] in ids:
            raise ValueError(f"duplicate targeted id: {row['id']}")
        ids.add(row["id"])
        if row["domain"] not in REQUIRED_DOMAINS:
            raise ValueError(f"unsupported targeted domain: {row['domain']}")
        if row["language"] not in {"de", "en"}:
            raise ValueError("targeted language must be de/en")
        if not all(isinstance(row[k], str) and row[k] for k in ("prompt", "preferred", "distractor")):
            raise ValueError(f"empty targeted text: {row['id']}")
        if row["preferred"] == row["distractor"]:
            raise ValueError(f"identical continuations: {row['id']}")
        by_domain[row["domain"]] += 1
    if set(by_domain) != REQUIRED_DOMAINS or any(by_domain[d] < 10 for d in REQUIRED_DOMAINS):
        raise ValueError(f"targeted domain coverage invalid: {dict(by_domain)}")
    return rows


def evaluate(model, device, rows):
    cases = []
    by = collections.defaultdict(list)
    for row in rows:
        good = _conditional_loss(model, row["prompt"], row["preferred"], device)
        bad = _conditional_loss(model, row["prompt"], row["distractor"], device)
        margin = bad["mean_nll"] - good["mean_nll"]
        correct = margin > 0
        item = {
            "id": row["id"],
            "domain": row["domain"],
            "language": row["language"],
            "correct": correct,
            "margin_nats_per_byte": margin,
            "preferred_score": good,
            "distractor_score": bad,
        }
        cases.append(item)
        by[row["domain"]].append(item)
    domains = {}
    for domain, items in sorted(by.items()):
        domains[domain] = {
            "correct": sum(x["correct"] for x in items),
            "total": len(items),
            "accuracy": sum(x["correct"] for x in items) / len(items),
            "mean_margin_nats_per_byte": statistics.mean(x["margin_nats_per_byte"] for x in items),
        }
    return {
        "overall": {
            "correct": sum(x["correct"] for x in cases),
            "total": len(cases),
            "accuracy": sum(x["correct"] for x in cases) / len(cases),
            "mean_margin_nats_per_byte": statistics.mean(x["margin_nats_per_byte"] for x in cases),
        },
        "domains": domains,
        "cases": cases,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--suite", default=str(DEFAULT_SUITE))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    suite = Path(args.suite)
    rows = load_suite(suite)
    torch.set_num_threads(4)
    torch.manual_seed(968_700)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, payload = load_model(args.checkpoint, device)
    model.eval()
    result = {
        "benchmark": "Neral Targeted v1 (supplemental)",
        "version": SUITE_VERSION,
        "definition": "Supplemental post-B3 two-way continuation preference diagnostic for German grammar, math, code and reasoning; model-only.",
        "suite_sha256": sha256_file(suite),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": sha(args.checkpoint),
        "parameters": model.parameter_report(),
        "step": payload.get("step"),
        "device": str(device),
        "gpu": torch.cuda.get_device_name() if device.type == "cuda" else None,
    }
    result.update(evaluate(model, device, rows))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")
    print(json.dumps({"overall_accuracy": result["overall"]["accuracy"], "domains": {k: v["accuracy"] for k, v in result["domains"].items()}, "saved": str(out)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
