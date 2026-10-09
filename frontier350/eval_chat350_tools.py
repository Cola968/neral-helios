"""CPU-only evaluation of the deterministic tool layer, never model-native ability.

Frozen and procedurally-generated benchmarks are evaluation-only. This module
reads their prompts, routes to the explicit math tool, and records abstentions.
Do not mix these scores with model-only capability benchmark reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .chat350_tools import tool_answer
    from .score_chat350_capabilities import score_one
    from .build_chat350_stress_cases import generate_cases
except ImportError:
    from chat350_tools import tool_answer
    from score_chat350_capabilities import score_one
    from build_chat350_stress_cases import generate_cases


def last_user_message(prefix: str) -> str:
    if not isinstance(prefix, str) or not prefix.endswith("\nAssistant:"):
        raise ValueError("unsupported chat benchmark prefix")
    value = prefix.rsplit("User:", 1)[-1].rsplit("\nAssistant:", 1)[0].strip()
    if not value:
        raise ValueError("empty user question")
    return value


def evaluate_tool_layer(cases: list[dict]) -> dict:
    if not cases or len({r["id"] for r in cases}) != len(cases):
        raise ValueError("invalid/duplicate benchmark cases")
    details = []
    per_category = {}
    for row in cases:
        question = last_user_message(row["prefix"])
        outcome = tool_answer(question)
        covered = outcome is not None
        correct = bool(covered and score_one(outcome.text, row["scoring"]))
        category = row["category"]
        group = per_category.setdefault(category, {"total": 0, "covered": 0, "correct": 0})
        group["total"] += 1
        group["covered"] += int(covered)
        group["correct"] += int(correct)
        details.append({"id": row["id"], "category": category, "covered": covered,
                        "correct": correct,
                        "tool_kind": outcome.kind if covered else None,
                        "response": outcome.text if covered else None})
    count = len(details)
    return {
        "schema": 1,
        "mode": "deterministic-tool-only",
        "not_model_native": True,
        "cases": count,
        "covered": sum(r["covered"] for r in details),
        "correct": sum(r["correct"] for r in details),
        "per_category": per_category,
        "details": details,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stress-seed", type=int, default=3502026)
    args = ap.parse_args()
    frozen = json.loads(Path(args.frozen_cases).read_text(encoding="utf8"))
    out = {
        "frozen": evaluate_tool_layer(frozen),
        "stress": evaluate_tool_layer(generate_cases(seed=args.stress_seed)),
        "stress_seed": args.stress_seed,
        "note": "Tool-only exact coverage. Native model scores are separate; no generated model weights are assessed.",
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"frozen": {k: out["frozen"][k] for k in ("cases", "covered", "correct")},
                      "stress": {k: out["stress"][k] for k in ("cases", "covered", "correct")}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
