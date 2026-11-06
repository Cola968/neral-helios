"""Read-only exact solver benchmark for the 8 frozen v2 linear equations.

This measures a deterministic tool oracle, NOT an observed model + tool
selection score, and deliberately does not change the original 80-case
strict response formatter. No prompts or raw model outputs are uploaded.
"""
from __future__ import annotations
import json
import re
from fractions import Fraction
from scripts.neral_math_v2_eval import cases
from neral_agent_math import solve_math_symbolic

EXPECTED_IDS = tuple(f"novel-{n:03d}" for n in range(41, 49))
EQUATION_PATTERN = re.compile(r"[+-]?\d+x\s*\+\s*\([+-]?\d+\)\s*=\s*[+-]?\d+")


def score_equations() -> dict:
    evals = {x["id"]: x for x in cases()}
    if len(evals) != 80 or any(key not in evals for key in EXPECTED_IDS):
        raise ValueError("Original 80-case math development suite changed")
    correct = 0
    checked = []
    for key in EXPECTED_IDS:
        item = evals[key]
        if item["family"] != "equation":
            raise ValueError("Changed math family for frozen equation case")
        matching = EQUATION_PATTERN.search(item["prompt"])
        if not matching:
            raise ValueError("Frozen equation prompt has an unrecognized format")
        answer = solve_math_symbolic(matching.group())["value"]
        passed = Fraction(answer) == Fraction(item["expected"])
        correct += int(passed)
        checked.append({"id": key, "passed": passed})
    return {
        "schema": 1,
        "status": "DETERMINISTIC_TOOL_ORACLE_DEV_ONLY",
        "equations": 8,
        "tool_oracle_correct": correct,
        "old_v2_without_tool_correct": 2,
        "model_selected_tool_actions_measured": False,
        "v3_weights_trained": False,
        "independent_release_holdout": False,
        "cases": checked,
    }


if __name__ == "__main__":
    print(json.dumps(score_equations(), sort_keys=True))
