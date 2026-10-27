"""NERAL-P1 independent native-response smoke test, separate from training.

An offline evaluator only: no model inference/training/network, and not NIS2.
Prediction metadata is supplied by the test runner; a manually forged JSON
file cannot prove a real model was evaluated. Keep the raw run artifact.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

EVAL_PATH = Path(__file__).with_name("eval") / "neral_persona_holdout_v1.jsonl"
CATEGORIES = frozenset({"format", "arithmetic", "factual", "logic"})
ALLOWED_MODELS = frozenset({
    "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "qwen3.5:4b", "qwen3.5:9b",
})


def load_holdout(path: Path | str = EVAL_PATH) -> list[dict]:
    file = Path(path)
    if not file.is_file() or file.stat().st_size > 200_000:
        raise ValueError("Missing or oversized frozen NERAL holdout")
    cases = []
    for line in file.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            cases.append(json.loads(line))
    return cases


def _norm(text: str) -> str:
    return " ".join(text.casefold().split())


def validate_holdout(cases: list[dict], training_examples: list[dict]) -> dict:
    if not isinstance(cases, list) or not 0 < len(cases) <= 100:
        raise ValueError("Invalid frozen cases")
    if not isinstance(training_examples, list):
        raise ValueError("Invalid training examples")
    train_questions = set()
    for row in training_examples:
        if not isinstance(row, dict):
            raise ValueError("Invalid training source")
        messages = row.get("messages")
        if (not isinstance(messages, list) or not messages
                or not isinstance(messages[0], dict)
                or not isinstance(messages[0].get("content"), str)):
            raise ValueError("Invalid training example in contamination check")
        train_questions.add(_norm(messages[0]["content"]))
    ids = set()
    prompts = set()
    categories = set()
    overlaps = 0
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "id", "category", "prompt", "expected", "evaluator", "purpose"
        }:
            raise ValueError("Invalid holdout schema")
        identifier = case["id"]
        if (not isinstance(identifier, str)
                or not re.fullmatch(r"neval-[0-9]{3,4}", identifier)
                or identifier in ids):
            raise ValueError("Invalid or duplicate evaluation ID")
        ids.add(identifier)
        if (not isinstance(case["category"], str)
                or case["category"] not in CATEGORIES
                or case["evaluator"] != "exact"
                or case["purpose"] != "small-independent-smoke-not-nis2"):
            raise ValueError("Unexpected holdout type")
        categories.add(case["category"])
        for attr in ("prompt", "expected"):
            val = case[attr]
            if not isinstance(val, str) or not val.strip() or len(val) > 1500:
                raise ValueError("Invalid holdout text")
        prompt = _norm(case["prompt"])
        if prompt in prompts:
            raise ValueError("Repeated frozen prompt")
        prompts.add(prompt)
        if prompt in train_questions:
            overlaps += 1
    if overlaps:
        raise ValueError("Training contamination: holdout prompt found in SFT seed")
    return {"count": len(cases), "categories": sorted(categories), "overlaps": 0}


def grade_predictions(cases: list[dict], predictions: dict) -> dict:
    """Score strictly complete, native, single-model exact responses.

    The result is an *input-data score*, not verification the JSON came from a
    model; maintain provenance separately and never advertise a fixture score.
    """
    validate_holdout(cases, [])
    if not isinstance(predictions, dict):
        raise ValueError("Predictions must map IDs to response records")
    ids = {case["id"] for case in cases}
    if set(predictions) - ids:
        raise ValueError("Unknown holdout IDs")
    correct = 0
    valid = 0
    invalid = []
    models = set()
    for case in cases:
        prediction = predictions.get(case["id"])
        if not isinstance(prediction, dict) or set(prediction) != {
            "text", "model_generated", "base_model", "tools_used"
        }:
            invalid.append(case["id"])
            continue
        if (prediction["model_generated"] is not True
                or type(prediction["tools_used"]) is not list
                or prediction["tools_used"] != []
                or not isinstance(prediction["base_model"], str)
                or prediction["base_model"] not in ALLOWED_MODELS
                or not isinstance(prediction["text"], str)
                or not prediction["text"].strip()
                or len(prediction["text"]) > 10000):
            invalid.append(case["id"])
            continue
        models.add(prediction["base_model"])
        valid += 1
        correct += int(prediction["text"].strip() == case["expected"])
    if len(models) != 1:
        invalid = sorted(ids)
    complete = not invalid and valid == len(cases)
    return {
        "benchmark": "neral-persona-native-smoke-v1",
        "status": "complete" if complete else "not_measured",
        "accuracy_pct": round(100 * correct / len(cases), 2) if complete else None,
        "evaluated": valid if complete else 0,
        "total_cases": len(cases),
        "invalid_or_missing": sorted(invalid),
        "model": next(iter(models)) if complete else None,
        "tools_allowed": False,
        "nis2": None,
        "provenance_warning": (
            "Result is calculated from supplied predictions; actual model inference "
            "requires independent run logs and hash review, not this JSON alone."
        ),
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Offline native model answer evaluator")
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--holdout", type=Path, default=EVAL_PATH)
    options = parser.parse_args()
    if options.predictions.stat().st_size > 512_000:
        raise ValueError("Oversized predictions")
    cases = load_holdout(options.holdout)
    predictions = json.loads(options.predictions.read_text(encoding="utf-8"))
    print(json.dumps(grade_predictions(cases, predictions), ensure_ascii=False,
                     indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
