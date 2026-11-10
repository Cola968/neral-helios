"""NERAL-P1 independent native-response smoke test, separate from training.

An offline evaluator only: no model inference/training/network, and not NIS2.
Prediction metadata is supplied by the test runner; a manually forged JSON
file cannot prove a real model was evaluated. Keep the raw run artifact.
"""
from __future__ import annotations

import json
import re
import hashlib
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
    return " ".join(re.sub(r"[\W_]+", " ", text.casefold(), flags=re.UNICODE).split())


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



def grade_recorded_run(cases: list[dict], run: dict) -> dict:
    """Validate a runner envelope before assigning any diagnostic score.

    Provenance is supplied by local software, not cryptographic attestation.
    Never imply this small exact-answer smoke test measures NIS2 or personality.
    """
    validate_holdout(cases, [])
    if not isinstance(run,dict) or set(run)!={"provenance","predictions"}:
        raise ValueError("Missing or ambiguous model run envelope")
    info=run["provenance"]
    predictions=run["predictions"]
    if not isinstance(info,dict) or not isinstance(predictions,dict):
        raise ValueError("Invalid run metadata")
    raw=json.dumps(cases,ensure_ascii=False,sort_keys=True,
                   separators=(",",":"),allow_nan=False).encode("utf-8")
    suite_hash=hashlib.sha256(raw).hexdigest()
    if info.get("suite_sha256")!=suite_hash:
        raise ValueError("Frozen holdout suite hash mismatch")
    digest=info.get("model_digest")
    if not isinstance(digest,str) or not re.fullmatch(r"[0-9a-f]{64}",digest):
        raise ValueError("Missing verifiable local model digest")
    model=info.get("model_id")
    if not isinstance(model,str) or model not in ALLOWED_MODELS:
        raise ValueError("Unknown model identity in run")
    if info.get("runner")!="NERAL-P1-native-ollama-smoke":
        raise ValueError("Unknown runner identity")
    generation=info.get("generation")
    if generation!={"temperature":0.0,"num_ctx":4096,"num_predict":128,
                    "think":False,"tools":False}:
        raise ValueError("Generation parameters are not comparable")
    if set(predictions)!={case["id"] for case in cases}:
        raise ValueError("Missing or unexpected predictions in run")
    if not all(
        isinstance(item,dict) and item.get("base_model")==model
        for item in predictions.values()
    ):
        raise ValueError("Mixed model IDs in one recorded run")
    mode=info.get("mode")
    if mode not in ("local_ollama_http","simulated_test_transport"):
        raise ValueError("Unknown inference mode")
    if mode=="simulated_test_transport" or any(
        item.get("model_generated") is not True for item in predictions.values()
    ):
        # The old exact-response scorer has no provenance and should not be
        # applied to injected fake transports; return explicit unknown.
        base=grade_predictions(cases,{})
        base.update({"run_provenance":mode,
                     "model_digest":None,
                     "reason":"No verifiable local model generation; simulation cannot earn a score"})
        return base
    result=grade_predictions(cases,predictions)
    return {
        **result,
        "run_provenance":mode,
        "model_digest":digest,
        "suite_sha256":suite_hash,
        "warning":(
            "Exact-match diagnostic only, not NIS2. Local log provenance "
            "is self-reported and is not cryptographic device attestation."
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
    result = (grade_recorded_run(cases, predictions)
              if isinstance(predictions,dict) and "provenance" in predictions
              else grade_predictions(cases, predictions))
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
