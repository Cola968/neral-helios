"""Read-only NERAL Assistant diagnostic; never download weights or train.

Native and tool-assisted scores are separate and NOT comparable to AIME/GPQA/HLE.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from neral_assistant import ALLOWED_MODELS, assistant_turn


def score_answer(response: str, spec: dict) -> bool:
    if not isinstance(response, str) or not isinstance(spec, dict):
        raise ValueError("invalid diagnostic answer/scoring contract")
    text = " ".join(response.strip().casefold().split())
    kind = spec.get("type")
    if kind == "exact":
        return response.strip() == spec["value"]
    if kind == "integer":
        matches = re.findall(r"(?<![\w.,])[+-]?\d+(?![\w]|[.,]\d)", response)
        return bool(matches) and int(matches[-1]) == int(spec["value"])
    if kind == "contains_all":
        values = spec.get("values", [])
        if not isinstance(values, list) or not values:
            raise ValueError("invalid contains_all scoring")
        return all(str(word).casefold() in text for word in values)
    if kind == "contains_any":
        values = spec.get("values", [])
        if not isinstance(values, list) or not values:
            raise ValueError("invalid contains_any scoring")
        return any(str(word).casefold() in text for word in values)
    raise ValueError("unknown diagnostic scoring type")


def validate_cases(cases: list[dict]) -> str:
    if not isinstance(cases, list) or not cases or len(cases) > 100:
        raise ValueError("cases must be a nonempty bounded list")
    ids = set()
    for row in cases:
        if not isinstance(row, dict):
            raise ValueError("case must be a JSON object")
        name = row.get("id")
        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z0-9_\-]{1,80}", name):
            raise ValueError("invalid case id")
        if name in ids:
            raise ValueError("duplicate diagnostic case id")
        ids.add(name)
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError("missing diagnostic prompt")
        if not isinstance(row.get("category"), str) or not row["category"].strip():
            raise ValueError("missing diagnostic category")
        if not isinstance(row.get("scoring"), dict):
            raise ValueError("missing diagnostic scoring spec")
        history = row.get("history", [])
        if not isinstance(history, list) or len(history) > 12:
            raise ValueError("invalid case history")
        if row["scoring"].get("type") not in {"exact", "integer", "contains_all", "contains_any"}:
            raise ValueError("unsupported scoring spec")
    canonical = json.dumps(sorted(cases, key=lambda row: row["id"]),
                           ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def evaluate_cases(cases: list[dict], *, model: str, enable_tools: bool,
                   turn=assistant_turn) -> dict:
    fingerprint = validate_cases(cases)
    if type(enable_tools) is not bool:
        raise ValueError("enable_tools must be boolean")
    if model not in ALLOWED_MODELS:
        raise ValueError("unsupported local model for diagnostics")
    rows = []
    counts = {}
    for case in cases:
        output = turn(case["prompt"], case.get("history", []),
                      model=model, enable_tools=enable_tools,
                      max_new=160, temperature=0.0, think=False)
        if not isinstance(output, dict) or output.get("base_model") != model:
            raise ValueError("model diagnostic output identity mismatch")
        used = output.get("tools_used")
        text = output.get("text")
        if not isinstance(text, str) or not isinstance(used, list) or any(x not in {"calculator", "code_check"} for x in used):
            raise ValueError("invalid diagnostic result")
        if not enable_tools and used:
            raise ValueError("native mode returned tool-assisted result")
        correct = score_answer(text, case["scoring"])
        row = {"id": case["id"], "category": case["category"], "passed": correct,
               "text": text, "tools_used": used,
               "generated_tokens": output.get("generated_tokens", 0)}
        rows.append(row)
        info = counts.setdefault(case["category"], {"total": 0, "passed": 0})
        info["total"] += 1
        info["passed"] += int(correct)
    return {
        "schema": 1, "title": "NERAL small diagnostic, NOT an independent capability benchmark",
        "model": model, "mode": "model+tools" if enable_tools else "model-native",
        "case_contract_sha256": fingerprint, "cases": len(rows),
        "passed": sum(int(row["passed"]) for row in rows),
        "tool_calls": sum(len(row["tools_used"]) for row in rows),
        "by_category": counts, "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(ALLOWED_MODELS), default="qwen3.5:9b")
    parser.add_argument("--mode", choices=("native", "tools", "both"), default="both")
    parser.add_argument("--out", required=True, help="Local report JSON; no upload")
    args = parser.parse_args()
    fixture = Path(__file__).with_name("neral_reasoning_cases_v1.json")
    cases = json.loads(fixture.read_text(encoding="utf-8"))
    modes = [False, True] if args.mode == "both" else [args.mode == "tools"]
    reports = [evaluate_cases(cases, model=args.model, enable_tools=enabled) for enabled in modes]
    result = {"model": args.model, "reports": reports}
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{"mode": x["mode"], "passed": x["passed"], "cases": x["cases"],
                       "tool_calls": x["tool_calls"]} for x in reports], ensure_ascii=False))


if __name__ == "__main__":
    main()
