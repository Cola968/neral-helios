"""NERAL 2022-style *development* capability gate, not a GPT-3.5 benchmark.

Default CLI mode only validates a fixed local JSON suite. Real local Ollama
inference requires --execute AND --allow-local-ollama. No network/cloud GPU,
downloads, checkpoint writes, fine-tuning or model claims.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
from pathlib import Path

from neral_assistant import ALLOWED_MODELS, AssistantError, assistant_turn

DEFAULT_SUITE = Path(__file__).with_name("neral_2022_cases_v1.json")
MAX_CASE_FILE = 128 * 1024
MAX_ROWS = 128
CASE_ID = re.compile(r"[a-z][a-z0-9_]{2,64}\Z")
CATEGORY = re.compile(r"[a-z][a-z_]{2,32}\Z")
G_TYPES = frozenset({"exact", "integer_only", "json_exact", "manual"})
EXPECTED = frozenset({"id", "category", "language", "kind", "prompt", "history", "grading"})
ROLES = {"user", "assistant"}


class GateError(ValueError):
    """Malformed suite, invalid model attribution or failed evaluation contract."""


def _pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise GateError("duplicate JSON object key")
        out[key] = value
    return out


def _constant(value):
    raise GateError("non-finite JSON values are not allowed")


def _sha(value):
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True,
                     separators=(",", ":"), allow_nan=False).encode("utf8")
    return hashlib.sha256(raw).hexdigest()


def _str(value, limit=4096):
    return (isinstance(value, str) and bool(value.strip())
            and len(value.encode("utf8")) <= limit
            and not any(ord(char) < 32 and char not in "\n\t" for char in value))


def _validate_grading(grading, kind):
    if not isinstance(grading, dict) or grading.get("type") not in G_TYPES:
        raise GateError("invalid scoring specification")
    gtype = grading["type"]
    if kind == "manual":
        if grading != {"type": "manual"}:
            raise GateError("manual cases cannot have automatic grading")
        return
    if kind != "auto" or gtype == "manual" or grading.keys() != {"type", "value"}:
        raise GateError("automatic case missing exact scoring specification")
    value = grading["value"]
    if gtype == "exact":
        if not _str(value, 256):
            raise GateError("invalid exact reference")
    elif gtype == "integer_only":
        if type(value) is not int or not -2_000_000_000 <= value <= 2_000_000_000:
            raise GateError("invalid integer reference")
    elif gtype == "json_exact":
        if type(value) not in (dict, list) or len(_sha(value)) != 64:
            raise GateError("invalid JSON reference")


def _validate(cases):
    if not isinstance(cases, list) or not 1 <= len(cases) <= MAX_ROWS:
        raise GateError("expected 1..128 frozen cases")
    ids = set()
    for case in cases:
        if not isinstance(case, dict):
            raise GateError("case must be an object")
        kind = case.get("kind")
        expected = EXPECTED | ({"review"} if kind == "manual" else set())
        if case.keys() != expected:
            raise GateError("missing or unknown case fields")
        cid = case["id"]
        if not isinstance(cid, str) or not CASE_ID.fullmatch(cid) or cid in ids:
            raise GateError("invalid or duplicated case ID")
        ids.add(cid)
        if not isinstance(case["category"], str) or not CATEGORY.fullmatch(case["category"]):
            raise GateError("invalid category")
        if case["language"] not in {"de", "en"}:
            raise GateError("unsupported language")
        if not _str(case["prompt"]):
            raise GateError("empty or oversized prompt")
        if kind == "manual" and not _str(case["review"], 1500):
            raise GateError("manual review instructions required")
        _validate_grading(case["grading"], kind)
        history = case["history"]
        if not isinstance(history, list) or len(history) > 12 or len(history) % 2:
            raise GateError("history must be bounded alternating user/assistant pairs")
        size = len(case["prompt"].encode("utf8"))
        for i, turn in enumerate(history):
            if not isinstance(turn, dict) or turn.keys() != {"role", "content"}:
                raise GateError("invalid history turn")
            if turn["role"] != ("user" if i % 2 == 0 else "assistant") or not _str(turn["content"]):
                raise GateError("invalid history role or content")
            size += len(turn["content"].encode("utf8"))
        if size > 4096:
            raise GateError("history + prompt exceeds standard byte budget")
    return _sha(sorted(cases, key=lambda c: c["id"]))


def read_suite(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_CASE_FILE:
        raise GateError("missing, symlinked or oversized case suite")
    try:
        data = json.loads(path.read_text(encoding="utf8"), object_pairs_hook=_pairs,
                          parse_constant=_constant)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
        raise GateError("invalid case suite JSON") from exc
    _validate(data)
    return data


def score(response, grading):
    """Strict objective grading. Human-review cases NEVER receive a fake pass."""
    _validate_grading(grading, "manual" if grading.get("type") == "manual" else "auto")
    if not isinstance(response, str):
        raise GateError("response must be text")
    typ = grading["type"]
    if typ == "manual":
        return None
    if typ == "exact":
        return response.strip() == grading["value"]
    if typ == "integer_only":
        text = response.strip()
        return bool(re.fullmatch(r"[+-]?\d{1,11}", text)) and int(text) == grading["value"]
    if typ == "json_exact":
        try:
            obj = json.loads(response.strip(), object_pairs_hook=_pairs,
                             parse_constant=_constant)
        except (json.JSONDecodeError, RecursionError, UnicodeError, GateError):
            return False
        return type(obj) == type(grading["value"]) and obj == grading["value"]
    raise GateError("unknown grading type")


def plan(cases):
    digest = _validate(cases)
    counts = collections.Counter(x["category"] for x in cases)
    return {
        "schema": "neral-2022-development-gate-v1",
        "status": "PLAN_ONLY_NO_INFERENCE",
        "suite_sha256": digest,
        "cases": len(cases),
        "automatic_cases": sum(x["kind"] == "auto" for x in cases),
        "manual_cases": sum(x["kind"] == "manual" for x in cases),
        "categories": dict(sorted(counts.items())),
        "gpt35_parity_proven": False,
        "limitations": "Visible project DEV cases; no archived ChatGPT-2022 comparator or independent general benchmark.",
    }


def _verified_response(output, *, model, quality, enable_tools):
    if (not isinstance(output, dict)
        or output.get("mode") != "neral-assistant"
        or output.get("source") != "external-model"
        or output.get("base_model") != model
        or output.get("model_generated") is not True
        or output.get("quality_mode") != quality
        or output.get("reasoning_enabled") is not (quality == "deep")):
        raise GateError("foreign, native, inconsistent or misidentified provider response")
    used = output.get("tools_used")
    if (not isinstance(used, list)
        or any(item not in {"calculator", "code_check"} for item in used)
        or len(used) > 2
        or len(set(used)) != len(used)
        or (not enable_tools and used)):
        raise GateError("tool attribution must match evaluation layer")
    if not _str(output.get("text"), 24000):
        raise GateError("empty or invalid response text")
    return used


def evaluate(cases, *, model, quality, enable_tools, turn=assistant_turn):
    """Explicit caller action only. Use existing bounded Ollama assistant_turn."""
    meta = plan(cases)
    if model not in ALLOWED_MODELS or quality not in {"standard", "deep"} or type(enable_tools) is not bool:
        raise GateError("invalid local model or evaluation configuration")
    rows = []
    passed = 0
    failures = 0
    human = 0
    calculator_count = 0
    code_check_count = 0
    for case in cases:
        try:
            response = turn(
                case["prompt"], case["history"], model=model,
                enable_tools=enable_tools, max_new=1024 if quality == "deep" else 256,
                temperature=0.0, think=False, quality=quality,
            )
        except AssistantError as exc:
            failures += case["kind"] == "auto"
            human += case["kind"] == "manual"
            rows.append({"id": case["id"], "category": case["category"],
                         "kind": case["kind"], "result": "provider_error",
                         "error_class": type(exc).__name__})
            continue
        used = _verified_response(response, model=model, quality=quality,
                                  enable_tools=enable_tools)
        calculator_count += used.count("calculator")
        code_check_count += used.count("code_check")
        outcome = score(response["text"], case["grading"])
        if response.get("incomplete") is True or response.get("finish_reason") == "length":
            failures += case["kind"] == "auto"
            human += case["kind"] == "manual"
            verdict = "incomplete"
        elif outcome is None:
            human += 1
            verdict = "manual_pending"
        else:
            passed += bool(outcome)
            failures += not outcome
            verdict = "passed" if outcome else "failed"
        rows.append({
            "id": case["id"], "category": case["category"],
            "kind": case["kind"], "result": verdict,
            "response_sha256": hashlib.sha256(response["text"].encode("utf8")).hexdigest(),
            "calculator_calls": used.count("calculator"),
            "code_check_calls": used.count("code_check"),
        })
    return {
        "schema": "neral-2022-development-results-v1",
        "suite_sha256": meta["suite_sha256"],
        "model": model,
        "source": "external-model",
        "quality": quality,
        "layer": "external-model+tools" if enable_tools else "external-model-native",
        "automatic": {"total": meta["automatic_cases"], "passed": passed, "failed": failures},
        "manual_pending": human,
        "calculator_calls": calculator_count,
        "code_check_calls": code_check_count,
        "rows": rows,
        "gpt35_parity_proven": False,
        "released": False,
        "limits": "Not a GPT-3.5 historical head-to-head or public benchmark. Manual answers are ungraded; code quality requires human review. No model promotion.",
    }


def evaluate_to_path(cases, destination, *, model, quality, enable_tools, turn=assistant_turn):
    dest = Path(destination)
    if dest.exists() or dest.is_symlink():
        raise FileExistsError("report already exists")
    if not dest.parent.is_dir() or dest.parent.is_symlink():
        raise GateError("output parent directory must exist and not be symlinked")
    report = evaluate(cases, model=model, quality=quality,
                      enable_tools=enable_tools, turn=turn)
    payload = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    with dest.open("x", encoding="utf8") as stream:
        stream.write(payload)
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cases", type=Path, default=DEFAULT_SUITE)
    ap.add_argument("--plan", action="store_true", help="Default: validate suite without inference")
    ap.add_argument("--execute", action="store_true", help="Explicitly run local Ollama; never GPU training")
    ap.add_argument("--allow-local-ollama", action="store_true", help="Second opt-in for model inference")
    ap.add_argument("--out", type=Path, help="New local report JSON; never overwrite")
    ap.add_argument("--model", choices=sorted(ALLOWED_MODELS), default="qwen3.5:9b")
    ap.add_argument("--quality", choices=("standard", "deep"), default="standard")
    ap.add_argument("--tools", choices=("off", "on"), default="off")
    args = ap.parse_args(argv)
    cases = read_suite(args.cases)
    if args.execute:
        if args.plan or not args.allow_local_ollama or args.out is None:
            ap.error("--execute requires --allow-local-ollama and --out, without --plan")
        report = evaluate_to_path(cases, args.out, model=args.model, quality=args.quality,
                                  enable_tools=args.tools == "on")
        print(json.dumps({
            "suite_sha256": report["suite_sha256"],
            "model": report["model"], "quality": report["quality"],
            "layer": report["layer"], "automatic": report["automatic"],
            "manual_pending": report["manual_pending"],
            "gpt35_parity_proven": False,
        }, sort_keys=True))
    elif args.allow_local_ollama or args.out:
        ap.error("without --execute, --out and --allow-local-ollama must be omitted")
    else:
        print(json.dumps(plan(cases), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
