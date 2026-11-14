"""One-shot historical PR542 RTX4070 acceptance. VERIFY BRANCH ONLY.

For the exact aafad4b2 source tree: collect separate raw model outputs only
on the PC; GitHub logs and artifacts contain scalar measurements only.
Uses three PUBLIC older case IDs, plus optional old-six latency driver. No
prospective sealed corpus import, no training, no merge.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
import urllib.request

from flask import Flask

from neral_agent_autonomy import OllamaLocalBackend, local_only_opener
from neral_local_chat_api import register_neral_local_chat_api
from neral_nl_verified_route import strict_source_certificate
from neral_typed_constraints_v2 import PREFIX, solve_typed_constraints
from scripts.neral_nl_schema_probe import old_case_question
from scripts.neral_nl_schema_diagnostics import collect_live_diagnostics

EXPECTED = {"21453": 1, "logic-02": 6, "logic-06": 4}
QLORA = "neral-qwen35-qlora-v1:latest"
BASE = "qwen3.5:9b"
REQUIRED_SHA = "aafad4b2dd7647b5e61a3c74452f9f03ab34200f"


def create(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, sort_keys=True, indent=2)
        fh.write("\n")


def original_source_count(question: str) -> int | None:
    doc = strict_source_certificate(question)
    if doc is None:
        return None
    return solve_typed_constraints(
        PREFIX + json.dumps(doc, ensure_ascii=False), max_seconds=1.5)


def model_summary(model: dict) -> dict:
    result = {
        key: model.get(key)
        for key in (
            "status", "done_reason", "candidate_count",
            "coverage_passed", "coverage_reason", "reference_count_match",
            "load_duration_ms", "total_duration_ms", "elapsed_ms",
            "prompt_eval_count", "eval_count", "validation_error",
            "error_class",
        )
    }
    result["variable_names"] = None
    result["one_based_positional_variables"] = False
    raw = model.get("raw_json")
    if type(raw) is str:
        try:
            parsed = json.loads(raw)
            vars_dict = parsed.get("variables") if type(parsed) is dict else None
            if type(vars_dict) is dict:
                names = sorted(vars_dict)
                result["variable_names"] = names
                result["one_based_positional_variables"] = (
                    names == ["p1", "p2", "p3", "p4", "p5"]
                    and all(type(v) is dict and v.get("domain") == [1, 2, 3, 4, 5]
                            for v in vars_dict.values())
                )
        except ValueError:
            pass
    return result


def local_gpu_models() -> dict:
    request = urllib.request.Request("http://127.0.0.1:11434/api/tags")
    with local_only_opener().open(request, timeout=12) as resp:
        info = json.load(resp)
    tags = {item.get("name"): item.get("digest")
            for item in info.get("models", []) if type(item) is dict}
    if QLORA not in tags or BASE not in tags:
        raise RuntimeError("Missing exact QLoRA and baseline Ollama tags")
    return {tag: tags[tag] for tag in (QLORA, BASE)}


def make_real_think_client():
    app = Flask("neral-pr542-aafad4b2-real-think-gpu")
    backend = OllamaLocalBackend(
        "qwen3.5:9b", inference_profile="measured_nonthinking_2048")
    register_neral_local_chat_api(app, backend)
    return app.test_client()


def endpoint_case(client, question: str, expected: int | None) -> dict:
    started = time.monotonic()
    try:
        resp = client.post("/api/neral/chat/think", json={
            "prompt": question, "nl_schema_consensus": True,
        })
        elapsed_ms = round((time.monotonic() - started) * 1000, 2)
        body = resp.get_json(silent=True)
        if type(body) is not dict:
            return {"http_status": resp.status_code, "elapsed_ms": elapsed_ms,
                    "error": "non_json_endpoint_reply", "safe_to_promote": False}
        reply = {
            "http_status": resp.status_code,
            "elapsed_ms": elapsed_ms,
            "safe_to_promote": body.get("safe_to_promote"),
            "verified": body.get("verified"),
            "model_generated": body.get("model_generated"),
            "candidate_count": body.get("candidate_count"),
            "consensus_status": body.get("consensus_status"),
            "proof_backend": body.get("proof_backend"),
            "verification_scope": body.get("verification_scope"),
            "verification_status": body.get("verification_status"),
            "response_source": body.get("response_source"),
            "selection_policy": body.get("selection_policy"),
            "calls_used": body.get("calls_used"),
            "answer_matches_expected": (
                body.get("text", "").strip() == str(expected)
                if type(body.get("text")) is str and expected is not None
                else None),
        }
        if resp.status_code != 200:
            reply["error_code"] = body.get("error_code")
            reply["error_type"] = body.get("error")
        return reply
    except Exception as exc:
        return {"http_status": None,
                "elapsed_ms": round((time.monotonic()-started)*1000,2),
                "safe_to_promote": False,
                "error_class": type(exc).__name__}


def three(report: Path, scalars: Path) -> None:
    if os.environ.get("NERAL_EXPERIMENTAL_NL_SCHEMA") != "1":
        raise RuntimeError("Explicit experimental route flag missing")
    models = local_gpu_models()
    client = make_real_think_client()
    rows = []
    for case, expected in EXPECTED.items():
        question = old_case_question(case)  # ONLY historically published old cases
        cert_count = original_source_count(question)
        if cert_count != expected:
            raise RuntimeError("Independent full-source parser does not match the historical reference")
        print("PR542_BEGIN_CASE="+case, flush=True)
        # Full raw model JSON is written only into a create-only PC directory.
        raw = collect_live_diagnostics(case, QLORA, BASE)
        create(report / ("raw_models_" + case + ".json"), raw)
        endpoint = endpoint_case(client, question, expected)
        row = {
            "case": case, "reference_count": expected,
            "full_source_certificate_count": cert_count,
            "qlora": model_summary(raw["models"]["qlora"]),
            "base": model_summary(raw["models"]["base"]),
            "consensus_status": raw.get("consensus_status"),
            "consensus_candidate_count": raw.get("consensus_candidate_count"),
            "coverage_reason": raw.get("coverage_reason"),
            "proof_backend": raw.get("proof_backend"),
            "offline_diagnostic_promoted": raw.get("safe_to_promote"),
            "think_endpoint": endpoint,
        }
        rows.append(row)
        print("PR542_CASE_SCALARS=" +
              json.dumps(row, ensure_ascii=False, separators=(",", ":")),
              flush=True)

    # Negative live HTTP check: append a genuine *new* condition. That syntax
    # cannot match the full-source grammar; no two-model consensus may bypass
    # the conservative original Qwen fallback. No text or raw output uploaded.
    unsafe = (old_case_question("21453") +
              " Außerdem muss die Ziffer 1 auf der ersten Position stehen.")
    blocked = endpoint_case(client, unsafe, None)
    blocked["unsafe_condition_added"] = True
    blocked["source_certificate_rejected"] = strict_source_certificate(unsafe) is None
    blocked["guard_preserved"] = (
        blocked["source_certificate_rejected"]
        and blocked.get("http_status") == 200
        and blocked.get("safe_to_promote") is False
        and blocked.get("verified") is False
        and blocked.get("response_source") == "raw_qwen")
    print("PR542_UNSAFE_GUARD_SCALARS=" +
          json.dumps(blocked, ensure_ascii=False, separators=(",", ":")),
          flush=True)
    create(scalars / "three_known_cases.json", {
        "source_commit": REQUIRED_SHA,
        "model_digests": models,
        "cases": rows,
        "unsafe_live_guard": blocked,
        "all_three_correct_and_safe_to_promote": all(
            (r["consensus_status"] == "consensus_unverified"
             and r["proof_backend"] == "z3_unsat"
             and r["consensus_candidate_count"] == r["reference_count"]
             and r["think_endpoint"].get("safe_to_promote") is True
             and r["think_endpoint"].get("verified") is True
             and r["think_endpoint"].get("proof_backend") == "z3_unsat"
             and r["think_endpoint"].get("answer_matches_expected") is True)
            for r in rows),
        "release_gate_eligible": False,
        "new_blind_test_executed": False,
    })


def fast_summary(report: Path, scalars: Path) -> None:
    path = report / "fast_old6.json"
    if not path.is_file():
        create(scalars / "fast_old6_status.json", {
            "status": "not_completed", "reason": "old6_output_missing",
            "release_gate_eligible": False})
        return
    with path.open(encoding="utf-8") as fh:
        source = json.load(fh)
    columns = (
        "id", "category", "full_latency_ms", "fast_latency_ms",
        "full_correct", "fast_correct", "full_verified", "fast_verified",
        "full_model_loads", "fast_model_loads", "full_status",
        "fast_status", "full_failure_phase", "fast_failure_phase",
        "full_schema_attempts", "fast_schema_attempts",
        "full_base_fallback_attempts", "fast_base_fallback_attempts",
        "full_both_resident", "fast_both_resident",
    )
    keys = (
        "full_p50_ms", "fast_p50_ms", "median_latency_savings_ms",
        "count", "full_correct", "fast_correct", "full_verified",
        "fast_verified", "full_model_load_events", "fast_model_load_events",
        "full_model_load_missing_cases", "fast_model_load_missing_cases",
        "full_token_limit_cases", "fast_token_limit_cases",
        "baseline_model_only_responses", "request_order",
    )
    create(scalars / "fast_old6_scalars.json", {
        "status": "completed",
        "source_commit": REQUIRED_SHA,
        "old6_public_development_only": True,
        "sealed_prospective_corpus_not_loaded": True,
        "summary": {key: source.get(key) for key in keys},
        "cases": [{key: row.get(key) for key in columns}
                  for row in source.get("cases", [])],
        "release_gate_eligible": False,
    })
    print("FAST_OLD6_SCALARS=" +
          json.dumps({key: source.get(key) for key in keys},
                     separators=(",", ":")), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("three", "fast_summary"), required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--scalars", type=Path, required=True)
    a = p.parse_args()
    if a.mode == "three":
        three(a.report, a.scalars)
    else:
        fast_summary(a.report, a.scalars)
