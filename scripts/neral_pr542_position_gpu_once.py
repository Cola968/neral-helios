"""RTX4070-only opt-in PR542 old development probes (not blind100).

This one-time verification-branch harness never imports the sealed prospective
generalization dataset or sends unredacted user/model text to GitHub.
Three unredacted old-case diagnostics remain strictly on the local PC; only
scalar verification/latency summaries may be published as a GitHub artifact.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

from flask import Flask

from neral_agent_autonomy import OllamaLocalBackend
from neral_local_chat_api import register_neral_local_chat_api
from neral_identity import build_identity_prompt
from scripts.neral_blind_100_cases import build as historical_old100
from scripts.neral_nl_schema_diagnostics import collect_live_diagnostics
from scripts.neral_nl_schema_probe import old_case_question

AUTH = "RUN_PR542_POSITION_OLD_ONLY_4070"
BASE = "qwen3.5:9b"
QLORA = "neral-qwen35-qlora-v1:latest"
QLORA_DIGEST = "9c45488f99cc446821f2008f74a4320a1de58a2f5382a68db889eadc8660dda8"
GUARD_GIT_BLOB = "b87c594ee54a20b2209b6bf176bc0aacfe256ce8"
OLD3 = ("21453", "logic-02", "logic-06")
OLD6 = (
    "blind100-mathematics-01",
    "blind100-logic-02",
    "blind100-logic-06",
    "blind100-programming-01",
    "blind100-programming-03",
    "blind100-mixed-01",
)


def _local_inventory() -> dict:
    req = urllib.request.Request("http://127.0.0.1:11434/api/tags")
    with urllib.request.urlopen(req, timeout=10) as response:
        data = response.read(32769)
    if len(data) > 32768:
        raise RuntimeError("Oversized local model inventory")
    packet = json.loads(data)
    models = packet.get("models")
    if type(models) is not list:
        raise RuntimeError("Missing local Ollama models")
    return {m.get("name"): m.get("digest")
            for m in models if type(m) is dict}


def _pr537_unchanged() -> bool:
    proc = subprocess.run(
        ["git", "hash-object", "--path=neral_tir_regression_guard.py",
         "neral_tir_regression_guard.py"],
        capture_output=True, text=True, timeout=10, check=False)
    return proc.returncode == 0 and proc.stdout.strip() == GUARD_GIT_BLOB


def _integer_answer(text: object) -> int | None:
    import re
    if type(text) is not str:
        return None
    m = re.fullmatch(r"\s*(?:FINAL_ANSWER:\s*)?(-?[0-9]{1,12})\s*", text)
    return int(m.group(1)) if m else None


def _without_raw_model_text(report: dict) -> dict:
    return {
        "case": report["case"],
        "reference_count": report["reference_count"],
        "consensus_status": report["consensus_status"],
        "candidate_count": report["consensus_candidate_count"],
        "proof_backend": report["proof_backend"],
        "verified": report["verified"],
        "safe_to_promote": report["safe_to_promote"],
        "models": {
            key: {
                "status": value.get("status"),
                "candidate_count": value.get("candidate_count"),
                "coverage_passed": value.get("coverage_passed"),
                "coverage_reason": value.get("coverage_reason"),
                "reference_count_match": value.get("reference_count_match"),
                "done_reason": value.get("done_reason"),
                "validation_error": value.get("validation_error"),
                "eval_count": value.get("eval_count"),
                "prompt_eval_count": value.get("prompt_eval_count"),
                "load_duration_ms": value.get("load_duration_ms"),
                "total_duration_ms": value.get("total_duration_ms"),
                "elapsed_ms": value.get("elapsed_ms"),
            }
            for key, value in report["models"].items()
        },
    }


def _http_result(client, question: str) -> dict:
    started = time.monotonic()
    packet = client.post(
        "/api/neral/chat/think",
        json={"prompt": question, "nl_schema_consensus": True})
    elapsed = round((time.monotonic()-started)*1000, 2)
    row = packet.get_json(silent=True) or {}
    if type(row) is not dict:
        row = {}
    return {
        "status_code": packet.status_code,
        "latency_ms": elapsed,
        "verified": row.get("verified") is True,
        "safe_to_promote": row.get("safe_to_promote") is True,
        "selected": row.get("selection_policy"),
        "proof_backend": row.get("proof_backend"),
        "consensus_status": row.get("consensus_status"),
        "answer_integer": _integer_answer(row.get("text")),
        "candidate_count": row.get("candidate_count")
        if type(row.get("candidate_count")) is int else None,
    }


def _base_result(backend, prompt: str) -> dict:
    started = time.monotonic()
    try:
        model_turns = [
            {"role": "system",
             "content": build_identity_prompt(
                 "web", "local_foundation_not_neral_adapter")},
            {"role": "user", "content": prompt},
        ]
        response = backend.respond_chat(model_turns)
        return {"status": "completed",
                "answer_integer": _integer_answer(response),
                "latency_ms": round((time.monotonic()-started)*1000, 2)}
    except (RuntimeError, ValueError, OSError, TimeoutError):
        return {"status": "inference_error", "answer_integer": None,
                "latency_ms": round((time.monotonic()-started)*1000, 2)}


def run(out: Path, *, authorize: str, old6: bool) -> dict:
    if authorize != AUTH or type(out) is not Path or out.exists():
        raise ValueError("Unauthorized or overwriting live PR542 probe")
    if os.environ.get("NERAL_EXPERIMENTAL_NL_SCHEMA") != "1":
        raise RuntimeError("NL source verified route flag not enabled")
    if not _pr537_unchanged():
        raise RuntimeError("PR537 source guard must be the original blob")
    installed = _local_inventory()
    if installed.get(QLORA) != QLORA_DIGEST or not installed.get(BASE):
        raise RuntimeError("Required exact locally installed model tags/digests missing")
    if installed[QLORA] == installed[BASE]:
        raise RuntimeError("Two model tags point to identical weights")

    private = Path.home() / "NERAL-PR542-OLDCASE-REPORTS" / (
        "run-" + os.environ.get("GITHUB_RUN_ID", "manual") + "-position")
    private.mkdir(parents=True, exist_ok=False)

    backend = OllamaLocalBackend(
        BASE, timeout=90.0, inference_profile="measured_nonthinking_2048")
    app = Flask("neral-pr542-old-probe")
    register_neral_local_chat_api(app, backend)
    client = app.test_client()

    old_three = []
    for case in OLD3:
        # The full old-case model text goes ONLY to the local private folder.
        raw = collect_live_diagnostics(case, QLORA, BASE)
        (private / ("position-v2-"+case+".json")).write_text(
            json.dumps(raw, indent=2, ensure_ascii=False), encoding="utf8")
        diagnostic = _without_raw_model_text(raw)
        http = _http_result(client, old_case_question(case))
        diagnostic["think_endpoint"] = http
        diagnostic["think_reference_count_match"] = (
            http["answer_integer"] == diagnostic["reference_count"]
            if http["verified"] else False)
        old_three.append(diagnostic)
        print("OLD3_SCALAR " + json.dumps({
            "case": case,
            "consensus": diagnostic["consensus_status"],
            "http_status": http["status_code"],
            "safe_to_promote": http["safe_to_promote"],
            "reference_match": diagnostic["think_reference_count_match"],
            "latency_ms": http["latency_ms"],
        }), flush=True)

    previous_old6 = []
    if old6:
        # This is the OLD disclosed development corpus, not the sealed 100.
        old_rows = {r["id"]: r for r in historical_old100()}
        for index, key in enumerate(OLD6):
            old = old_rows[key]
            question = old["question"]
            if index % 2 == 0:
                baseline = _base_result(backend, question)
                new = _http_result(client, question)
            else:
                new = _http_result(client, question)
                baseline = _base_result(backend, question)
            previous_old6.append({
                "id": key, "category": old["category"],
                "baseline": {
                    "latency_ms": baseline["latency_ms"],
                    "correct": baseline["answer_integer"] == old["gold_integer"],
                    "status": baseline["status"],
                },
                "optin_nl": {
                    "latency_ms": new["latency_ms"],
                    "correct": new["answer_integer"] == old["gold_integer"],
                    "verified": new["verified"],
                    "safe_to_promote": new["safe_to_promote"],
                    "selection_policy": new["selected"],
                    "status_code": new["status_code"],
                },
            })
            print("OLD6_SCALAR "+json.dumps({
                "id":key,"base_latency_ms":baseline["latency_ms"],
                "nl_latency_ms":new["latency_ms"],
                "base_correct":baseline["answer_integer"]==old["gold_integer"],
                "nl_correct":new["answer_integer"]==old["gold_integer"],
            }), flush=True)

    medians = None
    if previous_old6:
        medians = {
            "base_p50_ms": round(statistics.median(
                p["baseline"]["latency_ms"] for p in previous_old6), 2),
            "optin_nl_p50_ms": round(statistics.median(
                p["optin_nl"]["latency_ms"] for p in previous_old6), 2),
            "base_correct": sum(p["baseline"]["correct"] for p in previous_old6),
            "optin_nl_correct": sum(p["optin_nl"]["correct"] for p in previous_old6),
        }

    summary = {
        "run_type": "old3_proof_and_old6_exploratory_raw_vs_nl",
        "corpus": "already_disclosed_old_development_only",
        "prospective_100_opened": False,
        "main_changed": False,
        "pr537_hash_verified": True,
        "local_raw_files_only": True,
        "model_tags": [QLORA, BASE],
        "full_source_verification_not_general_prose": True,
        "blind_release_gate_eligible": False,
        "old_three": old_three,
        "old_six": previous_old6,
        "old_six_medians": medians,
        "old_six_protocol_comparable_to_original_full_fast_old6": False,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print("GPU_REPORT_SUMMARY " + json.dumps({
        "old3_http_verified": sum(
            x["think_endpoint"]["verified"] for x in old_three),
        "old3_http_reference_matched": sum(
            x["think_reference_count_match"] for x in old_three),
        "old6_measured": len(previous_old6),
        "old6_medians": medians,
    }), flush=True)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--old6", action="store_true")
    args = parser.parse_args(argv)
    run(args.out, authorize=args.authorize, old6=args.old6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
