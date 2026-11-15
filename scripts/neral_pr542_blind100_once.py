"""One-time sealed prospective 100 comparison for immutable PR542 source.
Evaluation wrapper lives only on an isolated verify branch; no model/guard edits.
Outputs original answers locally, uploads aggregate scalars only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

EXPECTED_SHA = "74cf83ce3e9b3644b306556438a3e51373589ffdaf254dc84abb74e08fd04258"
PREVIOUS_UNLOCATED_SHA = "3feca67d6ee9fd48b462bdbf3e15e256b9e72a1a35357d30d75d383b9a7b280d"
HISTORICAL_GENERATOR_COMMIT = "8c9c58b4cdc7ffda1fb3392a6e6277b59fae203f"
HISTORICAL_GENERATOR_LF_SHA = "477d78aa72a2f52f1b4b37bea6f6c3afa8ffabbce41b44d4743215cdf7487fcd"
SOURCE_SHA = "737413f03b347676108d507ff3fe7693975dd269"
CATS = ("mathematics", "logic", "programming", "mixed")

def verify_historical_generator(generator) -> str:
    """Pin exact pre-PR542 source bytes, without checking out or rewriting it."""
    checked = Path(generator.__file__).resolve()
    expected = (Path.cwd() / "scripts" / "neral_blind_100_cases.py").resolve()
    if checked != expected or not checked.is_file():
        raise RuntimeError("Historical generator is not imported from immutable source checkout")
    canonical = checked.read_bytes().replace(b"\r\n", b"\n")
    observed = hashlib.sha256(canonical).hexdigest()
    if observed != HISTORICAL_GENERATOR_LF_SHA:
        raise RuntimeError("Historical generator source integrity check failed")
    print("ORIGINAL_GENERATOR_COMMIT " + HISTORICAL_GENERATOR_COMMIT, flush=True)
    print("ORIGINAL_GENERATOR_SOURCE_LF_SHA256 " + observed, flush=True)
    return observed

def local_tags(request):
    response = request("GET", "/api/tags")
    models = response.get("models")
    if not isinstance(models, list):
        raise RuntimeError("Ollama model inventory unavailable")
    return {x.get("name"): x.get("digest") for x in models if isinstance(x, dict)}

def strictly_correct(text, gold, parse_answer):
    if type(text) is not str:
        return False
    return parse_answer(text) == gold

def measure(cases, out: Path, tags_before: dict):
    from neral_agent_autonomy import OllamaLocalBackend
    from neral_identity import build_identity_prompt
    from neral_preview_server import create_preview_app
    from scripts.neral_generalization_100_run import parse_answer, gpu_offload_mib
    from scripts.neral_qlora_holdout_live import request

    low = OllamaLocalBackend("qwen3:8b")
    high = OllamaLocalBackend("qwen3.5:9b",
                              inference_profile="measured_nonthinking_2048")
    high.set_benchmark_seed(0)
    app = create_preview_app(low, experimental_high_backend=high)
    client = app.test_client()
    origin = "http://127.0.0.1:9690"
    client.get("/", base_url=origin)
    session = client.get("/api/neral/preview/session", base_url=origin)
    if session.status_code != 200 or type(session.json.get("csrf")) is not str:
        raise RuntimeError("Could not obtain local evaluation API session")
    headers = {"X-Neral-CSRF": session.json["csrf"]}
    rows = []
    raw_path = out / "private-results.jsonl"
    with raw_path.open("x", encoding="utf8", newline="\n") as rawfile:
        for index, case in enumerate(cases, 1):
            prompt = case["question"]
            original = [
                {"role": "system", "content": build_identity_prompt(
                    "web", "local_foundation_not_neral_adapter")},
                {"role": "user", "content": prompt},
            ]
            baseline_text = None
            baseline_error = None
            start = time.monotonic()
            try:
                baseline_text = high.respond_chat(original)
            except Exception as error:
                baseline_error = type(error).__name__
            baseline_s = round(time.monotonic() - start, 3)
            if gpu_offload_mib("qwen3.5:9b") < 4096:
                raise RuntimeError("Baseline was not GPU offloaded: qwen3.5:9b")
            start = time.monotonic()
            candidate = client.post(
                "/api/neral/chat/think",
                json={"prompt": prompt, "nl_schema_consensus": True},
                headers=headers, base_url=origin,
            )
            candidate_s = round(time.monotonic() - start, 3)
            body = candidate.json if isinstance(candidate.json, dict) else {}
            if candidate.status_code != 200:
                raise RuntimeError("Candidate API HTTP " + str(candidate.status_code))
            candidate_text = body.get("text")
            if type(candidate_text) is not str or not candidate_text.strip():
                raise RuntimeError("Candidate API gave no complete answer")
            if baseline_error and baseline_error != "OllamaCompletionError":
                raise RuntimeError("Unexpected baseline transport failure: " +
                                   baseline_error)
            proof = (body.get("selection_policy") ==
                     "independently_source_verified_nl" and
                     body.get("proof_backend") == "z3_unsat" and
                     body.get("verified") is True and
                     body.get("safe_to_promote") is True)
            if proof and body.get("selected_extractor") not in ("qlora", "base"):
                raise RuntimeError("Malformed source-proof provenance")
            gold = case["gold_integer"]
            row = {
                "id": case["id"], "category": case["category"],
                "base_correct": strictly_correct(baseline_text, gold, parse_answer),
                "candidate_correct": strictly_correct(candidate_text, gold, parse_answer),
                "base_seconds": baseline_s,
                "candidate_seconds": candidate_s,
                "selection_policy": body.get("selection_policy", "unknown"),
                "source_verified": proof,
                "selected_extractor": body.get("selected_extractor") if proof else None,
                "baseline_error": baseline_error,
            }
            rows.append(row)
            rawfile.write(json.dumps({
                "id": case["id"], "question": prompt, "gold_integer": gold,
                "base_reply": baseline_text, "candidate_reply": candidate_text,
                "metadata": row}, ensure_ascii=False, sort_keys=True) + "\n")
            rawfile.flush()
            print("SEALED100_PROGRESS " + str(index) + "/100", flush=True)
    if len(rows) != 100:
        raise RuntimeError("Never grade incomplete 100-case evaluation")
    if local_tags(request) != tags_before:
        raise RuntimeError("Ollama inventory changed during evaluation")
    high.set_benchmark_seed(None)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--authorize", required=True)
    args = parser.parse_args()
    if args.authorize != "RUN_PR542_737413F0_SEALED100_ONCE":
        raise RuntimeError("Explicit one-shot authorization missing")
    if os.environ.get("NERAL_EXPERIMENTAL_NL_SCHEMA") != "1":
        raise RuntimeError("The actual PR542 opt-in NL route is not active")
    if args.out.exists() or args.out.is_symlink():
        raise FileExistsError("Refuse existing report destination")
    import scripts.neral_blind_100_cases as historical_generator
    from scripts.neral_generalization_100_run import (
        TUNED_TAG, TUNED_DIGEST, require_exact_guard,
    )
    from scripts.neral_qlora_holdout_live import request
    generator_source_sha = verify_historical_generator(historical_generator)
    cases = historical_generator.build()
    if len(cases) != 100 or {x["category"] for x in cases} != set(CATS):
        raise RuntimeError("Unexpected benchmark case structure")
    generated = historical_generator.serialize(cases)
    observed_sha = hashlib.sha256(generated).hexdigest()
    print("OFFICIALLY_RELOCKED_100_SHA256 " + observed_sha, flush=True)
    print("PREVIOUS_UNLOCATED_HASH_NOT_USED " + PREVIOUS_UNLOCATED_SHA, flush=True)
    if observed_sha != EXPECTED_SHA:
        raise RuntimeError("Immutable pre-PR542 LF corpus differs from authorized SHA; refuse inference")
    if Counter(x["category"] for x in cases) != {key: 25 for key in CATS}:
        raise RuntimeError("Benchmark categories are not 25/25/25/25")
    guard = require_exact_guard()
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True,
                         text=True, check=True).stdout.strip()
    if "RTX 4070" not in gpu:
        raise RuntimeError("Expected self-hosted RTX 4070 not found")
    tags_before = local_tags(request)
    if tags_before.get(TUNED_TAG) != TUNED_DIGEST:
        raise RuntimeError("QLoRA model digest differs from pinned PR542 expectation")
    if type(tags_before.get("qwen3.5:9b")) is not str:
        raise RuntimeError("Unmodified baseline tag qwen3.5:9b unavailable")
    args.out.mkdir(parents=True, exist_ok=False)
    rows = measure(cases, args.out, tags_before)
    groups = {}
    gains = regressions = 0
    for category in CATS:
        subset = [r for r in rows if r["category"] == category]
        a = sum(r["base_correct"] for r in subset)
        b = sum(r["candidate_correct"] for r in subset)
        gain = sum(not r["base_correct"] and r["candidate_correct"] for r in subset)
        loss = sum(r["base_correct"] and not r["candidate_correct"] for r in subset)
        gains += gain
        regressions += loss
        groups[category] = {
            "baseline": a, "candidate": b, "gained": gain,
            "regressions": loss, "cases": len(subset),
        }
    logic = [r for r in rows if r["category"] == "logic"]
    result = {
        "status": "PR542_PROSPECTIVE_100_SINGLE_QLORA_Z3_RTX4070_REVIEW_ONLY",
        "source_commit": SOURCE_SHA,
        "existing_sealed_sha256": observed_sha,
        "expected_sealed_sha256": EXPECTED_SHA,
        "historical_generator_commit": HISTORICAL_GENERATOR_COMMIT,
        "historical_generator_source_lf_sha256": generator_source_sha,
        "relocked_from_unlocated_sha256": PREVIOUS_UNLOCATED_SHA,
        "relock_basis": "Previously frozen pre-PR542 public deterministic generator; LF bytes; original seal not located",
        "evaluation_wrapper_commit": os.environ.get("GITHUB_SHA"),
        "github_actions_run_id": os.environ.get("GITHUB_RUN_ID"),
        "case_count": 100, "categories": groups,
        "baseline_model": "qwen3.5:9b",
        "baseline_digest": tags_before["qwen3.5:9b"],
        "baseline_inference_profile": "measured_nonthinking_2048",
        "matched_qwen_seed": 0,
        "candidate_api": "/api/neral/chat/think nl_schema_consensus=true",
        "qlora_model": TUNED_TAG, "qlora_digest": TUNED_DIGEST,
        "pr537_guard_blob": guard,
        "baseline_correct": sum(r["base_correct"] for r in rows),
        "candidate_correct": sum(r["candidate_correct"] for r in rows),
        "gained_cases": gains, "regressions": regressions,
        "net_gain": gains - regressions,
        "release_gate_passed": (regressions == 0 and gains >= 8),
        "release_gate_requirements": "0 regressions and >=8 gained cases",
        "logic_source_proof_count": sum(r["source_verified"] for r in logic),
        "logic_source_proof_rate": (
            sum(r["source_verified"] for r in logic) / 25),
        "logic_qlora_proofs": sum(r["source_verified"] and
                                   r["selected_extractor"] == "qlora" for r in logic),
        "logic_base_extractor_proofs": sum(r["source_verified"] and
                                            r["selected_extractor"] == "base"
                                            for r in logic),
        "selection_counts": dict(Counter(r["selection_policy"] for r in rows)),
        "baseline_latency_median_s": statistics.median(r["base_seconds"] for r in rows),
        "baseline_latency_max_s": max(r["base_seconds"] for r in rows),
        "candidate_latency_median_s": statistics.median(r["candidate_seconds"] for r in rows),
        "candidate_latency_max_s": max(r["candidate_seconds"] for r in rows),
        "raw_model_outputs_uploaded": False, "training": False,
        "merged_to_main": False,
        "full_raw_saved_locally": True,
        "note": "Observed 0 losses is not a future zero-regression guarantee; hash re-lock was expressly authorized after 3feca... was not recoverable.",
    }
    path = args.out / "summary.json"
    with path.open("x", encoding="utf8") as target:
        json.dump(result, target, ensure_ascii=False, sort_keys=True, indent=2)
        target.write("\n")
    print("SEALED100_FINAL_SCALARS " + json.dumps(result, sort_keys=True),
          flush=True)

if __name__ == "__main__":
    main()
