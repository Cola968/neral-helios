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

EXPECTED_SHA = "3feca67d6ee9fd48b462bdbf3e15e256b9e72a1a35357d30d75d383b9a7b280d"
SOURCE_SHA = "737413f03b347676108d507ff3fe7693975dd269"
CATS = ("mathematics", "logic", "programming", "mixed")

def find_sealed_corpus(home: Path) -> Path:
    """Hash likely pre-existing sealed JSONL files; never create a substitute."""
    roots = [p for p in home.iterdir()
             if p.is_dir() and not p.is_symlink()
             and p.name.lower().startswith("neral-")]
    roots += [home / "Documents" / "Codex", home / "Documents" / "NERAL"]
    omitted = {".git", ".venv", "venv", "site-packages", "node_modules",
               ".cache", "cache", "checkpoints", "adapter", "adapters",
               "models", "ollama", "training-env-py312"}
    matches = []
    seen = set()
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for directory, dirs, files in os.walk(root, followlinks=False):
            relative = Path(directory).relative_to(root)
            if len(relative.parts) > 6:
                dirs[:] = []
                continue
            dirs[:] = [d for d in dirs if d.lower() not in omitted
                       and not (Path(directory) / d).is_symlink()]
            for filename in files:
                p = Path(directory) / filename
                if p.suffix.lower() != ".jsonl" or p.is_symlink():
                    continue
                if not any(key in str(p).lower()
                           for key in ("blind", "prospective", "100", "sealed")):
                    continue
                try:
                    if not 4000 <= p.stat().st_size <= 2_000_000:
                        continue
                    if p in seen:
                        continue
                    seen.add(p)
                    if hashlib.sha256(p.read_bytes()).hexdigest() == EXPECTED_SHA:
                        matches.append(p)
                except (OSError, PermissionError):
                    continue
    if not matches:
        raise RuntimeError("SEALED_DATASET_NOT_FOUND_OR_HASH_MISMATCH: "
                           "no existing user-profile JSONL matches pinned SHA")
    return sorted(matches, key=str)[0]

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
    from scripts.neral_blind_100_cases import build, serialize
    from scripts.neral_generalization_100_run import (
        TUNED_TAG, TUNED_DIGEST, require_exact_guard,
    )
    from scripts.neral_qlora_holdout_live import request
    cases = build()
    if len(cases) != 100 or {x["category"] for x in cases} != set(CATS):
        raise RuntimeError("Unexpected benchmark case structure")
    generated = serialize(cases)
    generated_sha = hashlib.sha256(generated).hexdigest()
    if generated_sha != EXPECTED_SHA:
        raise RuntimeError("Frozen generator hash differs: " + generated_sha)
    existing = find_sealed_corpus(Path.home())
    original = existing.read_bytes()
    observed_sha = hashlib.sha256(original).hexdigest()
    print("SEALED100_VERIFIED_SHA256 " + observed_sha, flush=True)
    if observed_sha != EXPECTED_SHA or original != generated:
        raise RuntimeError("Existing sealed dataset does not match immutable generator")
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
        "case_count": 100, "categories": groups,
        "baseline_model": "qwen3.5:9b",
        "baseline_digest": tags_before["qwen3.5:9b"],
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
        "note": "Observed 0 losses is not a future zero-regression guarantee.",
    }
    path = args.out / "summary.json"
    with path.open("x", encoding="utf8") as target:
        json.dump(result, target, ensure_ascii=False, sort_keys=True, indent=2)
        target.write("\n")
    print("SEALED100_FINAL_SCALARS " + json.dumps(result, sort_keys=True),
          flush=True)

if __name__ == "__main__":
    main()
