"""Manual RTX 4070 two-gate probe for PR533 (test harness branch only).

Only local Ollama and the pinned candidate checkout are used. No training,
code execution outside the existing Docker sandbox, or private model output
is uploaded. This is a developmental two-case gate, not a blind benchmark.
"""
from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import subprocess
import sys
import time

AUTH = "NERAL_PR533_TWO_GATE_GPU_MANUAL_ONLY"
TARGET = "6d71eacd793820b586754d13f18be1e743cf01fc"
OUTER_SEED = 202611110533
SUFFIX = "\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."
SOURCE = (
    "data = [2, -32, -70, 50, 80, 25, 0, -9, -81]\n"
    "for i in range(len(data)):\n"
    "    if i % 2:\n"
    "        data[i] = data[i] * 2 + i\n"
    "    else:\n"
    "        data[i] -= i * 5\n"
    "print(sum(data[::2])*2 - sum(data[1::2]))"
)
TASK_A = ("What exact integer does this Python program print? "
          "You may execute the program in an offline sandbox:\n" + SOURCE + SUFFIX)
TASK_B = (
    "How many Boolean assignments to A,B,C,D,E,F,G satisfy ALL: "
    "(A XOR C XOR F) is False; (B OR D) is True; E = NOT F; "
    "(A AND G) is False; C implies G; "
    "at least 2 of the seven variables are True? Return one integer."
    + SUFFIX
)


def checkout_ok(root: Path) -> bool:
    p = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                       capture_output=True, text=True, check=True, timeout=8)
    return p.stdout.strip() == TARGET


def gpu_resident_bytes() -> int:
    conn = http.client.HTTPConnection("127.0.0.1", 11434, timeout=6)
    try:
        conn.request("GET", "/api/ps")
        response = conn.getresponse()
        if response.status != 200:
            return 0
        data = response.read(65536)
        model_list = json.loads(data).get("models", [])
        if type(model_list) is not list:
            return 0
        return max(
            (row.get("size_vram", 0)
             for row in model_list if type(row) is dict
             and row.get("name") == "qwen3.5:9b"
             and type(row.get("size_vram")) is int),
            default=0,
        )
    finally:
        conn.close()


def run_gate(client, headers, task: str, check: str) -> dict:
    from scripts.neral_tir_hard_live import safe_first_turn_event
    started = time.monotonic()
    response = client.post(
        "/api/neral/chat/think", json={"prompt": task, "experimental_tir": True},
        base_url="http://127.0.0.1:9690", headers=headers)
    packet = response.get_json(silent=True)
    if type(packet) is not dict:
        packet = {}
    status = packet.get("verification_status")
    extraction = packet.get("extraction_status")
    actual = packet.get("text")
    event = safe_first_turn_event(packet.get("first_turn_telemetry"))
    if check == "A":
        passed = (
            response.status_code == 200
            and actual == "FINAL_ANSWER: -422"
            and packet.get("verified") is True
            and packet.get("review_method") == "exact_user_ast_match"
            and packet.get("tool_calls_succeeded") == 1
            and packet.get("verification_scope") == "exact_user_python_execution")
    else:
        passed = (
            response.status_code == 200
            and actual == "FINAL_ANSWER: 12"
            and packet.get("tool_used") == "neral_logic_exact"
            and packet.get("tool_calls_succeeded") == 1
            and packet.get("verified") is False
            and packet.get("verification_status")
                 == "model_extracted_schema_unverified"
            and extraction in ("accepted", "repaired"))
    return {
        "case": check,
        "passed": passed,
        "http_status": response.status_code,
        "error_code": packet.get("error_code") if packet.get("error_code") in (
            "token_limit", "incomplete", "empty_answer", "timeout",
            "invalid_response", "model_error", "internal_error") else None,
        "first_turn_scalars": event,
        "seconds": round(time.monotonic() - started, 3),
        "verification_status": status if status in (
            "verified", "model_extracted_schema_unverified",
            "base_fallback_unverified", "not_available") else "other",
        "extraction_status": extraction if extraction in (
            "accepted", "repaired", "fallback") else "other",
        "solver_repair_status": event.get("solver_repair_status") if event else None,
        "solver_repair_input_format": event.get("solver_repair_input_format") if event else None,
        "solver_repair_rejection_reason": event.get("solver_repair_rejection_reason") if event else None,
        "solver_repair_solver_detail": event.get("solver_repair_solver_detail") if event else None,
        "solver_repair_ast_check_code": event.get("solver_repair_ast_check_code") if event else None,
        "tool_calls_succeeded": packet.get("tool_calls_succeeded") if type(packet.get("tool_calls_succeeded")) is int else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--authorize", required=True)
    parser.add_argument("--target-sha", required=True)
    parser.add_argument("--candidate-root", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    args = parser.parse_args()
    if (args.authorize != AUTH or args.target_sha != TARGET
            or args.seed != OUTER_SEED
            or os.environ.get("NERAL_EXPERIMENTAL_TIR") != "1"
            or not checkout_ok(args.candidate_root)):
        print("PR533_TWO_GATE_REFUSED", flush=True)
        return 3
    sys.path.insert(0, str(args.candidate_root.resolve()))
    from scripts.neral_foundation_2048_retest import ORIGINAL_DIGESTS
    from scripts.neral_high12_manual import MODEL, PROFILE
    from scripts.neral_same_base_live import require_model, _high_client
    from scripts.neral_v7_foundation_tournament import OllamaLocal
    from neral_agent_autonomy import OllamaLocalBackend

    if MODEL != "qwen3.5:9b" or PROFILE != "measured_nonthinking_2048":
        print("PR533_TWO_GATE_WRONG_MODEL_PROFILE", flush=True)
        return 3
    digest = require_model(OllamaLocal(timeout=20).inventory())
    if digest != ORIGINAL_DIGESTS[MODEL]:
        print("PR533_TWO_GATE_WRONG_DIGEST", flush=True)
        return 3
    backend = OllamaLocalBackend(MODEL, timeout=90, inference_profile=PROFILE)
    decoder_seed = args.seed % (2**31)  # Ollama measured profile allows <= 31 bits.
    output = []
    vram_bytes = 0
    try:
        client, headers = _high_client(backend)
        # Logic-only: Docker Desktop is unavailable. NEVER run original user
        # Python or substitute host execution for the isolated sandbox.
        backend.set_benchmark_seed(decoder_seed + 1)
        output.append(run_gate(client, headers, TASK_B, "B"))
        print("PR533_LOGIC_ONLY_PROGRESS B", flush=True)
        vram_bytes = gpu_resident_bytes()
    finally:
        backend.set_benchmark_seed(None)
        backend.unload()
    offload_seen = vram_bytes >= 4 * 1024**3
    success = len(output) == 1 and output[0]["passed"] and offload_seen
    print("PR533_LOGIC_ONLY_SCALARS " + json.dumps({
        "target_sha": TARGET, "seed": args.seed,
        "decoder_seed": decoder_seed + 1, "model": MODEL,
        "gpu_offload_observed": offload_seen,
        "resident_model_vram_mib": vram_bytes // 1048576,
        "case_a_run": False,
        "case_b_pass": output[0]["passed"],
        "case_b": output[0],
        "logic_gpu_gate_pass": success,
        "tir12_authorized_by_this_run": False,
        "docker_bypass": False,
    }, sort_keys=True, allow_nan=False), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("PR533_TWO_GATE_ABORT " + type(exc).__name__, flush=True)
        raise SystemExit(2)
