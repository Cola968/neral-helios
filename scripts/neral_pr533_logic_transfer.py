"""Four distinct logic-only GPU transfer cases for PR533's repair controller.

Development diagnostics only, NOT a blind holdout, not TIR12, no training.
The oracle is independent truth-table enumeration of public synthetic clauses.
Never log model-generated text, full prompts, JSON or model thoughts.
"""
from __future__ import annotations

import argparse
import http.client
from itertools import product
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

AUTH = "NERAL_PR533_LOGIC_TRANSFER_GPU_ONLY"
TARGET = "d280516e1a28a1cbdbc5b6b9b89d694769f6292f"
SEED = 202611110533
FORMAT_SUFFIX = "\nGib das Ergebnis in der letzten Zeile exakt als FINAL_ANSWER: <integer> aus."


def _true_count(m):
    return sum(m.values())


CASES = (
    {
        "id": "logic-transfer-01",
        "variables": "PQRST",
        "question": (
            "How many Boolean assignments to P,Q,R,S,T satisfy ALL: "
            "(P XOR R) is True; (Q OR S) is True; T = NOT R; "
            "S implies P; at least 2 of the five variables are True? "
            "Return one integer."
        ),
        "predicate": lambda m: (
            (m["P"] != m["R"]) and (m["Q"] or m["S"])
            and (m["T"] == (not m["R"]))
            and ((not m["S"]) or m["P"]) and _true_count(m) >= 2
        ),
    },
    {
        "id": "logic-transfer-02",
        "variables": "ABCDEF",
        "question": (
            "How many Boolean assignments to A,B,C,D,E,F satisfy ALL: "
            "(A AND B) is False; (C XOR E XOR F) is True; "
            "D = NOT F; B implies C; exactly 3 of the six variables "
            "are True? Return one integer."
        ),
        "predicate": lambda m: (
            not (m["A"] and m["B"])
            and ((m["C"] + m["E"] + m["F"]) % 2 == 1)
            and m["D"] == (not m["F"])
            and ((not m["B"]) or m["C"]) and _true_count(m) == 3
        ),
    },
    {
        "id": "logic-transfer-03",
        "variables": "ABCDEFG",
        "question": (
            "How many Boolean assignments to A,B,C,D,E,F,G satisfy ALL: "
            "(A OR G) is True; (B AND C) is False; "
            "(D XOR F) is False; E implies B; G = NOT C; "
            "at least 3 of the seven variables are True? Return one integer."
        ),
        "predicate": lambda m: (
            (m["A"] or m["G"]) and not (m["B"] and m["C"])
            and (m["D"] == m["F"]) and ((not m["E"]) or m["B"])
            and m["G"] == (not m["C"]) and _true_count(m) >= 3
        ),
    },
    {
        "id": "logic-transfer-04",
        "variables": "XYZUV",
        "question": (
            "How many Boolean assignments to X,Y,Z,U,V satisfy ALL: "
            "(X XOR Y XOR Z) is False; (U OR V) is True; "
            "X implies U; V = NOT Y; "
            "exactly 3 of the five variables are True? Return one integer."
        ),
        "predicate": lambda m: (
            (m["X"] + m["Y"] + m["Z"]) % 2 == 0
            and (m["U"] or m["V"]) and ((not m["X"]) or m["U"])
            and m["V"] == (not m["Y"]) and _true_count(m) == 3
        ),
    },
)


def count_matches(case: dict) -> int:
    names = case["variables"]
    if len(names) != len(set(names)) or not 4 <= len(names) <= 7:
        raise ValueError("Invalid test variable set")
    return sum(
        bool(case["predicate"](dict(zip(names, bits))))
        for bits in product((False, True), repeat=len(names))
    )


def evaluated_packet(packet: object, http_status: int, expected: int) -> dict:
    from scripts.neral_tir_hard_live import safe_first_turn_event
    p = packet if type(packet) is dict else {}
    event = safe_first_turn_event(p.get("first_turn_telemetry"))
    event = event if type(event) is dict else {}
    code = p.get("error_code")
    if code not in (
        "token_limit", "incomplete", "empty_answer", "timeout",
        "invalid_response", "model_error", "internal_error"
    ):
        code = None
    extraction = p.get("extraction_status")
    if extraction not in ("accepted", "repaired", "fallback"):
        extraction = None
    passed = (
        http_status == 200 and p.get("text") == f"FINAL_ANSWER: {expected}"
        and p.get("verified") is False
        and p.get("verification_status") == "model_extracted_schema_unverified"
        and extraction in ("accepted", "repaired")
        and p.get("tool_used") == "neral_logic_exact"
        and p.get("tool_calls_succeeded") == 1
    )
    return {
        "passed": bool(passed),
        "http_status": http_status,
        "error_code": code,
        "extraction_status": extraction,
        "solver_repair_status": event.get("solver_repair_status"),
        "solver_repair_input_format": event.get("solver_repair_input_format"),
        "solver_repair_rejection_reason": event.get("solver_repair_rejection_reason"),
        "solver_repair_solver_detail": event.get("solver_repair_solver_detail"),
        "first_extraction_rejection": event.get("extraction_rejection_reason"),
        "first_solver_detail": event.get("solver_rejection_detail"),
    }


def model_vram_bytes() -> int:
    conn = http.client.HTTPConnection("127.0.0.1", 11434, timeout=5)
    try:
        conn.request("GET", "/api/ps")
        r = conn.getresponse()
        if r.status != 200:
            return 0
        payload = r.read(65536)
        rows = json.loads(payload).get("models", [])
        if type(rows) is not list:
            return 0
        return max((x["size_vram"] for x in rows if type(x) is dict
                    and x.get("name") == "qwen3.5:9b"
                    and type(x.get("size_vram")) is int), default=0)
    finally:
        conn.close()


def run_suite(root: Path) -> dict:
    from scripts.neral_foundation_2048_retest import ORIGINAL_DIGESTS
    from scripts.neral_high12_manual import MODEL, PROFILE
    from scripts.neral_same_base_live import require_model, _high_client
    from scripts.neral_v7_foundation_tournament import OllamaLocal
    from neral_agent_autonomy import OllamaLocalBackend

    if MODEL != "qwen3.5:9b" or PROFILE != "measured_nonthinking_2048":
        raise ValueError("Wrong model profile")
    digest = require_model(OllamaLocal(timeout=20).inventory())
    if digest != ORIGINAL_DIGESTS[MODEL]:
        raise ValueError("Pinned model digest mismatch")
    backend = OllamaLocalBackend(MODEL, timeout=90, inference_profile=PROFILE)
    rows = []
    peak_model_vram = 0
    decoder_seed = SEED % (2 ** 31)
    started = time.monotonic()
    try:
        client, headers = _high_client(backend)
        for index, case in enumerate(CASES):
            expected = count_matches(case)
            backend.set_benchmark_seed(decoder_seed + 101 + index)
            start = time.monotonic()
            r = client.post(
                "/api/neral/chat/think",
                json={"prompt": case["question"] + FORMAT_SUFFIX,
                      "experimental_tir": True},
                base_url="http://127.0.0.1:9690",
                headers=headers,
            )
            result = evaluated_packet(r.get_json(silent=True), r.status_code, expected)
            result.update(case_id=case["id"], expected_count=expected,
                          elapsed_seconds=round(time.monotonic() - start, 3))
            rows.append(result)
            peak_model_vram = max(peak_model_vram, model_vram_bytes())
            print("NERAL_PR533_TRANSFER_PROGRESS " + str(index + 1) + "/4", flush=True)
    finally:
        backend.set_benchmark_seed(None)
        backend.unload()
    offloaded = peak_model_vram >= 4 * 1024 ** 3
    summary = {
        "target_sha": TARGET,
        "outer_seed": SEED,
        "decoder_seed_rule": "seed_mod_2pow31_plus_101_plus_index",
        "model": MODEL,
        "gpu_offload_observed": offloaded,
        "model_vram_mib": peak_model_vram // 1048576,
        "case_count": len(rows),
        "correct_formal_tool_cases": sum(x["passed"] for x in rows),
        "all_pass": len(rows) == 4 and all(x["passed"] for x in rows) and offloaded,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "cases": rows,
        "benchmark_kind": "development_synthetic_transfer_not_independent_holdout",
        "weights_updated": False,
        "docker_used": False,
        "tir12_passed": False,
    }
    return summary


def main():
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--authorize", required=True)
    p.add_argument("--target-sha", required=True)
    p.add_argument("--candidate-root", required=True, type=Path)
    args = p.parse_args()
    root = args.candidate_root
    actual = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True, timeout=8,
    ).stdout.strip()
    if (args.authorize != AUTH or args.target_sha != TARGET
            or actual != TARGET or os.environ.get("NERAL_EXPERIMENTAL_TIR") != "1"):
        raise ValueError("Unauthorized or non-pinned PR533 transfer")
    sys.path.insert(0, str(root.resolve()))
    result = run_suite(root)
    print("NERAL_PR533_TRANSFER_SCALARS "
          + json.dumps(result, sort_keys=True, allow_nan=False), flush=True)
    return 0 if result["all_pass"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("NERAL_PR533_TRANSFER_ABORT " + type(exc).__name__, flush=True)
        raise SystemExit(2)
