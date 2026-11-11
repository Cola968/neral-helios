"""Manual, private, 36-case HARD development A/B/C GPU diagnostic.

A=raw Qwen3.5, B=current HIGH with suffix normalization, C=opt-in
experimental TIR. Six parametrically new challenging families are NOT an
independent blind holdout. No training, downloads, automatic release promotion,
prompt uploads, or shared GPU runs. The workflow must verify a GREEN release
at the exact SHA before launching this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import secrets
import time

from scripts.neral_foundation_2048_retest import (
    ORIGINAL_DIGESTS, SampledVram, _read_gpu_memory,
)
from scripts.neral_same_base_ab_score import (
    ARMS, MODEL, PROFILE, SameBaseError, score_private_pairs, _correct,
)
from scripts.neral_same_base_live import (
    BASE_URL, PilotBlocked, checked_high_packet, observation,
    _high_client, release_gate, require_model,
)
from scripts.neral_low_high_quality import make_prompt

AUTHORIZATION = "NERAL_TIR_HARD36_GPU_MANUAL_ONLY"
STATUS = "NERAL_TIR_HARD36_DEVELOPMENT_ONLY"


def _case(domain: str, family: str, index: int, prompt: str, answer: int) -> dict:
    return {"id": f"hard-{domain}-{family}-{index:02d}",
            "domain": domain, "family": family, "kind": "integer",
            "prompt": prompt, "answer": answer}


def generate_hard_cases(seed: int) -> list[dict]:
    """Six independently computed families; 12 cases/domain and 6/family."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise PilotBlocked("Invalid private hard-case seed")
    rng = random.Random(seed)
    rows = []
    for index in range(6):
        a, c = (rng.randint(20_000, 900_000) for _ in range(2))
        b, d = (rng.randint(7, 37) for _ in range(2))
        extra = rng.randint(10_000, 600_000)
        gold = a*d + c*b + extra
        assert Fraction(a, b)*b*d + Fraction(c, d)*b*d + extra == gold
        rows.append(_case(
            "math", "nested_fraction_integer", index,
            f"Calculate exactly (({a}/{b}) + ({c}/{d})) * {b} * {d} + {extra}. "
            "Use exact arithmetic, return one integer.", gold))

        origin_min = rng.choice((-480, -300, -210, 0, 60, 330, 345, 540))
        dest_min = rng.choice((-600, -420, -300, 0, 120, 330, 480, 570))
        departure_min = rng.randint(0, 1439)
        flight_min = rng.randint(280, 1150)
        result_min = (departure_min - origin_min + flight_min + dest_min) % 1440
        origin_tz = timezone(timedelta(minutes=origin_min))
        dest_tz = timezone(timedelta(minutes=dest_min))
        departure = datetime(2026, 11, 17,
                             departure_min//60, departure_min%60,
                             tzinfo=origin_tz)
        arrival = (departure + timedelta(minutes=flight_min)).astimezone(dest_tz)
        assert arrival.hour*60 + arrival.minute == result_min
        rows.append(_case(
            "math", "fixed_utc_flight", index,
            f"An aircraft departs at {departure_min//60:02d}:{departure_min%60:02d} "
            f"local time in a fixed UTC{origin_min/60:+g} zone and flies "
            f"{flight_min} minutes to a fixed UTC{dest_min/60:+g} zone. "
            "What is its destination-local clock time as an integer number "
            "of minutes after midnight (0..1439)? Ignore daylight savings.",
            result_min))

        options = [(u, v) for u in range(7) for v in range(u+1, 7)]
        edges = sorted(rng.sample(options, 7))
        count = 0
        for order in itertools.permutations(range(7)):
            at = {x: i for i, x in enumerate(order)}
            count += all(at[u] < at[v] for u, v in edges)
        dp = [0]*(1<<7)
        dp[0] = 1
        for mask in range(1<<7):
            for v in range(7):
                if mask & (1<<v):
                    continue
                if all(mask & (1<<u) for u, w in edges if w == v):
                    dp[mask | (1<<v)] += dp[mask]
        assert count == dp[-1]
        rules = "; ".join(f"{chr(65+u)} before {chr(65+v)}" for u,v in edges)
        rows.append(_case(
            "reasoning", "seven_task_orderings", index,
            f"Seven distinct tasks A,B,C,D,E,F,G are ordered. The ONLY "
            f"precedence constraints are: {rules}. How many complete valid "
            "orders exist? Return one integer.", count))

        xor_bit = rng.randint(0, 1)
        # Six distinct Boolean problem variants; avoid duplicate prompts.
        threshold = index + 1
        def consistent(bits):
            A, B, C, D, E, F, G = bits
            return ((A ^ C ^ F) == xor_bit and
                    (B or D) and (E == (not F)) and
                    not (A and G) and (not C or G) and
                    sum(bits) >= threshold)
        valid = sum(bool(consistent(bits)) for bits in
                    itertools.product((False, True), repeat=7))
        assert valid == sum(bool(consistent(
            tuple(bool(mask & (1<<i)) for i in range(7))))
            for mask in range(128))
        rows.append(_case(
            "reasoning", "seven_variable_logic", index,
            f"How many Boolean assignments to A,B,C,D,E,F,G satisfy ALL: "
            f"(A XOR C XOR F) is {bool(xor_bit)}; (B OR D) is True; "
            "E = NOT F; (A AND G) is False; C implies G; "
            f"at least {threshold} of the seven variables are True? "
            "Return one integer.", valid))

        nums = [rng.randint(400, 4_000) for _ in range(rng.randint(5, 9))]
        multiplier = rng.randint(3, 19)
        gold_debug = sum(nums) + multiplier*len(nums)*(len(nums)-1)//2
        corrected = [n+multiplier*i for i,n in enumerate(nums)]
        assert sum(corrected) == gold_debug
        broken = (
            f"values = {nums!r}\n"
            f"for i in range(len(values) + 1):\n"
            f"    values[i] += {multiplier} * i\n"
            "print(sum(values))"
        )
        rows.append(_case(
            "code_trace", "debug_off_by_one", index,
            "This Python code has an off-by-one error and throws IndexError. "
            "Fix ONLY the loop bound, then report the integer it prints. "
            "You may run corrected code in an offline sandbox:\n"+broken,
            gold_debug))

        state = [rng.randint(-100, 100) for _ in range(9)]
        k = rng.randint(2, 7)
        updated = [
            (v*2+i if i%2 else v-i*k)
            for i,v in enumerate(state)
        ]
        expected = sum(updated[::2])*2 - sum(updated[1::2])
        checked = state[:]
        for i in range(len(checked)):
            if i % 2:
                checked[i] = checked[i]*2+i
            else:
                checked[i] = checked[i]-i*k
        assert expected == sum(checked[::2])*2-sum(checked[1::2])
        code = (
            f"data = {state!r}\n"
            "for i in range(len(data)):\n"
            "    if i % 2:\n"
            "        data[i] = data[i] * 2 + i\n"
            "    else:\n"
            f"        data[i] -= i * {k}\n"
            "print(sum(data[::2])*2 - sum(data[1::2]))"
        )
        rows.append(_case(
            "code_trace", "nested_mutation_trace", index,
            "What exact integer does this Python program print? "
            "You may execute the program in an offline sandbox:\n"+code,
            expected))
    counts = Counter(c["domain"] for c in rows)
    if (len(rows) != 36 or counts != {"math":12,"reasoning":12,"code_trace":12}
            or len({c["id"] for c in rows}) != 36
            or len({c["prompt"] for c in rows}) != 36
            or len({(c["domain"],c["family"]) for c in rows}) != 6):
        raise PilotBlocked("Incomplete hard36 case families")
    return rows


def validated_tool_counts(packet: dict) -> tuple[int, int]:
    if not isinstance(packet, dict):
        raise ValueError("Bad tool evidence")
    triggered = packet.get("tool_calls_triggered")
    succeeded = packet.get("tool_calls_succeeded")
    if (type(triggered) is not int or type(succeeded) is not int or
            not 0 <= succeeded <= triggered <= 2):
        raise ValueError("Invalid experimental tool evidence")
    return triggered, succeeded


# Only transient local inference failures are eligible for retry. A model
# exhausting 2,048 output tokens is a TERMINAL completion, not a server outage.
_TRANSIENT_FAILURES = frozenset({
    "timeout", "incomplete", "model_error", "server_503",
})


def safe_first_turn_event(event: object) -> dict | None:
    """Safe, scalar-only log payload; discard all model text and user content."""
    if type(event) is not dict:
        return None
    counts = {}
    for key in ("eval_count", "prompt_eval_count"):
        value = event.get(key)
        if value is not None and (
                type(value) is not int or not 0 <= value <= 1_000_000):
            return None
        counts[key] = value
    reason = event.get("done_reason")
    decision = event.get("decision")
    if (reason not in (None, "stop", "length", "load", "unload", "other")
            or decision not in (
                "tool", "final", "unstructured",
                "deterministic_formal", "deterministic_rejected",
                "extracted_formal", "extraction_fallback",
                "extraction_none", "unavailable_on_failure")):
        return None
    from neral_turn0_extractor import EXTRACTION_RESULT_CODES
    rejected = event.get("extraction_rejection_reason")
    if rejected is not None and rejected not in EXTRACTION_RESULT_CODES:
        return None
    return {**counts, "done_reason": reason, "decision": decision,
            **({"extraction_rejection_reason": rejected}
               if rejected is not None else {})}


def _run_one(arm: str, question: str, backend, client, headers: dict):
    from neral_agent_autonomy import AgentModelError, OllamaCompletionError
    from neral_identity import build_identity_prompt
    started = time.monotonic()
    if arm == "raw_qwen35":
        turns = [
            {"role": "system", "content": build_identity_prompt(
                "web", "local_foundation_not_neral_adapter")},
            {"role": "user", "content": question},
        ]
        try:
            result = backend.respond_chat(turns)
        except OllamaCompletionError as exc:
            # This is a local inference failure, NEVER an HTTP 503.
            return "", False, True, time.monotonic()-started, None, 0, 0, exc.reason
        except AgentModelError:
            return "", False, True, time.monotonic()-started, None, 0, 0, "model_error"
        if not isinstance(result, str) or not result.strip():
            raise PilotBlocked("Raw model returned empty response")
        return result, True, False, time.monotonic()-started, None, 0, 0, None
    if arm not in ("neral_high", "neral_high_router"):
        raise PilotBlocked("Unknown arm")
    payload = {"prompt": question}
    if arm == "neral_high_router":
        payload["experimental_tir"] = True
    # Synchronous Flask test client, no async tasks and no concurrent model calls.
    response = client.post("/api/neral/chat/think", json=payload,
                           base_url=BASE_URL, headers=headers)
    seconds = time.monotonic()-started
    if response.status_code == 503:
        packet = response.get_json(silent=True)
        if arm == "neral_high_router" and type(packet) is dict:
            event = safe_first_turn_event(packet.get("first_turn_telemetry"))
            if event is not None:
                print("NERAL_TIR_FIRST_TURN_SCALARS " +
                      json.dumps(event, sort_keys=True), flush=True)
        code = packet.get("error_code") if isinstance(packet, dict) else None
        if code not in {
            "token_limit", "incomplete", "empty_answer",
            "timeout", "invalid_response", "model_error", "internal_error",
        }:
            code = "server_503"
        return "", False, True, seconds, None, 0, 0, code
    if response.status_code != 200:
        raise PilotBlocked("HIGH gate denied or failed")
    packet = response.get_json(silent=True)
    checked_high_packet(packet)
    if arm == "neral_high_router":
        triggered, succeeded = validated_tool_counts(packet)
        if packet.get("experimental_tir") is not True:
            raise PilotBlocked("C arm missing experimental provenance")
        event = safe_first_turn_event(packet.get("first_turn_telemetry"))
        if event is not None:
            print("NERAL_TIR_FIRST_TURN_SCALARS " +
                  json.dumps(event, sort_keys=True), flush=True)
    else:
        triggered = succeeded = int(packet.get("tool_used") is not None)
    return (packet["text"], True, False, seconds, packet.get("tool_used"),
            triggered, succeeded, None)


def run_one_with_backoff(arm: str, question: str, backend, client, headers: dict,
                         *, sleeper=time.sleep):
    """At most two sequential retries for transient failures; never for truncation.

    Reuse the same backend/model and exactly the same prompt and decoding
    parameters. No retries for completed but wrong model answers.
    """
    started = time.monotonic()
    for attempt in range(3):
        result = _run_one(arm, question, backend, client, headers)
        if (not result[2] or result[-1] not in _TRANSIENT_FAILURES
                or attempt == 2):
            return (*result[:3], time.monotonic()-started, *result[4:])
        sleeper(0.25 * (2**attempt))
    raise AssertionError("unreachable")


def run_live_hard(*, output_dir: Path, receipt: dict, authorization: str) -> dict:
    if authorization != AUTHORIZATION or os.environ.get("NERAL_EXPERIMENTAL_TIR") != "1":
        raise PilotBlocked("Manual TIR GPU consent and env feature-gate required")
    release_gate(receipt)
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)
    if (output_dir.exists() or output_dir.is_symlink() or
            not output_dir.parent.is_dir() or output_dir.parent.is_symlink() or
            output_dir.resolve().is_relative_to(root)):
        raise PilotBlocked("Private output must be new and outside repository")
    from neral_agent_autonomy import OllamaLocalBackend
    require_model(__import__("scripts.neral_v7_foundation_tournament",
                             fromlist=["OllamaLocal"]).OllamaLocal(timeout=120).inventory())
    used, total = _read_gpu_memory()
    if total-used < 7000:
        raise PilotBlocked("GPU occupied: no concurrent inference")
    backend = OllamaLocalBackend(MODEL, timeout=90, inference_profile=PROFILE)
    client, headers = _high_client(backend)
    cases = generate_hard_cases(secrets.randbits(64))
    observations=[]
    extra={arm:{"tool_calls_triggered":0,"tool_calls_succeeded":0}
           for arm in ARMS}
    failures={arm:Counter() for arm in ARMS}
    matrix_private=[]
    output_dir.mkdir(exist_ok=False)
    try:
        with SampledVram() as sampler:
            for case in cases:
                prompt=make_prompt(case)
                order=list(ARMS)
                random.Random(int.from_bytes(hashlib.sha256(
                    case["id"].encode("ascii")).digest()[:8],"big")).shuffle(order)
                arms={}
                for arm in order:
                    text,complete,failed,seconds,tool,triggered,succeeded,error_code = run_one_with_backoff(
                        arm,prompt,backend,client,headers)
                    arms[arm] = observation(prompt,text,complete,failed,seconds,tool)
                    if failed:
                        failures[arm][error_code or "unknown"] += 1
                    extra[arm]["tool_calls_triggered"] += triggered
                    extra[arm]["tool_calls_succeeded"] += succeeded
                observations.append({"id":case["id"],"arms":arms})
                match = {arm:_correct(case["answer"],arms[arm]["text"],
                                       arms[arm]["complete"]) for arm in ARMS}
                matrix_private.append({"id":case["id"],"domain":case["domain"],
                    "tir_win_vs_raw":bool(match["neral_high_router"] and
                                          not match["raw_qwen35"]),
                    "tir_loss_vs_raw":bool(match["raw_qwen35"] and
                                           not match["neral_high_router"])})
                if len(observations)%6==0:
                    print(f"TIR_HARD36_PRIVATE_PROGRESS {len(observations)}/36",flush=True)
            vram=sampler.report()
    finally:
        backend.unload()
    private={"manifest":{
        "model": MODEL, "model_digest":ORIGINAL_DIGESTS[MODEL],
        "profile":PROFILE,"token_budget":2048},
        "cases":[{"id":c["id"],"domain":c["domain"],"family":c["family"],
                  "prompt":make_prompt(c),"answer":c["answer"]} for c in cases],
        "observations":observations}
    (output_dir/"hard36-private.json").write_text(
        json.dumps({"scorer_input":private,"wins_losses":matrix_private},
                   ensure_ascii=False),encoding="utf8")
    try:
        score=score_private_pairs(private)
    except SameBaseError as exc:
        raise PilotBlocked("Unusable hard36 paired scoring") from exc
    score["collector_status"]=STATUS
    # The legacy scorer calls all failures 'http_errors', even for raw
    # local Ollama. Publish sanitized reasons separately, never prompt text.
    score["failure_classes"]={arm:dict(failures[arm]) for arm in ARMS}
    score["legacy_http_errors_include_raw_local_failures"]=True
    score["tool_counters"]=extra
    score["gpu_telemetry"]=vram
    score["observed_peak_vram_gib"]=round(vram["peak_used_mib"]/1024,3)
    score["development_only"]=True
    score["individual_private_wins"] = sum(x["tir_win_vs_raw"] for x in matrix_private)
    score["individual_private_losses"] = sum(x["tir_loss_vs_raw"] for x in matrix_private)
    score["win_loss_by_domain"]={
        d:{"wins":sum(x["tir_win_vs_raw"] for x in matrix_private
                     if x["domain"]==d),
           "losses":sum(x["tir_loss_vs_raw"] for x in matrix_private
                       if x["domain"]==d)}
        for d in ("math","reasoning","code_trace")}
    (output_dir/"scalar-summary.json").write_text(
        json.dumps(score,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps({
        "status":STATUS,
        "correct":{arm:score["arms"][arm]["correct"] for arm in ARMS},
        "tool_counters":extra,
        "observed_peak_vram_gib":score["observed_peak_vram_gib"],
        "independent_holdout_verified":False,
        "weight_improvement_proven":False,
    },sort_keys=True),flush=True)
    return score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--private-output-dir",required=True,type=Path)
    p.add_argument("--release-receipt-json",required=True,type=Path)
    p.add_argument("--authorize",required=True)
    a=p.parse_args()
    receipt=json.loads(a.release_receipt_json.read_text(encoding="utf8"))
    run_live_hard(output_dir=a.private_output_dir,receipt=receipt,
                  authorization=a.authorize)


if __name__=="__main__":
    main()
