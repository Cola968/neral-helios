"""New, seeded 36-case CPU-oracle NERAL LOW/HIGH paired development probe.

A private random seed is generated on the GPU PC just before evaluation.
No cases, seed, raw answers, user data or training weights leave the PC.
The task structures are published and therefore NOT an independent blinded
release holdout or proof of generally superior neural weights.
"""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import secrets
import time

from scripts.neral_low_high_quality import (
    QualityGateError, _MODEL, _ORIGIN, _DIGEST, make_prompt, score_paired
)
from scripts.neral_v7_foundation_tournament import OllamaLocal


class FreshEvalError(ValueError):
    """Deliberately excludes private prompts, file paths and raw responses."""


def _case(domain: str, family: str, i: int, question: str, gold: int) -> dict:
    if (type(gold) is not int or abs(gold) > 1_000_000
            or not isinstance(question, str) or not 20 <= len(question) <= 1800):
        raise FreshEvalError("Invalid private oracle task")
    return {
        "id": f"fresh-{domain}-{family}-{i:02d}",
        "domain": domain, "family": family,
        "kind": "integer", "answer": gold, "prompt": question,
    }


def generate_fresh_cases(seed: int) -> list[dict]:
    """Six fresh parameterized families, each independently oracle-checked."""
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise FreshEvalError("Invalid local random seed")
    rng = random.Random(seed)
    cases = []
    for i in range(6):
        # Mathematics 1: obstacle-avoiding monotone lattice paths.
        width, height = rng.randint(4, 7), rng.randint(4, 7)
        bx, by = rng.randint(1, width-1), rng.randint(1, height-1)
        table = [[0] * (height + 1) for _ in range(width + 1)]
        for x in range(width + 1):
            for y in range(height + 1):
                if (x, y) == (bx, by):
                    continue
                if (x, y) == (0, 0):
                    table[x][y] = 1
                else:
                    table[x][y] = (table[x-1][y] if x else 0) + (
                        table[x][y-1] if y else 0)
        direct = table[width][height]
        formula = (math.comb(width + height, width)
                   - math.comb(bx + by, bx)
                   * math.comb(width - bx + height - by, width - bx))
        if direct != formula:
            raise FreshEvalError("Grid path oracle disagreement")
        cases.append(_case(
            "math", "blocked_grid_paths", i,
            f"On a grid go from (0,0) to ({width},{height}) with only "
            f"unit right/up steps. Never visit ({bx},{by}). "
            "How many distinct valid routes exist? Give one integer.", direct))

        # Mathematics 2: bounded Diophantine count (two independent methods).
        a, b, c = (rng.randint(3, 8) for _ in range(3))
        target = rng.randint(7, 22)
        brute = sum(x + 2*y + 3*z == target
                    for x in range(a+1) for y in range(b+1) for z in range(c+1))
        coefficients = [1] + [0] * target
        for maximum, weight in ((a, 1), (b, 2), (c, 3)):
            nxt = [0] * (target+1)
            for subtotal, n in enumerate(coefficients):
                for v in range(maximum+1):
                    if subtotal + weight*v <= target:
                        nxt[subtotal + weight*v] += n
            coefficients = nxt
        if brute != coefficients[target]:
            raise FreshEvalError("Bounded sum oracle disagreement")
        cases.append(_case(
            "math", "bounded_diophantine", i,
            f"Count integer triples (x,y,z) with 0<=x<={a}, "
            f"0<=y<={b}, 0<=z<={c}, and x+2*y+3*z={target}. "
            "Return the number of triples.", brute))

        # Logic 1: count total orderings satisfying a random DAG.
        edges = sorted(rng.sample(
            [(0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)],
            3))
        order_count = 0
        for order in itertools.permutations(range(5)):
            position = {v: n for n, v in enumerate(order)}
            order_count += all(position[u] < position[v] for u, v in edges)
        dp = [0] * (1 << 5)
        dp[0] = 1
        for mask in range(1 << 5):
            for task in range(5):
                if not mask & (1 << task) and all(
                        (mask & (1 << u)) for u, v in edges if v == task):
                    dp[mask | (1 << task)] += dp[mask]
        if order_count != dp[-1]:
            raise FreshEvalError("Partial order oracle disagreement")
        formatted = ", ".join(f"{chr(65+u)} before {chr(65+v)}" for u, v in edges)
        cases.append(_case(
            "reasoning", "partial_order_permutations", i,
            "There are five distinct tasks A, B, C, D, E. "
            f"The only precedence constraints are: {formatted}. "
            "How many complete orders satisfy all the constraints?",
            order_count))

        # Logic 2: count truth assignments from concrete boolean constraints.
        xor_target = bool(rng.getrandbits(1))
        equality_mode = bool(rng.getrandbits(1))
        def matches(A, B, C, D, E):
            return (bool(A ^ C) == xor_target and (B or D) and
                    (E == (not B if equality_mode else C)) and
                    not (A and E))
        combinations = sum(matches(*values)
                           for values in itertools.product((False, True), repeat=5))
        bitmask_check = 0
        for bits in range(32):
            A, B, C, D, E = (bool(bits & (1 << k)) for k in range(5))
            bitmask_check += bool(matches(A, B, C, D, E))
        if combinations != bitmask_check:
            raise FreshEvalError("Boolean oracle disagreement")
        relation = "E = NOT B" if equality_mode else "E = C"
        xor_rule = "TRUE" if xor_target else "FALSE"
        cases.append(_case(
            "reasoning", "five_variable_logic", i,
            "How many truth assignments of A,B,C,D,E satisfy all "
            f"four constraints? (A XOR C) is {xor_rule}; "
            f"(B OR D) is TRUE; {relation}; (A AND E) is FALSE.",
            combinations))

        # Code trace 1: indexed mutation, independent direct/comprehension gold.
        nums = [rng.randint(-4, 11) for _ in range(5)]
        multiplier = rng.randint(1, 4)
        state = nums[:]
        for k in range(len(state)):
            state[k] = (state[k] + multiplier*k if k % 2 == 0
                        else state[k]*2-k)
        golden = sum(state[::2]) - state[-1]
        alt = [(v+multiplier*k if k % 2 == 0 else v*2-k)
               for k, v in enumerate(nums)]
        if golden != sum(alt[::2])-alt[-1]:
            raise FreshEvalError("Mutation trace oracle disagreement")
        program = (
            f"data = {nums!r}\n"
            "for i in range(len(data)):\n"
            f"    if i % 2 == 0:\n        data[i] += {multiplier} * i\n"
            "    else:\n        data[i] = data[i] * 2 - i\n"
            "print(sum(data[::2]) - data[-1])"
        )
        cases.append(_case(
            "code_trace", "index_mutation", i,
            "What integer does this exact Python program print? "
            "Do not execute code:\n" + program, golden))

        # Code trace 2: moving bounded queue; list and deque agree.
        initial = [rng.randint(1, 7), rng.randint(1, 7)]
        deltas = [rng.randint(-4, 5) for _ in range(5)]
        state = initial[:]
        for delta in deltas:
            state.append(state[-1] + delta)
            if len(state) > 3:
                state.pop(0)
        dq = deque(initial)
        for delta in deltas:
            dq.append(dq[-1] + delta)
            if len(dq) > 3:
                dq.popleft()
        if list(dq) != state:
            raise FreshEvalError("Bounded queue oracle disagreement")
        program = (
            f"queue = {initial!r}\n"
            f"for delta in {deltas!r}:\n"
            "    queue.append(queue[-1] + delta)\n"
            "    if len(queue) > 3:\n"
            "        queue.pop(0)\n"
            "print(sum(queue))"
        )
        cases.append(_case(
            "code_trace", "bounded_queue_mutation", i,
            "What integer does the following Python code print? "
            "Do not execute code:\n" + program, sum(state)))

    if (len(cases) != 36 or len({c["id"] for c in cases}) != 36
            or len({c["prompt"] for c in cases}) != 36):
        raise FreshEvalError("New task coverage incomplete")
    return cases


def summarize_fresh(score: dict) -> dict:
    if (score.get("case_count") != 36
            or len(score.get("by_domain", {})) != 3
            or score.get("family_count") != 6):
        raise FreshEvalError("Incomplete fresh development score")
    domains = score["by_domain"]
    no_regression = all(
        row["high_correct"] >= row["low_correct"] for row in domains.values())
    passed = bool(score["difference_pp"] >= 10.0
                  and score["paired"]["wins"] > score["paired"]["losses"]
                  and score["paired"]["exact_mcnemar_p"] <= 0.05
                  and no_regression)
    return {
        **score,
        "status": "NERAL_LOW_HIGH_FRESH_36_DEVELOPMENT_NOT_RELEASE",
        "fresh_pilot_gate_passed": passed,
        "generation_budgets": {"low": 1024, "high": 2048},
        "private_random_seed_uploaded": False,
        "raw_prompts_or_responses_uploaded": False,
        "training_performed": False,
        "independent_holdout_verified": False,
        "neural_improvement_proven": False,
        "general_reasoning_superiority_proven": False,
        "release_validated": False,
        "limitations": "Fresh local random instances, but public task generators; "
                       "different LOW/HIGH token and computation budgets, "
                       "not an independent blinded holdout or weight comparison.",
    }


def run_fresh(output: Path) -> dict:
    output = Path(output)
    if (output.exists() or output.is_symlink() or not output.parent.is_dir()
            or output.parent.is_symlink() or output.suffix != ".json"):
        raise FreshEvalError("Fresh create-only output JSON required")
    digest = OllamaLocal().inventory().get(_MODEL)
    if not isinstance(digest, str) or _DIGEST.fullmatch(digest) is None:
        raise FreshEvalError("Pinned local Qwen3:8b absent")
    seed = secrets.randbits(64)
    cases = generate_fresh_cases(seed)
    # No private seed or questions are ever printed or written as a file.
    from neral_agent_autonomy import OllamaLocalBackend
    from neral_preview_server import create_preview_app
    backend = OllamaLocalBackend(_MODEL, timeout=90)
    app = create_preview_app(backend)
    client = app.test_client()
    base_url = "http://127.0.0.1:9690"
    if client.get("/", base_url=base_url).status_code != 200:
        raise FreshEvalError("Local isolated web preview inaccessible")
    sess = client.get("/api/neral/preview/session", base_url=base_url)
    if sess.status_code != 200 or not isinstance(sess.json, dict):
        raise FreshEvalError("Local isolated session unavailable")
    headers = {"X-Neral-CSRF": sess.json["csrf"]}
    observations = []
    error_count = {"low": 0, "high": 0}
    for index, case in enumerate(cases):
        prompt = make_prompt(case)
        row = {"id": case["id"]}
        order = ("low", "high") if index % 2 == 0 else ("high", "low")
        for mode in order:
            started = time.monotonic()
            response = client.post(
                "/api/neral/chat/generate" if mode == "low"
                else "/api/neral/chat/think",
                json={"prompt": prompt}, headers=headers, base_url=base_url)
            latency = time.monotonic() - started
            packet = response.get_json(silent=True)
            ok = response.status_code == 200 and isinstance(packet, dict)
            if ok and not (
                    packet.get("model") == _MODEL
                    and packet.get("model_origin") == _ORIGIN
                    and packet.get("adapter_loaded") is False
                    and packet.get("mode") == mode.upper()):
                raise FreshEvalError("Unexpected model provenance or mode")
            answer = packet.get("text") if ok else None
            complete = bool(isinstance(answer, str) and answer.strip())
            # The production backend checks done_reason=stop and refuses
            # incomplete output as an HTTP error. Do not grade partial text.
            if not complete:
                error_count[mode] += 1
            row[mode] = {
                "text": answer if complete else "",
                "complete": complete, "mode": mode.upper(),
                "model": _MODEL, "model_origin": _ORIGIN,
                "adapter_loaded": False,
                "seconds": round(latency, 3),
                "tool_used": packet.get("tool_used") if ok else None,
            }
        observations.append(row)
    if OllamaLocal().inventory().get(_MODEL) != digest:
        raise FreshEvalError("Model digest changed during random holdout")
    scored = score_paired(cases, observations, model_digest=digest)
    report = summarize_fresh(scored)
    report["failed_responses"] = error_count
    # Only scalar metadata leaves the local PC, never private model text.
    with output.open("x", encoding="utf8") as fd:
        fd.write(json.dumps(report, sort_keys=True, allow_nan=False)+"\n")
    print(json.dumps({"status": report["status"],
                      "case_count": report["case_count"],
                      "low_correct": report["low_correct"],
                      "high_correct": report["high_correct"],
                      "fresh_pilot_gate_passed": report["fresh_pilot_gate_passed"]},
                     sort_keys=True))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    run_fresh(args.out)


if __name__ == "__main__":
    main()
