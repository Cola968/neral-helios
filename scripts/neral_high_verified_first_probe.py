"""REAL localhost NERAL LOW vs HIGH system-level paired probe.

Uses 24 newly seeded parameter instances in three independently graded
families: modular arithmetic, directed-graph reachability and safe Python
tracing. These are KNOWN tool templates, not independent novel task methods.
No training, downloads, remote AI, executing generated Python or raw uploads.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import re
import time

from neral_think import extract_final
from scripts.neral_v7_foundation_tournament import OllamaLocal

SEED = 20261110
MODEL = "qwen3:8b"
HOST = "http://127.0.0.1:9690"
DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
DOMAINS = ("math", "logic", "python_trace")
TOOLS = frozenset(("neral_math_exact", "neral_logic_exact",
                   "neral_python_trace_static"))


class ScoreError(ValueError):
    """No private model responses, paths or case text in public errors."""


def generate_fresh_system_cases(seed=SEED):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ScoreError("Invalid seed")
    rng = random.Random(seed)
    rows = []
    ending = " Return a single integer as the final result."
    fence = chr(96) * 3
    for i in range(8):
        a = rng.randint(50000, 900000)
        exp = rng.randint(43, 360)
        mod = rng.randint(8001, 99001)
        rows.append({
            "id": f"math-{i:02d}", "domain": "math",
            "prompt": (f"Find the remainder of {a} to the power {exp} modulo "
                       f"{mod}." + ending),
            "answer": pow(a, exp, mod),
        })

        # Deliberately allow cycles and disconnected vertices. The reference
        # BFS is independent of NERAL's trusted_logic_question implementation.
        start = rng.randrange(14)
        edge_set = set()
        for _ in range(25):
            edge_set.add((rng.randrange(14), rng.randrange(14)))
        edges = sorted(edge_set)
        adjacency = {v: [] for v in range(14)}
        for u, v in edges:
            adjacency[u].append(v)
        visited = {start}
        frontier = [start]
        while frontier:
            u = frontier.pop()
            for v in adjacency[u]:
                if v not in visited:
                    visited.add(v)
                    frontier.append(v)
        rows.append({
            "id": f"logic-{i:02d}", "domain": "logic",
            "prompt": (f"A directed graph contains exactly these edges {edges}. "
                       f"How many other vertices can be reached from vertex {start}?"
                       + ending),
            "answer": len(visited) - 1,
        })

        length = rng.randint(23, 54)
        divisor = rng.randint(3, 9)
        code = (
            "total = 0\n"
            f"for value in range(1, {length}):\n"
            f"    if value % {divisor} == 0:\n"
            "        total += value * (value - 1)\n"
            "print(total)"
        )
        rows.append({
            "id": f"python-{i:02d}", "domain": "python_trace",
            "prompt": ("Which integer does this Python program print?\n"
                       + fence + "python\n" + code + "\n" + fence + ending),
            "answer": sum(v * (v - 1) for v in range(1, length)
                          if v % divisor == 0),
        })
    if (len(rows) != 24 or len({r["id"] for r in rows}) != 24 or
            any(type(r["answer"]) is not int for r in rows)):
        raise ScoreError("Invalid generated system suite")
    return rows


def _correct(text, answer, complete):
    if not complete or not isinstance(text, str) or len(text) > 32768:
        return False
    final = extract_final(text)
    if not isinstance(final, str) or not re.fullmatch(r"[+-]?\d{1,12}", final):
        return False
    return int(final) == answer


def score_pairs(cases, pairs, *, model_digest):
    if (not isinstance(model_digest, str) or DIGEST.fullmatch(model_digest) is None
            or not isinstance(cases, list) or len(cases) != 24
            or not isinstance(pairs, list) or len(pairs) != len(cases)):
        raise ScoreError("Invalid benchmark data or provenance")
    cases_by_id = {row["id"]: row for row in cases}
    if len(cases_by_id) != 24:
        raise ScoreError("Duplicate reference cases")
    results = {}
    for row in pairs:
        if (not isinstance(row, dict) or set(row) != {"id", "low", "high"}
                or row["id"] not in cases_by_id or row["id"] in results):
            raise ScoreError("Unpaired or duplicate result")
        for mode in ("low", "high"):
            part = row[mode]
            expected = {"text", "complete", "seconds"} if mode == "low" else {
                "text", "complete", "seconds", "verified",
                "model_generated", "tool_used",
            }
            if (not isinstance(part, dict) or set(part) != expected or
                    type(part["complete"]) is not bool or
                    not isinstance(part["text"], str) or
                    len(part["text"]) > 32768 or
                    type(part["seconds"]) not in (int, float) or
                    not math.isfinite(part["seconds"]) or
                    part["seconds"] < 0):
                raise ScoreError("Invalid result data")
        h = row["high"]
        if (h["verified"] is not True or h["model_generated"] is not False
                or h["tool_used"] not in TOOLS):
            raise ScoreError("HIGH did not use a verified exact tool")
        results[row["id"]] = row
    if len(results) != len(cases):
        raise ScoreError("Missing results")

    per_domain = {domain: {"n": 0, "low_correct": 0, "high_correct": 0}
                  for domain in DOMAINS}
    low = high = wins = losses = 0
    seconds = {"low": 0.0, "high": 0.0}
    for case in cases:
        pair = results[case["id"]]
        l = _correct(pair["low"]["text"], case["answer"],
                     pair["low"]["complete"])
        h = _correct(pair["high"]["text"], case["answer"],
                     pair["high"]["complete"])
        domain = per_domain[case["domain"]]
        domain["n"] += 1
        domain["low_correct"] += int(l)
        domain["high_correct"] += int(h)
        low += int(l)
        high += int(h)
        wins += int(h and not l)
        losses += int(l and not h)
        seconds["low"] += pair["low"]["seconds"]
        seconds["high"] += pair["high"]["seconds"]
    n_discordant = wins + losses
    p_value = 1.0 if n_discordant == 0 else min(
        1.0, 2 * sum(math.comb(n_discordant, k)
                    for k in range(min(wins, losses) + 1)) / (2 ** n_discordant))
    pp = 100 * (high - low) / 24
    good = bool(pp >= 20 and wins > losses and p_value <= .05 and
                all(d["high_correct"] >= d["low_correct"]
                    for d in per_domain.values()))
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")).encode("utf8")
    return {
        "schema": 1, "status": "NERAL_VERIFIED_FIRST_SYSTEM_DEV_ONLY",
        "model": MODEL, "model_digest": model_digest,
        "cases": 24, "seed": SEED, "case_sha256": sha256(canonical).hexdigest(),
        "domains": per_domain,
        "low_correct": low, "high_correct": high,
        "system_gain_pp": round(pp, 2),
        "paired_wins": wins, "paired_losses": losses,
        "exact_mcnemar_p": round(p_value, 8),
        "total_seconds": {k: round(v, 3) for k, v in seconds.items()},
        "development_gate_passed": good,
        "model_weights_changed": False,
        "neural_improvement_proven": False,
        "release_validated": False,
        "independent_novel_methods": False,
        "tools_granted_only_to_high": True,
        "no_prompt_or_response_upload": True,
        "limitations": ("System gain on three known exact-tool grammar families "
                        "with newly generated operands; Qwen with identical "
                        "tools may reach equal results; not general intelligence."),
    }


def run_local_probe(output: Path):
    output = Path(output)
    if (output.exists() or output.is_symlink() or
            not output.parent.is_dir() or output.parent.is_symlink()
            or output.suffix != ".json"):
        raise ScoreError("Create-only private output JSON required")
    installed = OllamaLocal().inventory()
    digest = installed.get(MODEL)
    if not isinstance(digest, str) or DIGEST.fullmatch(digest) is None:
        raise ScoreError("Pinned Qwen3-8B not installed")
    from neral_agent_autonomy import OllamaLocalBackend
    from neral_preview_server import create_preview_app

    app = create_preview_app(OllamaLocalBackend(MODEL, timeout=90))
    client = app.test_client()
    client.get("/", base_url=HOST)
    session = client.get("/api/neral/preview/session", base_url=HOST)
    if session.status_code != 200:
        raise ScoreError("Local preview CSRF session unavailable")
    headers = {"X-Neral-CSRF": session.json["csrf"]}
    rows = []
    for i, case in enumerate(generate_fresh_system_cases()):
        pair = {"id": case["id"]}
        for mode in (("low", "high") if i % 2 == 0 else ("high", "low")):
            start = time.monotonic()
            route = "/api/neral/chat/generate" if mode == "low" else "/api/neral/chat/think"
            response = client.post(route, json={"prompt": case["prompt"]},
                                   base_url=HOST, headers=headers)
            seconds = round(time.monotonic() - start, 3)
            packet = response.json if response.status_code == 200 else {}
            if response.status_code == 200 and (
                    packet.get("model") != MODEL or
                    packet.get("model_origin") != "local_foundation_not_neral_adapter" or
                    packet.get("adapter_loaded") is not False):
                raise ScoreError("Unexpected model origin or adapter state")
            value = packet.get("text", "") if isinstance(packet, dict) else ""
            complete = response.status_code == 200 and isinstance(value, str)
            part = {
                "text": value if complete else "", "complete": complete,
                "seconds": seconds,
            }
            if mode == "high":
                part.update({
                    "verified": packet.get("verified") if complete else False,
                    "model_generated": packet.get("model_generated") if complete else None,
                    "tool_used": packet.get("tool_used") if complete else None,
                })
            pair[mode] = part
        rows.append(pair)
    if OllamaLocal().inventory().get(MODEL) != digest:
        raise ScoreError("Local Qwen model changed during test")
    report = score_pairs(generate_fresh_system_cases(), rows, model_digest=digest)
    with output.open("x", encoding="utf8") as stream:
        stream.write(json.dumps(report, allow_nan=False, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"],
                      "low_correct": report["low_correct"],
                      "high_correct": report["high_correct"],
                      "development_gate_passed": report["development_gate_passed"]},
                     sort_keys=True))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    run_local_probe(args.out)


if __name__ == "__main__":
    main()
