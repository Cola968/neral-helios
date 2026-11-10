"""Local raw-model A/B generation-health audit. Never trains or promotes weights.

Inputs can contain private answers. Stdout/output expose only aggregates and hashes.
Input provenance and raw checkpoint identity must be verified independently.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

HEX = re.compile(r"[0-9a-f]{64}\Z")
ROLE = re.compile(r"(?im)^\s*(?:user|nutzer|assistant|system)\s*:")
REQUIRED = {"id", "prompt_sha256", "answer", "finish_reason", "raw_model", "tool_calls", "tokens_generated", "max_tokens"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def load(path: Path):
    if path.stat().st_size > 10_000_000:
        raise ValueError("report exceeds local audit size limit")
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON")))
    if not isinstance(data, dict) or set(data) != {"schema", "checkpoint_sha256", "protocol_sha256", "cases"} or data["schema"] != 1:
        raise ValueError("invalid raw generation report schema")
    for name in ("checkpoint_sha256", "protocol_sha256"):
        if not isinstance(data[name], str) or not HEX.fullmatch(data[name]):
            raise ValueError(f"invalid {name}")
    cases = data["cases"]
    if not isinstance(cases, list) or not 8 <= len(cases) <= 512:
        raise ValueError("require 8..512 real raw-generation cases")
    ids, prompts = set(), set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != REQUIRED:
            raise ValueError("invalid case schema")
        if not isinstance(case["id"], str) or not 1 <= len(case["id"]) <= 128 or case["id"] in ids:
            raise ValueError("invalid or duplicate case ID")
        ids.add(case["id"])
        digest = case["prompt_sha256"]
        if not isinstance(digest, str) or not HEX.fullmatch(digest):
            raise ValueError("invalid prompt hash")
        if digest in prompts:
            raise ValueError("repeated evaluation prompt hash inflates test population")
        prompts.add(digest)
        if not isinstance(case["answer"], str) or len(case["answer"]) > 10000:
            raise ValueError("invalid answer")
        if case["raw_model"] is not True or type(case["tool_calls"]) is not int or case["tool_calls"] != 0:
            raise ValueError("assisted/tool-generated output is not raw model evidence")
        if case["finish_reason"] not in ("eos", "max_tokens"):
            raise ValueError("an interrupted or artificial-stop response is not a complete raw run")
        generated, budget = case["tokens_generated"], case["max_tokens"]
        if type(generated) is not int or type(budget) is not int or not 1 <= generated <= budget <= 8192:
            raise ValueError("invalid generation budget or counter")
        if case["finish_reason"] == "max_tokens" and generated != budget:
            raise ValueError("claimed budget exhaustion without generating the full budget")
        # EOS as first token is real, but an empty unusable answer: nonempty_rate penalizes it.
    return data


def metrics(data):
    cases = data["cases"]
    counts = {"eos": 0, "nonempty": 0, "role_bleed": 0, "repeated_trigrams": 0, "trigrams": 0}
    for case in cases:
        answer = case["answer"]
        counts["eos"] += case["finish_reason"] == "eos"
        counts["nonempty"] += bool(answer.strip())
        counts["role_bleed"] += bool(ROLE.search(answer))
        words = re.findall(r"\w+", answer.casefold())
        grams = [tuple(words[i:i + 3]) for i in range(max(0, len(words) - 2))]
        counts["trigrams"] += len(grams)
        counts["repeated_trigrams"] += len(grams) - len(set(grams))
    total = len(cases)
    return {"cases": total, "eos_rate": counts["eos"] / total,
            "nonempty_rate": counts["nonempty"] / total,
            "role_bleed_rate": counts["role_bleed"] / total,
            "repetition_rate": counts["repeated_trigrams"] / max(1, counts["trigrams"])}


def compare(before, after):
    if before["protocol_sha256"] != after["protocol_sha256"]:
        raise ValueError("decoding/evaluation protocol changed")
    if before["checkpoint_sha256"] == after["checkpoint_sha256"]:
        raise ValueError("same checkpoint cannot show weight improvement")
    b = {c["id"]: (c["prompt_sha256"], c["max_tokens"]) for c in before["cases"]}
    a = {c["id"]: (c["prompt_sha256"], c["max_tokens"]) for c in after["cases"]}
    if b != a:
        raise ValueError("different evaluation prompts or token budgets")
    if len({case["prompt_sha256"] for case in before["cases"]}) != len(before["cases"]):
        raise ValueError("baseline repeats prompt hashes")
    if len({case["prompt_sha256"] for case in after["cases"]}) != len(after["cases"]):
        raise ValueError("candidate repeats prompt hashes")
    old, new = metrics(before), metrics(after)
    passed = (new["eos_rate"] >= old["eos_rate"] and
              new["nonempty_rate"] >= old["nonempty_rate"] and
              new["role_bleed_rate"] <= old["role_bleed_rate"] and
              new["repetition_rate"] <= old["repetition_rate"] + 0.01)
    return {"schema": 1, "classification": "generation_health_only_not_release",
            "baseline_sha256": before["checkpoint_sha256"],
            "candidate_sha256": after["checkpoint_sha256"],
            "protocol_sha256": before["protocol_sha256"], "baseline": old,
            "candidate": new, "health_nonregression": passed,
            "answer_correctness_evaluated": False, "nis2_evaluated": False,
            "production_approved": False,
            "note": "Health-only diagnostic. Independent blind correctness, full NIS2, safety, latency and provenance remain required."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = compare(load(args.baseline), load(args.candidate))
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        with args.out.open("x", encoding="utf-8") as file:
            file.write(text + "\n")
    print(text)
    return 0 if report["health_nonregression"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
