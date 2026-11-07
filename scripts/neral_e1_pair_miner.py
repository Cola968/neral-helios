"""NERAL E1: CPU-only mining from genuine *reported* v5 training-task failures.

Source of truth: deterministic v6 TRAIN/VALIDATION candidate, never holdout.
The submitted local NERAL responses must have locked prompt and model SHA.
Only parseable, semantically *incorrect* finals become rejected examples.
Unparseable / correct answers never become fake preference losses.

This is an internal deterministic math oracle, NOT an independent review of
the task, generated answer, or true on-policy provenance. GPU training requires
the separate E1 smoke approval manifest and actual verifier execution.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from fractions import Fraction
from pathlib import Path
import re

from scripts.neral_v6_compositional import FAMILIES, PER_FAMILY, make_case
from scripts.neral_e1_smoke import validate_pairs
from scripts.neral_v6_suite_intake import (
    _read_private_jsonl, audit_pair,
)

SCHEMA = {
    "id", "family", "split", "index", "prompt_sha256", "adapter_sha256",
    "decoding_sha256", "generation_source", "model_generated", "response",
}
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[A-Za-z0-9_-]{1,100}\Z")
FINAL = re.compile(r"(?:FINAL|FINAL_ANSWER)\s*:\s*(.+)", re.IGNORECASE)
EUR = re.compile(r"(\d{1,9})[,.](\d{2})\s*(?:EUR|€)\Z", re.I)
KG = re.compile(r"(\d{1,10}(?:/\d{1,10}|[,.]\d{1,10})?)\s*kg\Z", re.I)
CLOCK = re.compile(r"(\d{2}):(\d{2})\s*\(Tag\s*([+-]\d{1,3})\)\Z", re.I)
UNITS = re.compile(r"(\d{1,12})\s*Stück\Z", re.I)
PERCENT = re.compile(r"(\d{1,10}(?:/\d{1,10}|[,.]\d{1,10})?)\s*%\Z", re.I)


def _final_line(raw: str) -> str | None:
    """One unambiguous terminal answer; reject unsupported free-form text."""
    if (not isinstance(raw, str) or not raw.strip() or
            len(raw) > 8192 or any(ord(ch) < 32 and ch not in "\n\t\r"
                                   for ch in raw)):
        return None
    lines = [x.strip() for x in raw.splitlines() if x.strip()]
    if not lines or len(lines) > 120:
        return None
    marked = [FINAL.fullmatch(line) for line in lines]
    if any(x is not None for x in marked):
        if sum(x is not None for x in marked) != 1 or marked[-1] is None:
            return None
        return marked[-1].group(1).strip()
    return lines[0] if len(lines) == 1 else None


def _fraction(token: str) -> Fraction | None:
    try:
        v = Fraction(token.replace(",", "."))
        if v < 0 or v.numerator.bit_length() > 128 or v.denominator.bit_length() > 128:
            return None
        return v
    except (ValueError, ZeroDivisionError, OverflowError):
        return None


def _answer_value(family: str, raw: str):
    if family not in FAMILIES:
        raise ValueError("Unsupported original training-task family")
    answer = _final_line(raw)
    if answer is None:
        return None
    if family == "money_order":
        m = EUR.fullmatch(answer)
        return int(m.group(1)) * 100 + int(m.group(2)) if m else None
    if family == "two_leg_fuel":
        m = KG.fullmatch(answer)
        return _fraction(m.group(1)) if m else None
    if family == "timezone_stop":
        m = CLOCK.fullmatch(answer)
        if not m or int(m.group(1)) >= 24 or int(m.group(2)) >= 60:
            return None
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    if family == "inventory_flow":
        m = UNITS.fullmatch(answer)
        return int(m.group(1)) if m else None
    m = PERCENT.fullmatch(answer)
    return _fraction(m.group(1)) if m else None


def assess_responses(rows: list[dict], adapter_sha256: str,
                     decoding_sha256: str) -> tuple[list[dict], dict]:
    """Exact matching against train/validation program oracle, no test cases."""
    if (not isinstance(rows, list) or not 1 <= len(rows) <= 700
            or any(not isinstance(sha, str) or not HEX64.fullmatch(sha)
                   for sha in (adapter_sha256, decoding_sha256))):
        raise ValueError("Invalid provenance or response inventory")
    seen = set()
    selected = []
    summary = {"source_responses_checked": 0, "verified_wrong": 0,
               "semantically_correct": 0, "ungradeable": 0}
    for row in rows:
        if not isinstance(row, dict) or set(row) != SCHEMA:
            raise ValueError("Response schema mismatch")
        family, split, index = row["family"], row["split"], row["index"]
        if (family not in FAMILIES or split not in ("train", "validation")
                or type(index) is not int or not 0 <= index < PER_FAMILY[split]):
            raise ValueError("Holdout, unknown family or invalid index")
        expected_id = f"v6-{family}-{split}-{index:03}"
        key = (family, split, index)
        if key in seen or row["id"] != expected_id:
            raise ValueError("Response task identity collision")
        seen.add(key)
        case = make_case(family, split, index)
        prompt = case["messages"][1]["content"]
        chosen = case["messages"][2]["content"]
        if (row["model_generated"] is not True or
                row["generation_source"] != "neral_v5_local" or
                row["adapter_sha256"] != adapter_sha256 or
                row["decoding_sha256"] != decoding_sha256 or
                row["prompt_sha256"] != sha256(prompt.encode("utf8")).hexdigest() or
                not isinstance(row["response"], str) or
                not row["response"].strip() or len(row["response"]) > 8192):
            raise ValueError("Untrusted response provenance or content")
        expected = _answer_value(family, chosen)
        if expected is None:
            raise ValueError("Original deterministic oracle could not be parsed")
        observed = _answer_value(family, row["response"])
        summary["source_responses_checked"] += 1
        if observed is None:
            summary["ungradeable"] += 1
            continue
        if observed == expected:
            summary["semantically_correct"] += 1
            continue
        summary["verified_wrong"] += 1
        selected.append({
            "id": expected_id,
            "domain": "math",
            "family": family,
            "source_id": f"v6-original-{family}-{split}-{index:03}",
            "prompt": prompt,
            "chosen": chosen,
            "rejected": row["response"],
            "chosen_verified": True,
            "rejected_verified": False,
            "verifier_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "verifier_type": "numeric_oracle",
            "split": split,
        })
    selected.sort(key=lambda x: (x["split"], x["family"], x["id"]))
    summary.update({
        "schema": 1,
        "status": "NERAL_E1_ORIGINAL_TRAIN_MATH_PAIRS_CANDIDATE_ONLY",
        "adapter_sha256": adapter_sha256,
        "decoding_sha256": decoding_sha256,
        "generator_sha256": sha256(Path(__file__).with_name("neral_v6_compositional.py").read_bytes()).hexdigest(),
        "verifier_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
        "external_on_policy_proven": False,
        "external_independence_proven": False,
        "training_allowed": False,
        "human_verified": False,
        "training_performed": False,
        "release_validated": False,
        "limitations": (
            "Only synthetic train/validation tasks; generator shares code with "
            "internal oracle. JSON model_generated flag is NOT proof of actual "
            "on-policy generation. Human review and independent verifier replay "
            "needed before any DPO/ORPO smoke."
        ),
    })
    return selected, summary


def _encode(x: dict) -> bytes:
    return (json.dumps(x, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf8")


def build_pairs(responses: Path, dev_suite: Path, release_suite: Path,
                output: Path, *, adapter_sha256: str,
                decoding_sha256: str) -> dict:
    """Write private pair candidate ONLY after completed smoke schema + decon."""
    out = Path(output)
    if (out.is_symlink() or out.exists() or not out.parent.is_dir() or
            out.parent.is_symlink() or
            out.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError("Private output must be a new folder outside repository")
    raw, responses_rows = _read_private_jsonl(responses, 8_000_000)
    selected, status = assess_responses(responses_rows, adapter_sha256, decoding_sha256)
    smoke = validate_pairs(selected)
    _, dev = _read_private_jsonl(dev_suite, 8_000_000)
    _, release = _read_private_jsonl(release_suite, 8_000_000)
    suite = audit_pair(dev, release, [p["prompt"] for p in selected])
    if {p["source_id"] for p in selected} & {
        x["source_id"] for x in (*dev, *release)
    }:
        raise ValueError("Test suite task origin reused in preference training")
    payload = b"".join(_encode(p) for p in selected)
    status.update({
        "source_responses_sha256": sha256(raw).hexdigest(),
        "pairs_sha256": sha256(payload).hexdigest(),
        "train_pairs": smoke["train"],
        "validation_pairs": smoke["validation"],
        "dev_sha256": suite.get("dev_sha256", sha256(Path(dev_suite).read_bytes()).hexdigest()),
        "release_sha256": suite.get("release_sha256", sha256(Path(release_suite).read_bytes()).hexdigest()),
        "release_cases_used_for_training": 0,
        "suite_intake_status": suite["status"],
    })
    out.mkdir(exist_ok=False)
    (out / "pairs.jsonl").write_bytes(payload)
    (out / "manifest.json").write_bytes(_encode(status))
    return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--responses", type=Path, required=True)
    parser.add_argument("--dev-suite", type=Path, required=True)
    parser.add_argument("--release-suite", type=Path, required=True)
    parser.add_argument("--adapter-sha256", required=True)
    parser.add_argument("--decoding-sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    report = build_pairs(
        args.responses, args.dev_suite, args.release_suite, args.out,
        adapter_sha256=args.adapter_sha256,
        decoding_sha256=args.decoding_sha256,
    )
    print(json.dumps({
        "status": report["status"],
        "train_pairs": report["train_pairs"],
        "validation_pairs": report["validation_pairs"],
        "training_allowed": False,
        "human_verified": False,
        "release_validated": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
