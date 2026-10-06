"""Read-only cross-check of consecutive stage boundary logs against a 44k report.

This verifies internal consistency, NOT event authenticity, actual GPU execution,
checkpoint bytes, dataset provenance, timing instrumentation or model quality.
The expected ledger SHA must come from an independently retained source. Never
use this tool to start training, promote weights, or mutate checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import math
from pathlib import Path
import re

from admission import validate_evidence

HEX64 = re.compile(r"^[0-9a-f]{64}$")
MAX_LEDGER_BYTES = 1_048_576
MAX_STAGES = 256
FIELDS = {
    "schema", "stage_index", "token_start", "token_end", "monotonic_start_ns",
    "monotonic_end_ns", "checkpoint_start_sha256", "checkpoint_end_sha256",
    "input_chain_start_sha256", "input_chain_end_sha256", "rejected_tokens",
    "health_pass", "includes_eval_checkpoint_recovery",
}


def _integer(value: object, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name}: nonnegative integer required")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or HEX64.fullmatch(value) is None:
        raise ValueError(f"{name}: lowercase 64-character SHA-256 required")
    return value


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _bad_constant(value: str):
    raise ValueError(f"non-finite JSON constant: {value}")


def _read_events(ledger: bytes) -> list[dict]:
    if not ledger or len(ledger) > MAX_LEDGER_BYTES:
        raise ValueError("empty or oversized stage ledger")
    try:
        text = ledger.decode("utf-8", errors="strict")
    except UnicodeError as exc:
        raise ValueError("ledger must be UTF-8") from exc
    if not text.endswith("\n") or "\r" in text:
        raise ValueError("ledger requires canonical LF-terminated JSONL")
    lines = text.splitlines()
    if not 3 <= len(lines) <= MAX_STAGES:
        raise ValueError("ledger requires 3..256 complete stages")
    events = []
    for i, line in enumerate(lines):
        if not line.strip():
            raise ValueError(f"stage {i}: blank JSONL row")
        event = json.loads(line, object_pairs_hook=_no_duplicates, parse_constant=_bad_constant)
        if not isinstance(event, dict) or set(event) != FIELDS:
            raise ValueError(f"stage {i}: missing or unexpected ledger fields")
        events.append(event)
    return events


def audit_ledger(record: dict, ledger_bytes: bytes, expected_ledger_sha256: str) -> dict:
    """Cross-check a separately pinned byte-for-byte ledger with stage report.

    This is report-only: even a fully consistent forged ledger cannot prove
    that training occurred. Do not treat `contract_passed` as release approval.
    """
    expected = _sha(expected_ledger_sha256, "external ledger digest")
    digest = hashlib.sha256(ledger_bytes).hexdigest()
    if not hmac.compare_digest(digest, expected):
        raise ValueError("stage ledger differs from external pinned SHA-256")
    if not isinstance(record, dict) or not isinstance(record.get("stages"), list):
        raise ValueError("stage report required")
    events = _read_events(ledger_bytes)
    if len(events) != len(record["stages"]):
        raise ValueError("stage report and ledger lengths differ")
    if not isinstance(record.get("baseline"), dict) or not isinstance(record.get("candidate"), dict):
        raise ValueError("baseline and candidate checkpoint metadata required")
    first_checkpoint = _sha(record["baseline"].get("checkpoint_sha256"), "baseline checkpoint")
    last_checkpoint = _sha(record["candidate"].get("checkpoint_sha256"), "candidate checkpoint")
    previous = None
    seen_checkpoints = {first_checkpoint}
    elapsed_seconds = 0.0
    total_tokens = 0
    for index, (event, claimed) in enumerate(zip(events, record["stages"])):
        if not isinstance(claimed, dict):
            raise ValueError(f"stage {index}: report object required")
        if event["schema"] != 1 or _integer(event["stage_index"], "stage index") != index:
            raise ValueError(f"stage {index}: incorrect schema or stage index")
        start = _integer(event["token_start"], "token start")
        end = _integer(event["token_end"], "token end")
        t0 = _integer(event["monotonic_start_ns"], "monotonic start")
        t1 = _integer(event["monotonic_end_ns"], "monotonic end")
        rejected = _integer(event["rejected_tokens"], "rejected tokens")
        if end <= start or t1 <= t0:
            raise ValueError(f"stage {index}: non-increasing tokens or time")
        before = _sha(event["checkpoint_start_sha256"], "start checkpoint")
        after = _sha(event["checkpoint_end_sha256"], "end checkpoint")
        chain_before = _sha(event["input_chain_start_sha256"], "start input chain")
        chain_after = _sha(event["input_chain_end_sha256"], "end input chain")
        if before == after or chain_before == chain_after or after in seen_checkpoints:
            raise ValueError(f"stage {index}: reused checkpoint or unadvanced input chain")
        if previous is None:
            if before != first_checkpoint:
                raise ValueError("first stage does not start at baseline checkpoint")
        elif (start != previous["token_end"] or t0 != previous["monotonic_end_ns"] or
              before != previous["checkpoint_end_sha256"] or
              chain_before != previous["input_chain_end_sha256"]):
            raise ValueError(f"stage {index}: token/time/checkpoint/input continuity broken")
        if type(event["health_pass"]) is not bool or type(event["includes_eval_checkpoint_recovery"]) is not bool:
            raise ValueError(f"stage {index}: invalid health or accounting flags")
        seconds = (t1 - t0) / 1_000_000_000
        if type(claimed.get("accepted_tokens")) is not int or claimed["accepted_tokens"] != end - start:
            raise ValueError(f"stage {index}: accepted-token report differs from counters")
        wall = claimed.get("wall_seconds")
        if type(wall) not in (float, int) or not math.isfinite(wall) or not math.isclose(wall, seconds, rel_tol=0, abs_tol=1e-6):
            raise ValueError(f"stage {index}: reported wall time excludes time or differs from monotonic bounds")
        for key in ("rejected_tokens", "health_pass", "includes_eval_checkpoint_recovery"):
            if type(claimed.get(key)) is not type(event[key]) or claimed[key] != event[key]:
                raise ValueError(f"stage {index}: {key} report differs from ledger")
        previous = event
        seen_checkpoints.add(after)
        total_tokens += end - start
        elapsed_seconds += seconds
    if previous["checkpoint_end_sha256"] != last_checkpoint:
        raise ValueError("final ledger checkpoint differs from candidate")
    contract = validate_evidence(record)
    if total_tokens != contract["accepted_tokens"] or not math.isclose(
        elapsed_seconds, contract["wall_seconds"], rel_tol=0, abs_tol=len(events) * 1e-6
    ):
        raise ValueError("stage totals differ from quality/speed contract")
    return {
        "schema": 1,
        "classification": "read_only_stage_ledger_consistency_not_proof_of_training",
        "ledger_sha256": digest,
        "stage_count": len(events),
        "accepted_tokens": total_tokens,
        "full_wall_seconds": elapsed_seconds,
        "ledger_consistent": True,
        "contract_passed": contract["admitted"],
        "contract_failures": contract["failures"],
        "eligible_for_production": False,
        "independent_checkpoint_bytes_verified": False,
        "independent_training_verified": False,
        "warning": "Only internal consistency against an externally supplied digest. JSON events and timestamps can be fabricated; independent original logs, actual checkpoint file hashes, real-data provenance, exact resume, full NIS2 and sealed holdout are still required. No automatic promotion.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--expected-ledger-sha256", required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("refusing to overwrite an existing report")
    report = audit_ledger(
        json.loads(args.evidence.read_text(encoding="utf-8")),
        args.ledger.read_bytes(), args.expected_ledger_sha256,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
