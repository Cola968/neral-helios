"""Local-only, fail-closed, deduplicated training examples with audited provenance.

Never treat ordinary model output, benchmark gold labels or HTTP 200 as proof.
The runtime data file is intentionally gitignored; do not commit user prompts.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import threading

DATASET_PATH = Path(__file__).resolve().parent / "data" / "verified_training_dataset.jsonl"
_LOCK = threading.Lock()
_MAX_DATASET_BYTES = 64 * 1024 * 1024
_MAX_TEXT_BYTES = 12_000
_SECRET = re.compile(
    r"(?i)(?:sk-(?:proj-)?[a-z0-9_-]{8,}|"
    r"\b(?:api[_-]?key|access[_-]?token|password|passwort|secret)\s*[:=]\s*\S+|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
# Fail closed for local filesystem identities and common bearer credentials.
# This filter prevents obvious leakage, but it cannot anonymize arbitrary PII.
_LOCAL_PATH = re.compile(
    r"(?i)(?:\b[a-z]:[\\/]|\\\\[a-z0-9_.-]+[\\/]|"
    r"file:///?(?:[a-z]:[\\/]|/)|"
    r"(?<![a-z0-9])/(?:home|users|mnt|root|private|documents|appdata|temp|tmp)/)"
)
_TOKEN = re.compile(
    r"(?i)(?:\bAuthorization\s*:\s*Bearer\s+\S+|"
    r"\bgh[pousr]_[a-z0-9]{16,}|"
    r"\bxox[baprs]-[a-z0-9-]{12,}|"
    r"\bAKIA[0-9A-Z]{16}\b|"
    r"\beyJ[A-Za-z0-9_-]{12,}\.eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,})"
)

_WEB_PROOF = frozenset({
    "literal_property_verified_model_not_accepted",
    "read_page_literal_property_and_model_agreement",
    "literal_property_verified",
})


def verifier_ok(result: object) -> bool:
    """A positive independent verifier is required; cosmetic badges do not count."""
    if (type(result) is not dict or result.get("agent_checked") is not True
            or result.get("tool_verified") is not True
            or result.get("error_code") is not None):
        return False
    if result.get("verified") is True:
        return True
    if result.get("verification_status") not in _WEB_PROOF:
        return False
    sources = result.get("sources")
    return (type(sources) is list and len(sources) >= 2 and
            all(type(s) is dict and s.get("page_read") is True
                and type(s.get("url")) is str
                and s["url"].startswith("https://") for s in sources))


def _key(instruction: str) -> str:
    return hashlib.sha256(" ".join(instruction.split()).encode("utf-8")).hexdigest()


def _safe_text(instruction: object, response: object) -> bool:
    return (type(instruction) is str and type(response) is str
            and instruction.strip() and response.strip()
            and len(instruction.encode("utf-8")) <= _MAX_TEXT_BYTES
            and len(response.encode("utf-8")) <= _MAX_TEXT_BYTES
            and not any(pattern.search(value)
                        for pattern in (_SECRET, _LOCAL_PATH, _TOKEN)
                        for value in (instruction, response)))


def _make_row(instruction: str, response: str, *, source: str,
              verifier: bool, verification_kind: str, **metadata) -> dict:
    return {
        "instruction": instruction.strip(),
        "response": response.strip(),
        "meta": {"instruction_sha256": _key(instruction),
                 "source": source,
                 "verifier_ok": verifier,
                 "verification_kind": verification_kind,
                 **metadata},
    }


def append_verified_example(instruction: str, result: dict,
                            path: Path = DATASET_PATH) -> bool:
    """Add exactly one checked single-turn QA pair; reject secrets and duplicates.

    Returns false for unverified, existing, sensitive, or unusable records.
    Callers should isolate logging failures from the serving request.
    """
    if not verifier_ok(result):
        return False
    response = result.get("text")
    if not _safe_text(instruction, response):
        return False
    row = _make_row(instruction, response, source="live_verified",
                    verifier=True, verification_kind="independent_runtime_proof",
                    tool_used=result.get("tool_used") or "unknown")
    path = Path(path)
    serialized = json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
    with _LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            if path.is_symlink() or not path.is_file():
                return False
            if path.stat().st_size > _MAX_DATASET_BYTES:
                return False
            with path.open("r", encoding="utf-8") as previous:
                for line in previous:
                    if not line.strip():
                        continue
                    item = json.loads(line)
                    if item.get("meta", {}).get("instruction_sha256") == row["meta"]["instruction_sha256"]:
                        return False
        # Same-process lock and one append keep records unduplicated for the
        # single NERAL service; Windows and Unix both support O_APPEND.
        handle = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        try:
            payload = serialized.encode("utf-8")
            if os.write(handle, payload) != len(payload):
                raise OSError("Short JSONL append")
            os.fsync(handle)
        finally:
            os.close(handle)
        return True


def import_benchmark(details: bytes, summary: dict, destination: Path, *,
                     expected_source: str, expected_pass: int = 97) -> int:
    """Build only observed, independently graded passes from a sealed receipt.

    Benchmark-grader approval is deliberately distinct from live verifier_ok.
    This dataset MUST NOT be evaluated against the same benchmark again.
    """
    if (type(summary) is not dict or summary.get("status") != "COMPLETE"
            or summary.get("source_sha") != expected_source
            or summary.get("strict_e2e_pass") != expected_pass
            or hashlib.sha256(details).hexdigest() != summary.get("details_sha256")):
        raise ValueError("Benchmark summary, score, source or digest mismatch")
    source_rows = [json.loads(x) for x in details.decode("utf-8").splitlines() if x.strip()]
    if (len(source_rows) != summary.get("cases") or
            len(source_rows) != summary.get("detail_count")):
        raise ValueError("Incomplete benchmark detail receipt")
    output, keys = [], set()
    for record in source_rows:
        if record.get("grader", {}).get("strict_e2e_pass") is not True:
            continue
        if (record.get("grader", {}).get("answer_correct") is not True
                or record.get("http_status") != 200
                or record.get("observed", {}).get("error_code") is not None):
            raise ValueError("Inconsistent passing benchmark row")
        case, observed = record["case"], record["observed"]
        instruction, response = case["question"], observed["text"]
        if not _safe_text(instruction, response):
            raise ValueError("Invalid benchmark QA text")
        if hashlib.sha256(instruction.encode("utf-8")).hexdigest() != case["question_sha256"]:
            raise ValueError("Question SHA256 mismatch")
        key = _key(instruction)
        if key in keys:
            raise ValueError("Duplicate benchmark instruction")
        keys.add(key)
        output.append(_make_row(
            instruction, response, source="v39_benchmark",
            verifier=False, verification_kind="benchmark_strict_e2e",
            case_id=case["id"], category=case["category"],
            benchmark_source_sha=expected_source,
            tool_verified=observed.get("tool_verified") is True,
            evaluation_excluded=True))
    if len(output) != expected_pass:
        raise ValueError("Expected passed rows missing or duplicated")
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    # Never silently overwrite live user examples. A seed file must be newly
    # created, or exactly equal (idempotent import).
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in output)
    if target.exists():
        if target.read_text(encoding="utf-8") == body:
            return len(output)
        raise FileExistsError("Destination exists with different data")
    target.write_text(body, encoding="utf-8")
    return len(output)
