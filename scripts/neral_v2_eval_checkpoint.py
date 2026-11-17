"""Crash-safe immutable per-case checkpointing for the paired NERAL DEV24 HF test.

No raw prompts, gold answers, model traces or model weights are written.
A completed case is committed using atomic rename and fsync. Replays refuse
different data, model adapter bytes, token budget, or poisoned case indices.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

PROTOCOL = "NERAL_DEV24_HF_BOUND_256_PAIRED_V1"


def model_sha(adapter_dir: Path) -> str:
    path = adapter_dir / "adapter_model.safetensors"
    if not path.is_file():
        raise FileNotFoundError("Frozen HF adapter missing")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write_once(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = (json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))+"\n").encode("utf8")
    temp = path.with_name(path.name+".pending")
    if path.exists() or path.is_symlink() or temp.exists() or temp.is_symlink():
        raise FileExistsError("Immutable checkpoint already exists")
    with temp.open("xb") as f:
        f.write(body)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def prepare(root: Path, *, label: str, casepack_sha: str, adapter_sha: str,
            max_new_tokens: int, n: int = 24) -> Path:
    if label not in ("v1", "v2full"):
        raise ValueError("Invalid frozen-model label")
    if (type(casepack_sha) is not str or len(casepack_sha) != 64 or
        type(adapter_sha) is not str or len(adapter_sha) != 64 or
        any(x not in "0123456789abcdef" for x in casepack_sha+adapter_sha)):
        raise ValueError("Requires pinned dataset and adapter sha256")
    if max_new_tokens != 256 or n != 24:
        raise ValueError("Only approved bounded DEV24 protocol")
    folder = root / label
    folder.mkdir(parents=True, exist_ok=True)
    meta = {"protocol": PROTOCOL, "model": label,
            "casepack_sha256": casepack_sha, "adapter_sha256": adapter_sha,
            "max_new_tokens": max_new_tokens, "cases": n,
            "human_scored": False, "no_training": True}
    pin = folder / "manifest.json"
    if pin.is_file():
        if json.loads(pin.read_text("utf8")) != meta:
            raise RuntimeError("Frozen checkpoint provenance drift")
    elif pin.exists() or pin.is_symlink():
        raise RuntimeError("Invalid checkpoint manifest")
    else:
        _write_once(pin, meta)
    return folder


def load_completed(folder: Path, cases: list[dict]) -> dict[int, dict]:
    if len(cases) != 24 or len({c["id"] for c in cases}) != 24:
        raise ValueError("Expected 24 unique DEV cases")
    completed = {}
    for p in (folder / "cases").glob("*.json"):
        if p.is_symlink() or not p.stem.isdecimal() or len(p.stem) != 2:
            raise RuntimeError("Invalid per-case checkpoint filename")
        i = int(p.stem)
        if not 0 <= i < 24 or i in completed:
            raise RuntimeError("Wrong checkpoint index")
        record = json.loads(p.read_text("utf8"))
        if set(record) != {"index", "case_id", "question_sha256", "category", "family", "result"}:
            raise RuntimeError("Unrecognized checkpoint record")
        case = cases[i]
        if (record["index"] != i or record["case_id"] != case["id"] or
            record["question_sha256"] != case["question_sha256"] or
            record["category"] != case["category"] or record["family"] != case["family"]):
            raise RuntimeError("Checkpoint no longer matches the original casepack")
        result = record["result"]
        if (type(result) is not dict or result.get("id") != case["id"] or
            result.get("question_sha256") != case["question_sha256"] or
            result.get("category") != case["category"] or
            result.get("family") != case["family"] or
            any(type(result.get(k)) is not bool for k in ("base_strict_correct","neral_strict_correct","independent_verified")) or
            any(type(result.get(k)) not in (int,float) or result[k] < 0
                for k in ("base_seconds","neral_seconds","proposal_calls","fallback_calls","docker_calls"))):
            raise RuntimeError("Invalid recorded per-case metrics")
        completed[i] = result
    return completed


def save_completed(folder: Path, *, index: int, case: dict, result: dict) -> None:
    if not isinstance(index,int) or not 0 <= index < 24:
        raise ValueError("Invalid case index")
    if any(result.get(key) != case.get(key) for key in
           ("id","question_sha256","category","family")):
        raise RuntimeError("Wrong case/result pair")
    if "raw" in result or "gold_integer" in result or "question" in result:
        raise RuntimeError("No raw data or answers in checkpoint")
    payload = {"index":index,"case_id":case["id"],
               "question_sha256":case["question_sha256"],
               "category":case["category"],"family":case["family"],
               "result":result}
    _write_once(folder/"cases"/f"{index:02d}.json",payload)
