"""Prepare a pinned GSM8K *test-only* holdout for future Neral evaluation.

This is evaluation data, never training data. The script downloads only the GSM8K main test
Parquet at a pinned Hugging Face revision, verifies its SHA-256, and writes a reduced local
JSONL containing the question and final numeric answer. The worked rationale is deliberately
not copied into the Neral holdout JSONL. No model is called and no weights are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from hf_source_policy50 import SOURCES

DATASET_ID = "openai/gsm8k"
UPSTREAM_REVISION = "e53f048856ff4f594e959d75785d2c2d37b678ee"
FILE_NAME = "main/test-00000-of-00001.parquet"
UPSTREAM_URL = f"https://huggingface.co/datasets/{DATASET_ID}/resolve/{UPSTREAM_REVISION}/{FILE_NAME}"
UPSTREAM_SHA256 = "ee7b8da9e381df27b9e3f7758a159ab2bdaa4dbaa910546cbbc47e0cb44e4f59"
USER_AGENT = "NeralEvalBuilder/1.0 (+test-only public benchmark preparation)"
FINAL_RE = re.compile(r"####\s*([^\r\n]+)\s*$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as response, path.open("wb") as out:
        for chunk in iter(lambda: response.read(1024 * 1024), b""):
            out.write(chunk)


def extract_final_answer(reference: str) -> str:
    if not isinstance(reference, str):
        raise ValueError("GSM8K answer must be text")
    match = FINAL_RE.search(reference.strip())
    if not match:
        raise ValueError("GSM8K answer missing #### final marker")
    value = match.group(1).strip()
    if not value or len(value) > 80:
        raise ValueError("invalid GSM8K final answer")
    return value


def read_parquet(path: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to prepare GSM8K; install pyarrow on the data-prep environment") from exc
    table = pq.read_table(path, columns=["question", "answer"])
    return table.to_pylist()


def prepare(out: Path, keep_raw: bool = False):
    policy = SOURCES[DATASET_ID]
    if policy.allow_training or policy.role != "public-eval-holdout":
        raise RuntimeError("GSM8K policy must remain evaluation-only")
    out.mkdir(parents=True, exist_ok=True)
    raw = out / "gsm8k-test.parquet"
    download(UPSTREAM_URL, raw)
    actual = sha256_file(raw)
    if actual != UPSTREAM_SHA256:
        raise ValueError(f"Pinned GSM8K test SHA-256 mismatch: {actual}")

    upstream = read_parquet(raw)
    rows = []
    seen = set()
    for i, row in enumerate(upstream):
        question = row.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ValueError(f"GSM8K row {i}: missing question")
        question = question.strip()
        digest = hashlib.sha256(question.encode("utf8")).hexdigest()
        if digest in seen:
            raise ValueError(f"GSM8K duplicate test question at row {i}")
        seen.add(digest)
        rows.append({
            "id": f"gsm8k/main/test/{i:04d}",
            "question": question,
            "final_answer": extract_final_answer(row.get("answer")),
            "question_sha256": digest,
        })
    if len(rows) < 1000:
        raise ValueError(f"GSM8K test split unexpectedly small: {len(rows)}")

    jsonl = out / "gsm8k_test_holdout.jsonl"
    jsonl.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")
    manifest = {
        "version": 1,
        "dataset_id": DATASET_ID,
        "role": policy.role,
        "training_allowed": False,
        "license": policy.license,
        "upstream_revision": UPSTREAM_REVISION,
        "upstream_file": FILE_NAME,
        "upstream_sha256": actual,
        "examples": len(rows),
        "prepared_file": jsonl.name,
        "prepared_sha256": sha256_file(jsonl),
        "fields": ["question", "final_answer"],
        "worked_rationale_copied": False,
        "external_model_used": False,
        "weights_modified": False,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf8")
    if not keep_raw:
        raw.unlink()
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="public_eval50_data/gsm8k")
    ap.add_argument("--keep-raw", action="store_true")
    args = ap.parse_args()
    result = prepare(Path(args.out), keep_raw=args.keep_raw)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
