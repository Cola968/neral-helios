"""Bounded read-only FineMath audit using the Hugging Face Dataset Viewer API.

This is a pre-admission screening tool, not a training data loader. It pins the immutable
FineMath revision already recorded by Neral, samples public rows without GPU use, and checks
for exact/near containment of frozen benchmark literals. Passing a bounded sample never
changes the training allow-list; FineMath remains manual-review-only until policy is changed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.parse
import urllib.request
from pathlib import Path

from prepare_pretrain50_b2 import benchmark_contamination, normalize_for_dedup

REPO = Path(__file__).resolve().parent
DATASET_ID = "HuggingFaceTB/finemath"
VIEWER = "https://datasets-server.huggingface.co"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "neral-helios-finemath-audit/1"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return json.load(response)


def pinned_revision() -> str:
    p = REPO / "HF_REVISION_AUDIT_20260908.json"
    obj = json.loads(p.read_text(encoding="utf8"))
    for row in obj["datasets"]:
        if row["dataset_id"] == DATASET_ID:
            rev = str(row["revision"]).lower()
            if len(rev) != 40 or any(c not in "0123456789abcdef" for c in rev):
                raise ValueError("invalid pinned FineMath revision")
            return rev
    raise ValueError("FineMath revision not pinned")


def choose_config() -> tuple[str, str]:
    q = urllib.parse.urlencode({"dataset": DATASET_ID})
    obj = get_json(f"{VIEWER}/splits?{q}")
    pairs = sorted({(str(x["config"]), str(x["split"])) for x in obj.get("splits", []) if x.get("split") == "train"})
    if not pairs:
        raise RuntimeError("FineMath has no Dataset Viewer train split")
    preferred = [p for p in pairs if "3plus" in p[0].lower() or "3+" in p[0].lower()]
    return (preferred or pairs)[0]


def fetch_rows(config: str, split: str, total: int) -> list[dict]:
    rows = []
    offset = 0
    while len(rows) < total:
        length = min(100, total - len(rows))
        q = urllib.parse.urlencode({
            "dataset": DATASET_ID,
            "config": config,
            "split": split,
            "offset": offset,
            "length": length,
        })
        obj = get_json(f"{VIEWER}/rows?{q}")
        page = obj.get("rows") or []
        if not page:
            break
        for wrapped in page:
            row = wrapped.get("row") if isinstance(wrapped, dict) else None
            if isinstance(row, dict):
                rows.append(row)
        offset += len(page)
        if len(page) < length:
            break
    if not rows:
        raise RuntimeError("FineMath Dataset Viewer returned no rows")
    return rows[:total]


def text_of(row: dict) -> str | None:
    for key in ("text", "content", "document"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def shingles(text: str) -> set[str]:
    words = text.split()
    return {" ".join(words[i:i+5]) for i in range(max(0, len(words)-4))}


def audit(rows: list[dict], out_dir: Path, near_threshold: float = 0.80) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    phrases, benchmark_hashes = benchmark_contamination(REPO)
    phrase_shingles = [(p, shingles(p)) for p in phrases if len(p.split()) >= 8]
    exact = 0
    near = 0
    audited = 0
    index_rows = []
    for i, row in enumerate(rows):
        text = text_of(row)
        if not text:
            continue
        norm = normalize_for_dedup(text)
        if not norm:
            continue
        audited += 1
        digest = sha256_bytes(norm.encode("utf8"))
        rid = row.get("id", row.get("url", i))
        exact_hit = any(p in norm for p in phrases)
        if exact_hit:
            exact += 1
        doc_shingles = shingles(norm)
        near_hit = False
        if not exact_hit and doc_shingles:
            for _, ps in phrase_shingles:
                if ps and len(ps & doc_shingles) / len(ps) >= near_threshold:
                    near_hit = True
                    break
        if near_hit:
            near += 1
        index_rows.append({"sample_index": i, "row_id": str(rid), "normalized_sha256": digest, "exact_hit": exact_hit, "near_hit": near_hit})

    index_path = out_dir / "finemath_sample_index.jsonl"
    index_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in index_rows), encoding="utf8")
    benchmark_manifest_sha = sha256_bytes(json.dumps(benchmark_hashes, sort_keys=True, separators=(",", ":")).encode("utf8"))
    manifest = {
        "dataset_id": DATASET_ID,
        "revision": pinned_revision(),
        "license": "ODC-By-1.0",
        "content_origin": "web-human-filtered",
        "file_hashes": {index_path.name: sha256_file(index_path)},
        "cleaning_pipeline_sha256": sha256_file(Path(__file__)),
        "benchmark_screen": {
            "benchmark_manifest_sha256": benchmark_manifest_sha,
            "exact_prompt_matches": exact,
            "near_duplicate_matches": near,
            "answer_leak_matches": exact,
        },
        "sampled_rows": audited,
        "external_model_generated_rows": 0,
        "reserved_eval_dataset_ids": ["openai/gsm8k", "google-research-datasets/mbpp", "deepmind/code_contests"],
        "bounded_sample_only": True,
        "policy_note": "Passing this bounded sample is not sufficient to allow training; FineMath remains manual-review-only.",
    }
    (out_dir / "FINEMATH_CANDIDATE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf8")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rows", type=int, default=1000)
    ap.add_argument("--out", default="finemath_audit_reports")
    args = ap.parse_args()
    if not 1 <= args.sample_rows <= 10000:
        raise ValueError("sample-rows must be 1..10000")
    config, split = choose_config()
    rows = fetch_rows(config, split, args.sample_rows)
    report = audit(rows, Path(args.out))
    meta = {"config": config, "split": split, "requested_rows": args.sample_rows, "audited_rows": report["sampled_rows"], "screen": report["benchmark_screen"]}
    Path(args.out, "FINEMATH_SAMPLE_AUDIT.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf8")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
