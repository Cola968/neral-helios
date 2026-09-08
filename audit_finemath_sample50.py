"""Bounded read-only FineMath audit using the Hugging Face Dataset Viewer API.

This is a pre-admission screening tool, not a training data loader. It pins the immutable
FineMath revision already recorded by Neral and stores only hashes/metadata, never raw text.
Passing any sample never changes the training allow-list.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

from hf_source_policy50 import SOURCES
from prepare_pretrain50_b2 import benchmark_contamination, normalize_for_dedup

REPO = Path(__file__).resolve().parent
DATASET_ID = "HuggingFaceTB/finemath"
VIEWER = "https://datasets-server.huggingface.co"
HUB_API = "https://huggingface.co/api/datasets/HuggingFaceTB/finemath"
RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_json(url: str, timeout: int = 45, max_attempts: int = 8) -> dict:
    """GET JSON with bounded exponential backoff for Hub/Dataset Viewer throttling."""
    last_error: Exception | None = None
    for attempt in range(max_attempts):
        req = urllib.request.Request(url, headers={"User-Agent": "neral-helios-finemath-audit/3"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in RETRYABLE_HTTP or attempt + 1 >= max_attempts:
                raise
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            try:
                delay = float(retry_after) if retry_after else min(30.0, 1.5 * (2**attempt))
            except ValueError:
                delay = min(30.0, 1.5 * (2**attempt))
            print(f"HF HTTP {exc.code}; retrying in {delay:.1f}s ({attempt + 1}/{max_attempts})", flush=True)
            time.sleep(delay)
        except (TimeoutError, urllib.error.URLError) as exc:
            last_error = exc
            if attempt + 1 >= max_attempts:
                raise
            delay = min(20.0, 1.0 * (2**attempt))
            print(f"HF transient network error; retrying in {delay:.1f}s ({attempt + 1}/{max_attempts})", flush=True)
            time.sleep(delay)
    raise RuntimeError("HF request retry loop exhausted") from last_error


def pinned_revision() -> str:
    obj = json.loads((REPO / "HF_REVISION_AUDIT_20260908.json").read_text(encoding="utf8"))
    for row in obj["datasets"]:
        if row["dataset_id"] == DATASET_ID:
            revision = str(row["revision"]).lower()
            if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
                raise ValueError("invalid pinned FineMath revision")
            return revision
    raise ValueError("FineMath revision not pinned")


def assert_live_revision_pinned() -> str:
    pinned = pinned_revision()
    live = str(get_json(HUB_API).get("sha", "")).lower()
    if live != pinned:
        raise RuntimeError(f"FineMath revision drift: pinned={pinned} live={live}")
    return pinned


def choose_config() -> tuple[str, str]:
    q = urllib.parse.urlencode({"dataset": DATASET_ID})
    obj = get_json(f"{VIEWER}/splits?{q}")
    pairs = sorted({
        (str(x["config"]), str(x["split"]))
        for x in obj.get("splits", [])
        if x.get("split") == "train"
    })
    if not pairs:
        raise RuntimeError("FineMath has no train split")
    preferred = [p for p in pairs if "3plus" in p[0].lower() or "3+" in p[0].lower()]
    return (preferred or pairs)[0]


def fetch_page(config: str, split: str, offset: int, length: int) -> dict:
    q = urllib.parse.urlencode({
        "dataset": DATASET_ID,
        "config": config,
        "split": split,
        "offset": offset,
        "length": length,
    })
    return get_json(f"{VIEWER}/rows?{q}")


def total_rows(config: str, split: str) -> int:
    obj = fetch_page(config, split, 0, 1)
    total = obj.get("num_rows_total") or obj.get("num_rows")
    if not total:
        raise RuntimeError("Dataset Viewer did not expose total row count")
    return int(total)


def fetch_rows(
    config: str,
    split: str,
    total: int,
    sampling: str = "sequential",
    request_interval: float = 0.45,
) -> tuple[list[dict], dict]:
    """Fetch at most 100 rows/request and deliberately pace Dataset Viewer requests."""
    n = total_rows(config, split)
    rows: list[dict] = []
    if sampling == "sequential":
        offsets = list(range(0, min(total, n), 100))
    elif sampling == "stratified-v1":
        windows = max(1, (total + 99) // 100)
        maxoff = max(0, n - 100)
        offsets = sorted({round(i * maxoff / max(1, windows - 1)) for i in range(windows)})
    else:
        raise ValueError("unknown sampling mode")

    used_offsets: list[int] = []
    for page_index, off in enumerate(offsets):
        need = min(100, total - len(rows))
        if need <= 0:
            break
        if page_index:
            time.sleep(max(0.0, request_interval))
        page = fetch_page(config, split, off, need)
        used_offsets.append(off)
        for wrapped in page.get("rows") or []:
            row = wrapped.get("row") if isinstance(wrapped, dict) else None
            if isinstance(row, dict):
                rows.append(row)
        if len(rows) >= total:
            break

    if not rows:
        raise RuntimeError("FineMath Dataset Viewer returned no rows")
    return rows[:total], {
        "dataset_rows_total": n,
        "window_offsets": used_offsets,
        "sampling": sampling,
        "request_interval_seconds": request_interval,
        "requests_used": len(used_offsets) + 1,
    }


def text_of(row: dict) -> str | None:
    for key in ("text", "content", "document"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return None


def shingles(text: str) -> set[str]:
    words = text.split()
    return {" ".join(words[i:i + 5]) for i in range(max(0, len(words) - 4))}


def audit(rows: list[dict], out_dir: Path, near_threshold: float = .80, sampling_meta: dict | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    phrases, benchmark_hashes = benchmark_contamination(REPO)
    phrase_shingles = [shingles(p) for p in phrases if len(p.split()) >= 8]
    exact = near = audited = 0
    index: list[dict] = []
    for i, row in enumerate(rows):
        text = text_of(row)
        if not text:
            continue
        norm = normalize_for_dedup(text)
        if not norm:
            continue
        audited += 1
        digest = sha256_bytes(norm.encode())
        rid = row.get("id", row.get("url", i))
        exact_hit = any(p in norm for p in phrases)
        exact += int(exact_hit)
        doc = shingles(norm)
        near_hit = False
        if not exact_hit and doc:
            near_hit = any(ps and len(ps & doc) / len(ps) >= near_threshold for ps in phrase_shingles)
        near += int(near_hit)
        index.append({
            "sample_index": i,
            "row_id": str(rid),
            "normalized_sha256": digest,
            "exact_hit": exact_hit,
            "near_hit": near_hit,
        })

    index_path = out_dir / "finemath_sample_index.jsonl"
    index_path.write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in index), encoding="utf8")
    benchmark_sha = sha256_bytes(json.dumps(benchmark_hashes, sort_keys=True, separators=(",", ":")).encode())
    reserved_eval_ids = sorted(k for k, v in SOURCES.items() if v.role == "public-eval-holdout")
    manifest = {
        "dataset_id": DATASET_ID,
        "revision": assert_live_revision_pinned(),
        "license": "ODC-By-1.0",
        "content_origin": "web-human-filtered",
        "file_hashes": {index_path.name: sha256_file(index_path)},
        "cleaning_pipeline_sha256": sha256_file(Path(__file__)),
        "benchmark_screen": {
            "benchmark_manifest_sha256": benchmark_sha,
            "exact_prompt_matches": exact,
            "near_duplicate_matches": near,
            "answer_leak_matches": exact,
        },
        "sampled_rows": audited,
        "external_model_generated_rows": 0,
        "reserved_eval_dataset_ids": reserved_eval_ids,
        "bounded_sample_only": True,
        "sampling": sampling_meta or {},
        "policy_note": "Passing this sample is not sufficient to allow training; FineMath remains manual-review-only.",
    }
    (out_dir / "FINEMATH_CANDIDATE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf8",
    )
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample-rows", type=int, default=1000)
    ap.add_argument("--sampling", choices=("sequential", "stratified-v1"), default="sequential")
    ap.add_argument("--request-interval", type=float, default=.45)
    ap.add_argument("--out", default="finemath_audit_reports")
    args = ap.parse_args()
    if not 1 <= args.sample_rows <= 25000:
        raise ValueError("sample-rows must be 1..25000")
    if not 0.0 <= args.request_interval <= 10.0:
        raise ValueError("request-interval must be 0..10 seconds")
    assert_live_revision_pinned()
    config, split = choose_config()
    rows, meta = fetch_rows(config, split, args.sample_rows, args.sampling, args.request_interval)
    report = audit(rows, Path(args.out), sampling_meta=meta)
    summary = {
        "config": config,
        "split": split,
        "requested_rows": args.sample_rows,
        "audited_rows": report["sampled_rows"],
        "sampling": meta,
        "screen": report["benchmark_screen"],
    }
    Path(args.out, "FINEMATH_SAMPLE_AUDIT.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
