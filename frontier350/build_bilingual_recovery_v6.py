from __future__ import annotations

"""ARCTUS V6 bilingual-preservation corpus builder.

Builds 10M-token development-only continuation corpora from the immutable 20M
recovery seed plus a bounded amount of human DE<->EN Tatoeba bitext. The local
source ratios follow V5 ``math_language`` and are scaled down uniformly as the
bilingual share increases. A deterministic Tatoeba holdout is excluded from all
training variants and written separately for independent evaluation.

This is a development experiment, not a release corpus.
"""

import argparse
import ast
import hashlib
import io
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np

from build_frontier_bpe_data import LocalBPE, StreamingShardWriter, sha256
from build_weakness_recovery_pilot_bpe_data import PROFILES, START_TOKENS
from remix_recovery_corpus import slice_stream, source_ranges
from security_policy350_bpe import validate_manifest

TATOEBA_REPO = "Helsinki-NLP/tatoeba"
TATOEBA_LICENSE = "cc-by-2.0"
TATOEBA_DATE = "v2021-07-22"
TATOEBA_URL = (
    "https://object.pouta.csc.fi/OPUS-Tatoeba/"
    f"{TATOEBA_DATE}/moses/de-en.txt.zip"
)
TRAIN_TOKENS = 10_000_000
HOLDOUT_PAIRS = 200
MATH_LANGUAGE = PROFILES["math_language"]
V6_SHARES = {
    "xling_05": 0.05,
    "xling_10": 0.10,
    "xling_15": 0.15,
}

_WS = re.compile(r"\s+")
_TRIM = re.compile(r"^[\s'\"“”‘’.,!?;:()\[\]{}]+|[\s'\"“”‘’.,!?;:()\[\]{}]+$")


def clean(text: str) -> str:
    return _WS.sub(" ", text.replace("\x00", " ")).strip()


def key(text: str) -> str:
    return _TRIM.sub("", clean(text).lower())


def pair_hash(de: str, en: str) -> str:
    return hashlib.sha256((de + "\0" + en).encode("utf-8")).hexdigest()


def recovery_bilingual_cases() -> list[tuple[str, str, list[str]]]:
    """Read the dev-suite bilingual literals without importing torch-dependent eval code."""
    eval_path = Path(__file__).with_name("eval_recovery_dev.py")
    tree = ast.parse(eval_path.read_text(encoding="utf8"), filename=str(eval_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "bilingual" for target in node.targets):
            continue
        value = ast.literal_eval(node.value)
        if not isinstance(value, list) or len(value) != 20:
            raise RuntimeError("unexpected bilingual recovery-dev case count")
        return value
    raise RuntimeError("bilingual recovery-dev cases not found")


def benchmark_blacklist() -> set[str]:
    blocked: set[str] = set()
    for prefix, preferred, distractors in recovery_bilingual_cases():
        for value in [preferred, *distractors]:
            value_key = key(value)
            if value_key:
                blocked.add(value_key)
        for quoted in re.findall(r"'([^']+)'", prefix):
            value_key = key(quoted)
            if value_key:
                blocked.add(value_key)
    return blocked


def read_pairs(zip_path: Path) -> tuple[list[tuple[str, str]], str]:
    digest = sha256(zip_path)
    blocked = benchmark_blacklist()
    seen: set[str] = set()
    rows: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        de_name = next((n for n in names if n.endswith(".de") and not n.endswith(".xml.de")), None)
        en_name = next((n for n in names if n.endswith(".en") and not n.endswith(".xml.en")), None)
        if not de_name or not en_name:
            raise RuntimeError(f"DE/EN files not found in Tatoeba zip: {names[:20]}")
        with io.TextIOWrapper(zf.open(de_name), encoding="utf-8") as de_f, io.TextIOWrapper(
            zf.open(en_name), encoding="utf-8"
        ) as en_f:
            for de_raw, en_raw in zip(de_f, en_f):
                de, en = clean(de_raw), clean(en_raw)
                if not de or not en or len(de) > 240 or len(en) > 240:
                    continue
                if len(de.split()) > 24 or len(en.split()) > 24:
                    continue
                if key(de) in blocked or key(en) in blocked:
                    continue
                digest_pair = pair_hash(de, en)
                if digest_pair in seen:
                    continue
                seen.add(digest_pair)
                rows.append((de, en))
    rows.sort(key=lambda x: pair_hash(*x))
    if len(rows) < HOLDOUT_PAIRS + 5_000:
        raise RuntimeError(f"not enough admitted Tatoeba pairs: {len(rows)}")
    return rows, digest


def make_holdout(pairs: list[tuple[str, str]], zip_sha: str) -> dict:
    chosen = pairs[:HOLDOUT_PAIRS]
    cases = []
    n = len(chosen)
    offsets = (37, 83, 149)
    for i, (de, en) in enumerate(chosen):
        de_distractors = [chosen[(i + off) % n][0] for off in offsets]
        en_distractors = [chosen[(i + off) % n][1] for off in offsets]
        cases.append(
            {
                "id": f"en-de-{i+1:03d}",
                "direction": "en_to_de",
                "prefix": f'English: "{en}"\nGerman:',
                "preferred": " " + de,
                "distractors": [" " + x for x in de_distractors],
            }
        )
        cases.append(
            {
                "id": f"de-en-{i+1:03d}",
                "direction": "de_to_en",
                "prefix": f'Deutsch: "{de}"\nEnglish:',
                "preferred": " " + en,
                "distractors": [" " + x for x in en_distractors],
            }
        )
    return {
        "schema": 1,
        "suite": "arctus-bilingual-holdout-v1",
        "source": {
            "repo": TATOEBA_REPO,
            "license": TATOEBA_LICENSE,
            "snapshot": TATOEBA_DATE,
            "url": TATOEBA_URL,
            "archive_sha256": zip_sha,
        },
        "pairs": HOLDOUT_PAIRS,
        "cases": cases,
        "training_overlap": False,
        "purpose": "development-only independent bilingual recovery check; never training data",
    }


def local_quotas(total_local: int) -> dict[str, int]:
    names = list(MATH_LANGUAGE)
    raw = [int(total_local * MATH_LANGUAGE[n]) for n in names[:-1]]
    quotas = {name: count for name, count in zip(names[:-1], raw)}
    quotas[names[-1]] = total_local - sum(raw)
    if sum(quotas.values()) != total_local:
        raise AssertionError("local quota accounting")
    return quotas


def add_pair_tokens(
    writer: StreamingShardWriter,
    tokenizer: LocalBPE,
    training_pairs: list[tuple[str, str]],
    target_tokens: int,
) -> tuple[int, int]:
    emitted = 0
    pair_docs = 0
    for de, en in training_pairs:
        docs = (
            f"English: {en}\nGerman: {de}",
            f"German: {de}\nEnglish: {en}",
        )
        for text in docs:
            ids = tokenizer.encode_document(text)
            remaining = target_tokens - emitted
            if remaining <= 0:
                return emitted, pair_docs
            if len(ids) > remaining:
                ids = ids[:remaining]
            writer.add(ids)
            emitted += len(ids)
            pair_docs += 1
            if emitted >= target_tokens:
                return emitted, pair_docs
    raise RuntimeError(
        f"Tatoeba admitted stream exhausted at {emitted:,}/{target_tokens:,} bilingual tokens"
    )


def verify_files(root: Path, subdir: str, items: list[dict]) -> int:
    total = 0
    for item in items:
        path = root / subdir / item["path"]
        if not path.is_file():
            raise RuntimeError(f"missing shard: {path}")
        if path.stat().st_size != int(item["tokens"]) * 2:
            raise RuntimeError(f"bad shard size: {path}")
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"bad shard hash: {path}")
        total += int(item["tokens"])
    return total


def build(seed: Path, root: Path, tokenizer_path: Path, zip_path: Path, profile: str) -> dict:
    if profile not in V6_SHARES:
        raise RuntimeError(f"unknown V6 profile: {profile}")
    seed, root, tokenizer_path, zip_path = map(Path, (seed, root, tokenizer_path, zip_path))
    validate_manifest(seed / "manifest.json", seed, tokenizer_path)
    seed_manifest = json.loads((seed / "manifest.json").read_text(encoding="utf8"))
    if (
        seed_manifest.get("continuation_profile") != "arctus-2b-balanced-pilot-v1"
        or int(seed_manifest.get("train_tokens", 0)) != 20_000_000
        or int(seed_manifest.get("validation_tokens", 0)) != 200_000
    ):
        raise RuntimeError("requires immutable admitted 20M recovery seed")
    if root.exists() and any(root.iterdir()):
        raise RuntimeError(f"refusing non-empty output root: {root}")
    root.mkdir(parents=True, exist_ok=True)

    tokenizer = LocalBPE(tokenizer_path)
    pairs, zip_sha = read_pairs(zip_path)
    holdout = make_holdout(pairs, zip_sha)
    training_pairs = pairs[HOLDOUT_PAIRS:]
    holdout_path = root / "bilingual_holdout.json"
    holdout_path.write_text(json.dumps(holdout, indent=2, ensure_ascii=False), encoding="utf8")

    bilingual_share = V6_SHARES[profile]
    bilingual_tokens = int(TRAIN_TOKENS * bilingual_share)
    local_tokens = TRAIN_TOKENS - bilingual_tokens
    quotas = local_quotas(local_tokens)

    ranges = source_ranges(seed_manifest)
    arrays = [
        np.memmap(seed / "train" / item["path"], dtype="<u2", mode="r")
        for item in seed_manifest["train_files"]
    ]
    writer = StreamingShardWriter(root / "train", "train", 5_000_000)
    reports: list[dict] = []
    for name, count in quotas.items():
        start, available, original = ranges[name]
        if count > available:
            raise RuntimeError(f"quota exceeds admitted seed source: {name}")
        for chunk in slice_stream(arrays, start, count):
            writer.add(chunk)
        scaled_share = count / TRAIN_TOKENS
        reports.append(
            dict(
                original,
                source=dict(original["source"], share=scaled_share),
                emitted_tokens=count,
                parent_train_offset=start,
                parent_train_tokens=available,
                accounting="token-prefix remix from admitted recovery seed",
            )
        )

    emitted_bilingual, pair_docs = add_pair_tokens(
        writer, tokenizer, training_pairs, bilingual_tokens
    )
    if emitted_bilingual != bilingual_tokens:
        raise RuntimeError("bilingual token accounting mismatch")

    train_files = writer.finish()
    (root / "valid").mkdir(parents=True)
    valid_files = []
    for item in seed_manifest["valid_files"]:
        shutil.copyfile(seed / "valid" / item["path"], root / "valid" / item["path"])
        valid_files.append(item)

    result = dict(
        seed_manifest,
        train_files=train_files,
        valid_files=valid_files,
        sources=[r for r in seed_manifest["sources"] if r["partition"] == "valid"] + reports,
        train_tokens=TRAIN_TOKENS,
        validation_tokens=200_000,
        target_tokens=10_200_000,
        continuation_profile=f"arctus-v6-{profile}",
        continuation_from_tokens=START_TOKENS,
        target_cumulative_tokens=START_TOKENS + TRAIN_TOKENS,
        focus_shares={
            **{name: count / TRAIN_TOKENS for name, count in quotas.items()},
            "tatoeba_de_en": bilingual_share,
        },
        external_sources=[
            {
                "name": "tatoeba_de_en",
                "repo": TATOEBA_REPO,
                "license": TATOEBA_LICENSE,
                "snapshot": TATOEBA_DATE,
                "url": TATOEBA_URL,
                "archive_sha256": zip_sha,
                "emitted_tokens": bilingual_tokens,
                "paired_documents": pair_docs,
                "directions": ["en_to_de", "de_to_en"],
            }
        ],
        licenses=sorted(set(seed_manifest["licenses"]) | {TATOEBA_LICENSE}),
        parent_manifest_sha256=sha256(seed / "manifest.json"),
        prior_rejected_pilot_data=True,
        development_only=True,
        benchmark_content_included=False,
        bilingual_holdout_sha256=sha256(holdout_path),
        bilingual_holdout_pairs=HOLDOUT_PAIRS,
        synthetic_eos_added=0,
    )
    for field in ("created_unix", "new_deduplicated_documents", "deduplicated_documents"):
        result.pop(field, None)
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf8")

    if verify_files(root, "train", train_files) != TRAIN_TOKENS:
        raise RuntimeError("V6 train shard accounting mismatch")
    if verify_files(root, "valid", valid_files) != 200_000:
        raise RuntimeError("V6 validation shard accounting mismatch")
    return {
        "ok": True,
        "profile": profile,
        "train_tokens": TRAIN_TOKENS,
        "bilingual_tokens": bilingual_tokens,
        "bilingual_share": bilingual_share,
        "holdout_cases": len(holdout["cases"]),
        "holdout_sha256": sha256(holdout_path),
        "tatoeba_archive_sha256": zip_sha,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--tatoeba-zip", required=True)
    ap.add_argument("--profile", choices=sorted(V6_SHARES), required=True)
    args = ap.parse_args()
    print(
        json.dumps(
            build(
                Path(args.seed),
                Path(args.root),
                Path(args.tokenizer),
                Path(args.tatoeba_zip),
                args.profile,
            ),
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
