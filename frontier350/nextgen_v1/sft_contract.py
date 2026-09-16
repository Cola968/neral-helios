"""Prepare an isolated *local-only* 350M SFT pilot pack; never trains or uploads.

All inputs are trusted local files. A signed/independent near-duplicate and licensing
review is still required: JSON assertions alone cannot establish dataset legality.
No frozen benchmark content is imported by this module.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import unicodedata

HEX64 = re.compile(r"^[0-9a-f]{64}$")
FORBIDDEN = ("<|bos|>", "<|eos|>", "<|pad|>", "\nAssistant:", "\nUser:")
MAX_ROWS = 20_000  # deliberately bounded, experimental pilot only


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def hex_digest(value: object, name: str) -> str:
    if not isinstance(value, str) or not HEX64.fullmatch(value):
        raise ValueError(f"invalid {name} SHA-256")
    return value


def norm(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def prompt_hash(prompt: str) -> str:
    return hashlib.sha256(norm(prompt).encode("utf-8")).hexdigest()


def read_json(path: Path) -> object:
    return json.loads(path.read_text(encoding="utf-8"))


def check_manifest(manifest: dict, train: Path, valid: Path, tokenizer: Path,
                   holdout: Path, screen: Path) -> dict:
    if manifest.get("version") != 1:
        raise ValueError("SFT manifest v1 required")
    for field, path in (("train_sha256", train), ("validation_sha256", valid),
                        ("tokenizer_sha256", tokenizer), ("holdout_sha256", holdout),
                        ("screen_report_sha256", screen)):
        expected = hex_digest(manifest.get(field), field)
        if sha(path) != expected:
            raise ValueError(f"{field}: content changed")
    sources = manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise ValueError("nonempty sources required")
    source_ids = set()
    for source in sources:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str) or not source["id"]:
            raise ValueError("source id required")
        if source["id"] in source_ids:
            raise ValueError("duplicate source id")
        source_ids.add(source["id"])
        hex_digest(source.get("revision_sha256"), "source revision")
        if not isinstance(source.get("license"), str) or not source["license"].strip():
            raise ValueError("source license missing")
        if any(source.get(k) is not True for k in
               ("approved_for_training", "privacy_reviewed", "benchmark_screened")):
            raise ValueError("source is not approved for training")
    report = read_json(screen)
    if not isinstance(report, dict) or report.get("version") != 1:
        raise ValueError("independent screening report required")
    for key in ("train_sha256", "validation_sha256", "holdout_sha256"):
        if report.get(key) != manifest[key]:
            raise ValueError(f"screen report {key} mismatch")
    if report.get("approved") is not True or any(report.get(key) != 0 for key in
        ("exact_overlap_count", "near_overlap_count", "privacy_hits")):
        raise ValueError("screening did not clear training inputs")
    return {"source_ids": source_ids, "report": report}


def parse_rows(path: Path, split: str, source_ids: set[str], seen: set[str],
               holdout_hashes: set[str]) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{split}:{index}: blank row")
            if index > MAX_ROWS:
                raise ValueError(f"{split}: exceeds bounded pilot size")
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"id", "source_id", "user", "assistant"}:
                raise ValueError(f"{split}:{index}: invalid row schema")
            if not isinstance(row["id"], str) or not row["id"].strip():
                raise ValueError(f"{split}:{index}: missing id")
            if row["source_id"] not in source_ids:
                raise ValueError(f"{split}:{index}: unapproved source")
            for field in ("user", "assistant"):
                text = row[field]
                if not isinstance(text, str) or not text.strip() or len(text) > 8192:
                    raise ValueError(f"{split}:{index}: invalid {field}")
                if any(marker in text for marker in FORBIDDEN):
                    raise ValueError(f"{split}:{index}: injected role/special marker")
            digest = prompt_hash(row["user"])
            if digest in holdout_hashes:
                raise ValueError(f"{split}:{index}: exact frozen holdout overlap")
            if digest in seen:
                raise ValueError(f"{split}:{index}: repeated user prompt / split leakage")
            seen.add(digest)
            rows.append(row)
    if not rows:
        raise ValueError(f"{split}: empty")
    return rows


def encode_row(row: dict, tokenizer: object, bos: int, eos: int,
               vocab_size: int, seq: int) -> dict:
    # Matches the NIS2 User/Assistant prefix. The leading answer space is explicit.
    prefix = "User: " + row["user"].strip() + "\nAssistant:"
    answer = " " + row["assistant"].strip()
    prefix_ids = [bos, *tokenizer.encode(prefix, add_special_tokens=False).ids]
    answer_ids = list(tokenizer.encode(answer, add_special_tokens=False).ids)
    if not answer_ids:
        raise ValueError("empty tokenized answer")
    full = [*prefix_ids, *answer_ids, eos]
    if len(full) < 3 or len(full) - 1 > seq:
        raise ValueError("sequence exceeds pilot context; never truncate an EOS")
    if any(not isinstance(t, int) or not 0 <= t < vocab_size for t in full):
        raise ValueError("tokenizer emitted out-of-range ids")
    x = full[:-1]
    labels = [-100] * (len(prefix_ids) - 1) + full[len(prefix_ids):]
    if len(x) != len(labels) or labels[-1] != eos or labels.count(eos) != 1:
        raise AssertionError("incorrect supervised EOS/target alignment")
    if any(label != -100 for label in labels[:len(prefix_ids) - 1]):
        raise AssertionError("user prefix not fully masked")
    return {"id": row["id"], "input_ids": x, "labels": labels,
            "answer_tokens_including_eos": len(answer_ids) + 1}


def prepare(manifest: Path, train: Path, validation: Path, tokenizer_path: Path,
            holdout_path: Path, screening: Path, out: Path, seq: int = 512) -> dict:
    if not 32 <= seq <= 8192:
        raise ValueError("invalid context")
    if out.exists():
        raise FileExistsError("refusing to reuse/overwrite an output directory")
    spec = read_json(manifest)
    if not isinstance(spec, dict):
        raise ValueError("manifest object required")
    approved = check_manifest(spec, train, validation, tokenizer_path, holdout_path, screening)
    registry = read_json(holdout_path)
    if not isinstance(registry, list) or not registry:
        raise ValueError("frozen holdout SHA-256 registry required")
    hashes = {hex_digest(value, "holdout prompt") for value in registry}
    if len(hashes) != len(registry):
        raise ValueError("duplicate holdout hashes")
    seen: set[str] = set()
    train_rows = parse_rows(train, "train", approved["source_ids"], seen, hashes)
    valid_rows = parse_rows(validation, "validation", approved["source_ids"], seen, hashes)
    from tokenizers import Tokenizer  # deliberately optional until local preparation
    tok = Tokenizer.from_file(str(tokenizer_path))
    if tok.get_vocab_size() != 16384:
        raise ValueError("incompatible 350M tokenizer vocabulary")
    bos = tok.token_to_id("<|bos|>")
    eos = tok.token_to_id("<|eos|>")
    if bos is None or eos is None or bos == eos:
        raise ValueError("missing or invalid BOS/EOS")
    prepared = {
        "train": [encode_row(row, tok, bos, eos, 16384, seq) for row in train_rows],
        "validation": [encode_row(row, tok, bos, eos, 16384, seq) for row in valid_rows],
    }
    # Create only after ALL validations and tokenization succeed.
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": 1, "classification": "local_only_untrained_sft_pack",
              "tokenizer_sha256": spec["tokenizer_sha256"],
              "source_manifest_sha256": sha(manifest), "screen_report_sha256": sha(screening),
              "context": seq, "bos_id": bos, "eos_id": eos,
              "rows": {key: len(value) for key, value in prepared.items()},
              "note": "SHA/approval assertions do not prove legal provenance or near-duplicate safety. Review original screening evidence independently. No training has run."}
    try:
        for split, rows in prepared.items():
            path = out / f"{split}.jsonl"
            with path.open("x", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row, separators=(",", ":")) + "\n")
            report[f"{split}_packed_sha256"] = sha(path)
        (out / "pack_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    except Exception:
        # Partial output is explicitly not a usable training pack.
        (out / "INVALID_PACK").write_text("Packing failed; do not train.\n", encoding="utf-8")
        raise
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("manifest", "train", "validation", "tokenizer", "holdout-hashes", "screening", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--seq", type=int, default=512)
    a = p.parse_args()
    result = prepare(a.manifest, a.train, a.validation, a.tokenizer,
                     a.holdout_hashes, a.screening, a.out, a.seq)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
