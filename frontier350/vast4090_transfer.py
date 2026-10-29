"""Prepare and verify a minimal local Arctus350 transfer bundle for Vast.

Pure Python stdlib for transfer/copy checks. Checkpoint metadata is inspected
through NERAL's existing restricted PyTorch loader; NO networking, cloud rental,
GPU training, auto-upload, secret handling or invocation of any external process.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

from frontier350.chat350_contract import EXPECTED_TOKENIZER_SHA256
from frontier350.vast4090_guard import (
    _load_verified_checkpoint, _sha, make_plan, sha256_file,
    validate_data_manifest, DEFAULT_INCREMENT
)

MAX_RECEIPT_BYTES = 2 * 1024 * 1024
TRANSFER_RECEIPT = "transfer-receipt.json"
_ALLOWED_PATH = re.compile(
    r"(?:inputs/original-resume\.pt|inputs/tokenizer\.json|"
    r"inputs/data/manifest\.json|"
    r"inputs/data/(?:train|valid)/[A-Za-z0-9_-]{1,110}\.bin)"
)


def _real_file(file: Path, description: str) -> Path:
    path = Path(file)
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{description} missing or symlinked")
    return path


def _select(files: list, minimum: int, partition: str) -> tuple[list, int]:
    if type(minimum) is not int or not 1024 <= minimum <= 100_000_000:
        raise ValueError("Invalid bounded subset token quota")
    selected, count = [], 0
    for entry in sorted(files, key=lambda obj: obj["path"]):
        selected.append(entry)
        count += int(entry["tokens"])
        if count >= minimum:
            break
    if count < minimum:
        raise ValueError(f"{partition} data too small for requested pilot subset")
    return selected, count


def _relative(directory: Path, rel: str) -> Path:
    if not isinstance(rel, str) or not _ALLOWED_PATH.fullmatch(rel):
        raise ValueError(f"unsafe transfer path: {rel!r}")
    file = directory.joinpath(*rel.split("/"))
    if file.is_symlink():
        raise ValueError("transfer input/output contains a symlink")
    return file


def _copy_and_receipt(source: Path, root: Path, rel: str, expected_sha: str,
                      inventory: list):
    _sha(expected_sha, "source file SHA256")
    src = _real_file(source, "transfer source")
    dest = _relative(root, rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with src.open("rb") as origin, dest.open("xb") as output:
        shutil.copyfileobj(origin, output, 4 * 1024 * 1024)
    try:
        os.chmod(dest, 0o600)
    except OSError:
        pass  # Windows ACLs must be configured separately by operator.
    if sha256_file(dest) != expected_sha:
        raise ValueError(f"SHA256 changed during local copy: {rel}")
    inventory.append({
        "path": rel, "size_bytes": dest.stat().st_size,
        "sha256": expected_sha
    })


def verify_bundle(root: Path, *, expected_checkpoint_sha256: str | None = None,
                  expected_receipt_sha256: str | None = None,
                  test_tokenizer_sha256: str | None = None) -> dict:
    """Verify all declared files, limited inventory, manifest and locked tokenizer.

    A receipt cannot establish origin authenticity alone: compare original
    checkpoint SHA with the separately recorded trusted local SHA as well.
    """
    base = Path(root)
    receipt_file = _real_file(base / TRANSFER_RECEIPT, "transfer receipt")
    if expected_receipt_sha256 is not None:
        _sha(expected_receipt_sha256, "trusted source receipt")
        if sha256_file(receipt_file) != expected_receipt_sha256:
            raise ValueError("Transfer receipt SHA256 differs from trusted local copy")
    if receipt_file.stat().st_size > MAX_RECEIPT_BYTES:
        raise ValueError("Transfer receipt unexpectedly large")
    receipt = json.loads(receipt_file.read_text(encoding="utf8"))
    if receipt.get("schema") != 1 or receipt.get("uploaded") is not False:
        raise ValueError("Invalid transfer receipt")
    entries = receipt.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Empty transfer inventory")
    if len(entries) > 2000:
        raise ValueError("Unbounded transfer inventory")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Invalid transfer file record")
        rel = entry.get("path")
        dest = _relative(base, rel)
        if rel in seen:
            raise ValueError("Duplicate transfer inventory path")
        seen.add(rel)
        _real_file(dest, "bundled file")
        digest = entry.get("sha256")
        _sha(digest, "transfer file SHA256")
        if type(entry.get("size_bytes")) is not int or entry["size_bytes"] != dest.stat().st_size:
            raise ValueError("Transfer file size mismatch")
        if sha256_file(dest) != digest:
            raise ValueError(f"Transfer SHA256 mismatch: {rel}")

    required = {"inputs/original-resume.pt", "inputs/tokenizer.json",
                "inputs/data/manifest.json"}
    if not required.issubset(seen):
        raise ValueError("Missing transfer-critical files")
    expected = seen | {TRANSFER_RECEIPT}
    observed = set()
    for item in base.rglob("*"):
        if item.is_symlink():
            raise ValueError("Symlink not allowed in transfer bundle")
        if item.is_file():
            observed.add(item.relative_to(base).as_posix())
    if observed != expected:
        raise ValueError("Unexpected/missing file in transfer bundle")
    if next(e["sha256"] for e in entries if e["path"]=="inputs/original-resume.pt") != receipt.get("checkpoint_sha256"):
        raise ValueError("Checkpoint SHA256 differs from bundle receipt")
    if expected_checkpoint_sha256 is not None:
        _sha(expected_checkpoint_sha256, "trusted original checkpoint")
        if receipt["checkpoint_sha256"] != expected_checkpoint_sha256:
            raise ValueError("Checkpoint SHA256 differs from independently verified original")
    expected_tok = test_tokenizer_sha256 or EXPECTED_TOKENIZER_SHA256
    if next(e["sha256"] for e in entries if e["path"]=="inputs/tokenizer.json") != expected_tok:
        raise ValueError("Original tokenizer SHA256 not locked")
    guard = validate_data_manifest(base / "inputs/data/manifest.json")
    assert guard["train_shards"] > 0 and guard["valid_shards"] > 0
    return {
        "verified": True, "transfer_complete": True,
        "gpu_rented": False, "training_started": False,
        "files_verified": len(entries),
        "selected_train_tokens": receipt["selected_train_tokens"],
        "selected_valid_tokens": receipt["selected_valid_tokens"],
        "checkpoint_processed_tokens": receipt["checkpoint_processed_tokens"],
        "checkpoint_sha256": receipt["checkpoint_sha256"],
        "source_manifest_sha256": receipt["source_manifest_sha256"],
        "receipt_file": str(receipt_file),
        "note": "Local hash and format checks only; provider host is not attested.",
    }


def build_bundle(*, checkpoint: Path, expected_checkpoint_sha256: str,
                 tokenizer: Path, corpus_manifest: Path, output: Path,
                 min_train_tokens: int = 512_000,
                 min_valid_tokens: int = 16_384,
                 checkpoint_loader=_load_verified_checkpoint,
                 test_tokenizer_sha256: str | None = None) -> dict:
    """Only local, explicit pack operation; no upload and no GPU.

    The subset is for a short engineering pilot, not a new deduplicated
    foundation-training campaign. The operator must audit corpus chronology
    and possible historical repetition before real weight updates.
    """
    ck = _real_file(checkpoint, "original Arctus checkpoint")
    tok = _real_file(tokenizer, "Arctus BPE tokenizer")
    manifest = _real_file(corpus_manifest, "original BPE manifest")
    dest_root = Path(output)
    if dest_root.exists():
        raise FileExistsError("Output folder exists: no overwrite of private files")
    if not dest_root.parent.is_dir() or dest_root.parent.is_symlink():
        raise ValueError("Output parent must be an existing trusted directory")
    if dest_root.resolve().is_relative_to(manifest.parent.resolve()):
        raise ValueError("Destination cannot be nested inside the source corpus")
    if any(source.resolve().is_relative_to(dest_root.resolve()) for source in (ck, tok, manifest)):
        raise ValueError("Destination cannot contain the original input files")
    _sha(expected_checkpoint_sha256, "original checkpoint")
    if sha256_file(ck) != expected_checkpoint_sha256:
        raise ValueError("Original Arctus checkpoint SHA256 mismatch")
    expected_tok = test_tokenizer_sha256 or EXPECTED_TOKENIZER_SHA256
    _sha(expected_tok, "original tokenizer")
    if sha256_file(tok) != expected_tok:
        raise ValueError("Original Arctus tokenizer SHA256 mismatch")

    # Fail BEFORE copying if original optimizer/schedule is not resumable.
    meta = checkpoint_loader(ck)
    p = make_plan(meta, extra_tokens=DEFAULT_INCREMENT)
    if any(meta.get(key) is None for key in ("torch_rng","python_rng","numpy_rng")):
        raise ValueError("Original checkpoint lacks training RNG state")

    # Existing format SHA-verification checks ALL source shards. The packer
    # only copies a small documented deterministic subset of those verified.
    validate_data_manifest(manifest)
    raw = json.loads(manifest.read_text(encoding="utf8"))
    train_entries, train_total = _select(raw["train_files"], min_train_tokens, "train")
    valid_entries, valid_total = _select(raw["valid_files"], min_valid_tokens, "valid")
    selected = [("train", row) for row in train_entries] + [
        ("valid", row) for row in valid_entries]
    if len(selected) > 1997:
        raise ValueError("Too many shards for bounded transfer manifest")
    subset = {
        "schema": 2,
        "architecture": "frontier350-v1",
        "corpus_role": raw.get("corpus_role", "foundation-pretraining"),
        "benchmark_content_included": False,
        "tokenizer": raw["tokenizer"],
        "train_files": train_entries,
        "valid_files": valid_entries,
        "subset_for_pilot": True,
        "source_manifest_sha256": sha256_file(manifest),
        "selected_train_tokens": train_total,
        "selected_valid_tokens": valid_total,
        "do_not_claim_new_unique_training_data": True,
    }

    bytes_needed = ck.stat().st_size + tok.stat().st_size + sum(
        (manifest.parent / part / row["path"]).stat().st_size
        for part, row in selected)
    bytes_needed += 100_000_000  # receipt, metadata and a small safety margin
    if shutil.disk_usage(dest_root.parent).free < int(bytes_needed * 1.05):
        raise ValueError("Insufficient local disk for staged private data")
    dest_root.mkdir(mode=0o700)
    try:
        inventory = []
        _copy_and_receipt(ck, dest_root, "inputs/original-resume.pt", expected_checkpoint_sha256, inventory)
        _copy_and_receipt(tok, dest_root, "inputs/tokenizer.json", expected_tok, inventory)
        for part, row in selected:
            _copy_and_receipt(manifest.parent / part / row["path"], dest_root,
                              f"inputs/data/{part}/{row['path']}", row["sha256"], inventory)
        manifest_out = dest_root / "inputs/data/manifest.json"
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with manifest_out.open("x", encoding="utf8") as handle:
            json.dump(subset, handle, ensure_ascii=False, indent=2)
        try:
            os.chmod(manifest_out, 0o600)
        except OSError:
            pass
        inventory.append({
            "path": "inputs/data/manifest.json",
            "size_bytes": manifest_out.stat().st_size,
            "sha256": sha256_file(manifest_out)
        })
        receipt = {
            "schema": 1, "purpose": "Arctus350 private offline GPU pilot transfer",
            "checkpoint_sha256": expected_checkpoint_sha256,
            "checkpoint_processed_tokens": meta["processed_tokens"],
            "checkpoint_target_tokens": p["schedule_target_tokens"],
            "continue_without_schedule_reset": True,
            "source_manifest_sha256": subset["source_manifest_sha256"],
            "selected_train_tokens": train_total,
            "selected_valid_tokens": valid_total,
            "files_packed": len(inventory),
            "private_weights_in_bundle": True,
            "uploaded": False,
            "training_started": False,
            "gpu_rented": False,
            "trained_tokens": 0,
            "warning": "Transfer to a marketplace host reveals unencrypted private weights to the host. "
                       "Receipt verifies byte identity, not host trust or training quality.",
            "files": inventory
        }
        (dest_root / TRANSFER_RECEIPT).write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf8")
        try:
            os.chmod(dest_root / TRANSFER_RECEIPT, 0o600)
        except OSError:
            pass
        verify_bundle(dest_root, test_tokenizer_sha256=expected_tok)
        return {**receipt, "receipt_sha256": sha256_file(dest_root / TRANSFER_RECEIPT)}
    except BaseException:
        shutil.rmtree(dest_root)
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--pack", action="store_true", help="Copy a local bounded pilot subset to a NEW private folder")
    group.add_argument("--verify", action="store_true", help="Check a complete transferred bundle, no write")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256", default="")
    parser.add_argument("--receipt-sha256", default="", help="SHA256 of transfer receipt, independently recorded on trusted PC")
    parser.add_argument("--tokenizer", type=Path)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--min-train-tokens", type=int, default=512_000)
    parser.add_argument("--min-valid-tokens", type=int, default=16_384)
    a = parser.parse_args(argv)
    if a.verify:
        if a.root is None or not a.checkpoint_sha256 or not a.receipt_sha256:
            parser.error("--verify requires --root, --checkpoint-sha256 and --receipt-sha256 from TRUSTED local records")
        result = verify_bundle(
            a.root, expected_checkpoint_sha256=a.checkpoint_sha256,
            expected_receipt_sha256=a.receipt_sha256)
    elif a.pack:
        if not all((a.checkpoint, a.checkpoint_sha256, a.tokenizer, a.data_manifest, a.out)):
            parser.error("--pack requires --checkpoint, --checkpoint-sha256, "
                         "--tokenizer, --data-manifest and --out")
        result = build_bundle(
            checkpoint=a.checkpoint, expected_checkpoint_sha256=a.checkpoint_sha256,
            tokenizer=a.tokenizer, corpus_manifest=a.data_manifest,
            output=a.out, min_train_tokens=a.min_train_tokens,
            min_valid_tokens=a.min_valid_tokens)
        result = {key: value for key, value in result.items() if key != "files"}
    else:
        result = {
            "state": "READ_ONLY_TRANSFER_PLAN",
            "action_required": "Use --pack on trusted local disk after locating ORIGINAL 2B checkpoint and BPE corpus",
            "requires_uploaded_checkpoint": False,
            "auto_upload": False, "gpu_rented": False, "training_started": False,
            "checkpoint_required_sha256": "verify against the separately recorded native checkpoint receipt",
            "note": "A verified transfer package is not a claim that training data are novel or model weights improved.",
        }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
