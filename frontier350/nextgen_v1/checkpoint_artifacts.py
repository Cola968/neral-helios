"""Read-only checkpoint-byte audit for a pinned NextGen stage ledger.

This verifies four or more ORIGINAL local files against expected SHA-256 digests.
It never unpickles torch files, starts training, authenticates the source of the
manifest, establishes exact resume or promotes a model. Run on the local machine
with a separately retained expected ledger digest and independently located files.
"""
from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat

from stage_ledger import _bad_constant, _no_duplicates, _read_events, _sha, audit_ledger

SAFE_FILENAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
MAX_FILES = 257


def _hash_regular_file(path: Path) -> tuple[str, int]:
    """Stream file bytes through a no-follow descriptor; reject changes during hashing."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_size <= 0:
            raise ValueError("checkpoint must be a nonempty regular file")
        digest = hashlib.sha256()
        with os.fdopen(os.dup(fd), "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = os.fstat(fd)
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns,
            before.st_ctime_ns) != (after.st_dev, after.st_ino, after.st_size,
                                    after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError("checkpoint changed while hashing")
        return digest.hexdigest(), before.st_size
    finally:
        os.close(fd)


def verify_artifacts(record: dict, ledger_bytes: bytes, expected_ledger_sha256: str,
                     checkpoint_dir: Path, manifest: dict) -> dict:
    """Check stage consistency THEN require one matching on-disk file per checkpoint.

    The external SHA supplied here must NOT be calculated from the untrusted ledger
    in the same verification step. An attacker can forge a consistent ledger.
    """
    ledger_result = audit_ledger(record, ledger_bytes, expected_ledger_sha256)
    events = _read_events(ledger_bytes)
    expected = [events[0]["checkpoint_start_sha256"]] + [
        event["checkpoint_end_sha256"] for event in events
    ]
    if len(expected) > MAX_FILES or len(set(expected)) != len(expected):
        raise ValueError("checkpoint identity sequence must be unique and bounded")
    if not isinstance(manifest, dict) or set(manifest) != {"schema", "checkpoints"} or manifest["schema"] != 1:
        raise ValueError("strict checkpoint manifest v1 required")
    rows = manifest["checkpoints"]
    if not isinstance(rows, list) or len(rows) != len(expected):
        raise ValueError("checkpoint manifest must cover every boundary exactly once")
    if checkpoint_dir.is_symlink() or not checkpoint_dir.is_dir():
        raise ValueError("real existing checkpoint directory required")
    seen_sha, seen_names, checked = set(), set(), []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"sha256", "filename"}:
            raise ValueError("strict checkpoint entry required")
        digest = _sha(row["sha256"], "manifest checkpoint")
        filename = row["filename"]
        if not isinstance(filename, str) or SAFE_FILENAME.fullmatch(filename) is None or filename in (".", ".."):
            raise ValueError("checkpoint filename must be a safe basename")
        if digest in seen_sha or filename in seen_names:
            raise ValueError("duplicate checkpoint digest or filename")
        seen_sha.add(digest)
        seen_names.add(filename)
        path = checkpoint_dir / filename
        if path.is_symlink():
            raise ValueError("symlink checkpoint refused")
        observed, size = _hash_regular_file(path)
        if not hmac.compare_digest(digest, observed):
            raise ValueError(f"checkpoint bytes do not match manifest: {filename}")
        checked.append({"sha256": observed, "size_bytes": size, "filename": filename})
    if seen_sha != set(expected):
        raise ValueError("checkpoint files do not match every ordered stage boundary")
    return {
        "schema": 1,
        "classification": "local_checkpoint_bytes_verified_not_training_proof",
        "ledger_sha256": ledger_result["ledger_sha256"],
        "ledger_consistent": True,
        "checkpoint_count": len(checked),
        "checkpoint_bytes_verified": True,
        "checkpoint_files": checked,
        "contract_passed": ledger_result["contract_passed"],
        "contract_failures": ledger_result["contract_failures"],
        "independent_training_verified": False,
        "exact_resume_verified": False,
        "eligible_for_production": False,
        "warning": "Bytes match a separately pinned, internally consistent ledger, but checkpoint files might be untrusted/forged. No pickle was loaded. Independently audit provenance, optimizer/RNG/data order, actual GPU timing, exact resume, full NIS2 and sealed holdout; NEVER automatically promote.",
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--evidence", type=Path, required=True)
    p.add_argument("--ledger", type=Path, required=True)
    p.add_argument("--expected-ledger-sha256", required=True)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error("refusing to overwrite an existing result")
    strict = lambda path: json.loads(path.read_text(encoding="utf-8"),
                                     object_pairs_hook=_no_duplicates, parse_constant=_bad_constant)
    report = verify_artifacts(strict(args.evidence), args.ledger.read_bytes(),
                              args.expected_ledger_sha256, args.checkpoint_dir,
                              strict(args.manifest))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["contract_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
