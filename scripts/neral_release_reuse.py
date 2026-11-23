"""Validate already-published NERAL prerelease assets before allowing CI reruns."""
from __future__ import annotations
import argparse
import hashlib
import json
import re
from pathlib import Path

ARCHIVE = "NERAL-1.0-Local-Preview.zip"
ASSETS = (ARCHIVE, "SHA256SUMS.txt", "SOURCE-COMMIT.txt")


def verify_existing_release(local: Path, existing: Path, metadata: dict, source_sha: str) -> bool:
    if not isinstance(source_sha, str) or re.fullmatch(r"[0-9a-f]{40}", source_sha) is None:
        raise ValueError("Invalid exact source SHA")
    if not isinstance(metadata, dict) or (
        metadata.get("tagName") != "neral-preview-" + source_sha[:12]
        or metadata.get("targetCommitish") != source_sha
        or metadata.get("isPrerelease") is not True
        or metadata.get("isDraft") is not False
    ):
        raise ValueError("Existing release tag, source commit or state does not match")
    local, existing = Path(local), Path(existing)
    for name in ASSETS:
        left, right = local / name, existing / name
        if not left.is_file() or not right.is_file() or left.is_symlink() or right.is_symlink():
            raise ValueError("Missing or unsafe release asset: " + name)
        if left.stat().st_size != right.stat().st_size:
            raise ValueError("Published release asset differs: " + name)
        if left.stat().st_size > 32 * 1024 * 1024:
            raise ValueError("Unreasonably large release asset: " + name)
        if left.read_bytes() != right.read_bytes():
            raise ValueError("Published release asset differs: " + name)
    expected_sha = hashlib.sha256((local / ARCHIVE).read_bytes()).hexdigest()
    if (local / "SHA256SUMS.txt").read_text(encoding="utf-8") != (
        f"{expected_sha}  {ARCHIVE}\n"
    ):
        raise ValueError("Invalid release checksum manifest")
    if (local / "SOURCE-COMMIT.txt").read_text(encoding="ascii") != source_sha + "\n":
        raise ValueError("Invalid release source manifest")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-dir", required=True, type=Path)
    parser.add_argument("--existing-dir", required=True, type=Path)
    parser.add_argument("--metadata-json", required=True, type=Path)
    parser.add_argument("--source-sha", required=True)
    args = parser.parse_args()
    metadata = json.loads(args.metadata_json.read_text(encoding="utf8"))
    verify_existing_release(args.local_dir, args.existing_dir, metadata, args.source_sha)
    print("Existing prerelease is byte-for-byte identical to the verified current commit")


if __name__ == "__main__":
    main()
