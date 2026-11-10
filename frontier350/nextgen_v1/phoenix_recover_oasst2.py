"""Recover the exact pinned public OASST2 archive on the owner's private PC.

Only the known 54 MB source archive is downloaded. Never reads model weights,
creates a holdout, authorizes a dataset, trains, or uploads source material.
The target is created only after a complete size + SHA256 verification.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import secrets
import urllib.parse
import urllib.request

from fetch_oasst2_sft50 import ARCHIVE_NAME, UPSTREAM_SHA256, UPSTREAM_URL

EXPECTED_BYTES = 54_301_900
CHUNK_BYTES = 1024 * 1024
ALLOWED_HOSTS = {"huggingface.co", "cdn-lfs.huggingface.co", "cas-bridge.xethub.hf.co"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recover(root: Path, *, open_url=None) -> dict:
    """Download into a private temporary file, verify, then publish without overwriting."""
    if root.is_symlink() or not root.is_dir():
        raise ValueError("private assets root must exist and must not be a symlink")
    root = root.resolve(strict=True)
    workspace = os.environ.get("GITHUB_WORKSPACE")
    if workspace and root.is_relative_to(Path(workspace).resolve()):
        raise ValueError("refusing to place raw OASST2 data inside the GitHub checkout")
    source_dir = root / "sft50_sources"
    if source_dir.is_symlink() or (source_dir.exists() and not source_dir.is_dir()):
        raise ValueError("source directory is unsafe")
    source_dir.mkdir(exist_ok=True)
    target = source_dir / ARCHIVE_NAME
    if target.is_symlink():
        raise ValueError("archive symlink refused")
    if target.exists():
        if not target.is_file() or target.stat().st_size != EXPECTED_BYTES or _sha256(target) != UPSTREAM_SHA256:
            raise ValueError("existing archive differs from pinned source; refusing overwrite")
        return {"status": "already_verified", "bytes": EXPECTED_BYTES,
                "pinned_sha256_match": True, "dataset_approved": False, "training_started": False}
    part = source_dir / (ARCHIVE_NAME + ".partial-" + secrets.token_hex(12))
    if open_url is None:
        open_url = urllib.request.urlopen
    request = urllib.request.Request(UPSTREAM_URL, headers={"User-Agent": "NERAL-Phoenix-Pinned-Recovery/1.0"})
    try:
        with open_url(request, timeout=45) as response:
            final = urllib.parse.urlparse(response.geturl())
            if final.scheme != "https" or final.hostname not in ALLOWED_HOSTS:
                raise ValueError("unexpected OASST2 redirect destination")
            digest = hashlib.sha256()
            count = 0
            with part.open("xb") as output:
                while True:
                    chunk = response.read(CHUNK_BYTES)
                    if not chunk:
                        break
                    count += len(chunk)
                    if count > EXPECTED_BYTES:
                        raise ValueError("OASST2 archive exceeds pinned length")
                    output.write(chunk)
                    digest.update(chunk)
        if count != EXPECTED_BYTES or digest.hexdigest() != UPSTREAM_SHA256:
            raise ValueError("OASST2 pinned byte length or SHA-256 mismatch")
        # An exclusive hard link publishes the fully verified file atomically and
        # fails if another process already created the target. Same directory/volume.
        os.link(part, target)
        return {"status": "recovered_verified", "bytes": count,
                "pinned_sha256_match": True, "dataset_approved": False, "training_started": False}
    finally:
        part.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="explicitly authorize the ~54 MB pinned download")
    args = parser.parse_args()
    if not args.execute:
        parser.error("no download without --execute")
    print("PHOENIX_ARCHIVE_RECOVERY " + json.dumps(recover(args.assets_root), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
