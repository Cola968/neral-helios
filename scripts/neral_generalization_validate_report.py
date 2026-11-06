"""Separate CPU-only allowlist validation before uploading any benchmark artifact."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from scripts.neral_generalization_gpu import validate_report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--summary",type=Path,required=True)
    args=p.parse_args(argv)
    file=args.summary
    if file.is_symlink() or not file.is_file() or file.stat().st_size>12000:
        p.error("Only one bounded scalar summary is permitted")
    raw=file.read_bytes()
    if not raw.endswith(b"\n") or b"\x00" in raw:
        raise ValueError("Noncanonical report")
    report=json.loads(raw)
    validate_report(report)
    print("NERAL_32_SCALAR_REPORT_VERIFIED_NO_PROMPTS_OR_TRAINING")
    return 0


if __name__=="__main__":
    raise SystemExit(main())
