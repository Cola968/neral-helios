"""Audit NERAL NIS2 evidence coverage without inventing a model score."""
from __future__ import annotations
import argparse
import json

from neral_intelligence_score_v2 import nis2_evidence_status


def main() -> None:
    parser = argparse.ArgumentParser(description="Report NIS2 evidence coverage; no inference/training")
    parser.add_argument("report_dir", help="Directory containing frozen raw-model benchmark reports")
    args = parser.parse_args()
    result = nis2_evidence_status(args.report_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
