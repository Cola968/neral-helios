"""CPU-only importer for the sealed V3.9 97/100 RTX4070 result artifact."""
import argparse
import json
from pathlib import Path
import zipfile

from neral_verified_training import import_benchmark

SOURCE_SHA = "062007185d6721cec8392993f3cff3ef860d80ef"
RUN_ID = 44810676439


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--details-zip", type=Path, required=True)
    parser.add_argument("--summary-zip", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("datasets/neral_v39_seed97.jsonl"))
    args = parser.parse_args(argv)
    with zipfile.ZipFile(args.details_zip) as archive:
        details = archive.read("benchmark_details.jsonl")
    with zipfile.ZipFile(args.summary_zip) as archive:
        summary = json.loads(archive.read("summary.json"))
    count = import_benchmark(details, summary, args.out,
                             expected_source=SOURCE_SHA, expected_pass=97)
    print(json.dumps({"status":"seed97_cpu_only","source_run":RUN_ID,
                      "rows":count,"path":str(args.out),
                      "benchmark_must_not_be_reused_as_holdout":True}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
