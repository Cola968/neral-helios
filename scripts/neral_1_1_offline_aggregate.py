"""Pure offline aggregate for original 43722550890 artifact.

No model, GPU, Docker, network, prompt or answer contents are emitted.
Original ZIP and all previous datasets are opened read-only.
"""
from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path

RUN = "43722550890"
EXPECTED_RAW_SHA = "e1cb153c59d0712b63ad77df86074ac2b763ee85e6ee8e2ccea099bae72141f9"
EXPECTED_SEAL = "c43287a4ca6eb18fe4f45a45706f7af8155d27033638bb13c397a7417aacb157"


def aggregate(archive: Path) -> dict:
    with zipfile.ZipFile(archive) as z:
        required = {"raw-200.jsonl", "summary.json", "new-seal.json", "inference-complete.json"}
        if not required.issubset(z.namelist()):
            raise ValueError("raw_evidence_missing")
        raw = z.read("raw-200.jsonl")
        digest = hashlib.sha256(raw).hexdigest()
        summary = json.loads(z.read("summary.json"))
        seal = json.loads(z.read("new-seal.json"))
        if (digest != EXPECTED_RAW_SHA or digest != summary.get("raw_sha256")
                or summary.get("source_run_id") != RUN
                or summary.get("sealed_sha256") != EXPECTED_SEAL
                or seal.get("dataset_sha256") != EXPECTED_SEAL):
            raise ValueError("raw_evidence_mismatch")
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    if (len(rows) != 200
            or Counter(item.get("arm") for item in rows) != {"qwen": 100, "neral": 100}):
        raise ValueError("raw_evidence_invalid_count")
    baseline = {x["id"]: x for x in rows if x["arm"] == "qwen"}
    candidate = {x["id"]: x for x in rows if x["arm"] == "neral"}
    if set(baseline) != set(candidate) or len(baseline) != 100:
        raise ValueError("raw_evidence_unpaired")
    failed = [x for x in candidate.values() if x.get("correct") is False]
    if len(failed) != 23:
        raise ValueError("raw_evidence_invalid_outcome")
    report = {"run_id": RUN, "source_commit": summary["source_commit"],
              "raw_records": 200, "sha256_verified": True,
              "failure_count": 23, "categories": {}}
    for category in ("logic", "mathematics"):
        cases = [x for x in candidate.values() if x["category"] == category]
        failures = [x for x in cases if x.get("correct") is False]
        report["categories"][category] = {
            "cases": len(cases), "failed": len(failures),
            "failed_families": dict(sorted(Counter(x["family"] for x in failures).items())),
            "candidate_token_limit": sum(x.get("error_code") == "token_limit" for x in failures),
            "baseline_token_limit": sum(baseline[x["id"]].get("error_code") == "token_limit"
                                        for x in failures),
            "explicit_safe_refusal": sum(x.get("decision") == "explicit_safe_refusal"
                                         for x in failures),
            "attempted_schema_extraction": sum(bool(x.get("extra", {}).get("extraction_attempts"))
                                               for x in cases),
            "coverage_reason_recorded": sum(x.get("extra", {}).get("coverage_reason") is not None
                                            for x in cases),
            "source_verified_z3": sum(x.get("proof", {}).get("z3") is True for x in cases),
        }
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m scripts.neral_1_1_offline_aggregate ARTIFACT.zip")
    print(json.dumps(aggregate(Path(sys.argv[1])), indent=2, sort_keys=True))
