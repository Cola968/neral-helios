from __future__ import annotations

"""Fail-closed parity gate for proposed single-load health evaluator.

Evaluate the SAME frozen checkpoint using the independent and combined implementations
with the same tokenizer, validation data, rotation and arguments. No gate is changed.
"""

import argparse
import json
from pathlib import Path


def compare(legacy_smoke: dict, combined_smoke: dict,
            legacy_validation: dict, combined_validation: dict) -> dict:
    required = ("checkpoint_sha256", "checkpoint_processed_tokens", "tokenizer_sha256")
    identities = [{key: report.get(key) for key in required} for report in
                  (legacy_smoke, combined_smoke, legacy_validation, combined_validation)]
    identical_identity = all(item == identities[0] for item in identities)
    smoke_exact = legacy_smoke == combined_smoke
    validation_exact = legacy_validation == combined_validation
    passed = identical_identity and smoke_exact and validation_exact and bool(identities[0]["checkpoint_sha256"])
    return {"parity_pass": passed, "checkpoint_sha256": identities[0]["checkpoint_sha256"],
            "checkpoint_processed_tokens": identities[0]["checkpoint_processed_tokens"],
            "same_checkpoint_and_tokenizer": identical_identity,
            "smoke_report_exact": smoke_exact, "validation_report_exact": validation_exact,
            "production_combined_eval_enabled": False,
            "note": "Numerical parity only; speed benefit needs independent accepted-token wall-time measurement."}


def main() -> None:
    p = argparse.ArgumentParser()
    for name in ("legacy-smoke", "combined-smoke", "legacy-validation", "combined-validation", "out"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    files = [getattr(a, name.replace("-", "_")) for name in
             ("legacy-smoke", "combined-smoke", "legacy-validation", "combined-validation")]
    report = compare(*(json.loads(path.read_text(encoding="utf-8-sig")) for path in files))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["parity_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
