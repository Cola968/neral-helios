"""Select the best fully-passing V10 alpha without loading model weights."""
import argparse
import json
from pathlib import Path

from recovery_identity import file_sha

ALPHAS = (0.075, 0.05, 0.025, 0.0125)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", required=True)
    ap.add_argument("--scratch-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    reports = Path(args.reports).resolve()
    scratch = Path(args.scratch_root).resolve()

    rows = []
    for alpha in ALPHAS:
        tag = "v10_a" + str(alpha).replace(".", "p")
        result_path = reports / tag / "result.json"
        if not result_path.exists():
            raise FileNotFoundError(f"missing completed V10 result: {result_path}")
        row = json.loads(result_path.read_text(encoding="utf8"))
        if float(row.get("alpha")) != alpha or row.get("tag") != tag:
            raise ValueError("V10 result identity mismatch")
        ck = scratch / tag / "candidate.pt"
        if not ck.exists() or file_sha(ck) != row.get("checkpoint_sha256"):
            raise ValueError(f"staged checkpoint mismatch for {tag}")
        rows.append(row)

    eligible = [row for row in rows if row.get("pass") is True]
    eligible.sort(key=lambda row: (row["validation_loss"], -row["bilingual_pct"], -row["alpha"]))
    selected = eligible[0] if eligible else None
    result = {
        "schema": 1,
        "records": rows,
        "selection_passed": selected is not None,
        "selected": selected,
        "selected_tag": selected["tag"] if selected else None,
        "selected_checkpoint": str(scratch / selected["tag"] / "candidate.pt") if selected else None,
        "candidate_tokens": 1_008_003_584,
        "next_action": "promote_selected_for_confirmation" if selected else "reasoning_retention_retrain",
        "benchmarks_used_for_training": False,
        "final_audit_pending": True,
    }
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
