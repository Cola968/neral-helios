from __future__ import annotations

"""Fail-closed selector for ARCTUS V6 cross-lingual recovery.

Reject missing, NaN, infinite or out-of-range metrics; these must never
be silently interpreted as a valid non-regression. Pure CPU metadata logic.
"""

import argparse
import json
import math
import re
from pathlib import Path

SUITE = "arctus-bilingual-holdout-v1"
SHA = re.compile(r"[0-9a-f]{64}")
DIRECTIONS = frozenset({"en_to_de", "de_to_en"})


def load(path: str) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf8"))
    if not isinstance(value, dict):
        raise RuntimeError("invalid report object")
    return value


def pct(value):
    if type(value) not in (int, float):
        return None
    number=float(value)
    return number if math.isfinite(number) and 0 <= number <= 100 else None


def loss(value):
    if type(value) not in (int, float):
        return None
    number=float(value)
    return number if math.isfinite(number) and number >= 0 else None


def read_holdout_metrics(report):
    """Validate a full independently evaluated, balanced 400-case holdout.

    The source evaluator produces exactly 200 directional cases each.
    Percentages alone are unauditable: compare them with raw counts so
    neither forged/non-finite scores nor truncated suites can be promoted.
    """
    if not isinstance(report, dict) or type(report.get("cases")) is not int:
        return None
    if report["cases"] != 400:
        return None
    data=report.get("directions")
    if not isinstance(data, dict) or set(data) != DIRECTIONS:
        return None
    overall=pct(report.get("overall_accuracy_pct"))
    if overall is None:
        return None
    result={}
    total_correct=0
    total_cases=0
    for name in sorted(DIRECTIONS):
        row=data[name]
        if not isinstance(row, dict):
            return None
        correct,total=row.get("correct"),row.get("total")
        if (type(correct) is not int or type(total) is not int or
                total != 200 or not 0 <= correct <= total):
            return None
        score=pct(row.get("accuracy_pct"))
        if score is None or abs(score-100.0*correct/total) > 1e-6:
            return None
        result[name]=score
        total_correct+=correct
        total_cases+=total
    if total_cases != report["cases"] or abs(overall-100.0*total_correct/total_cases) > 1e-6:
        return None
    return overall,result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-holdout", required=True)
    ap.add_argument("--candidate", action="append", nargs=4,
                    metavar=("NAME", "GATE", "HOLDOUT", "OUT_DIR"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    baseline = load(args.baseline_holdout)
    digest=baseline.get("cases_sha256")
    if (baseline.get("suite") != SUITE or
            not isinstance(digest, str) or not SHA.fullmatch(digest)):
        raise RuntimeError("invalid baseline suite or case digest")
    baseline_metrics=read_holdout_metrics(baseline)
    if baseline_metrics is None:
        raise RuntimeError("invalid baseline holdout metrics/directions/counts")
    base_overall,base_dirs=baseline_metrics

    rows=[]
    for name, gate_path, holdout_path, out_dir in args.candidate or []:
        gate=load(gate_path)
        holdout=load(holdout_path)
        failures=[]
        if gate.get("selection_allowed") is not True:
            failures.append("immutable_dev_or_validation_gate")
        if (holdout.get("suite") != SUITE or
                holdout.get("cases_sha256") != digest):
            failures.append("holdout_identity_mismatch")

        candidate_metrics=read_holdout_metrics(holdout)
        actual_overall,raw_dirs=(candidate_metrics if candidate_metrics is not None
                                 else (None,None))
        if actual_overall is None or raw_dirs is None:
            failures.append("invalid_holdout_metric")
        overall=actual_overall if actual_overall is not None else -1.0
        dirs=raw_dirs if raw_dirs is not None else {}
        if overall < base_overall:
            failures.append(f"holdout_overall:{overall}<{base_overall}")
        for direction,floor in base_dirs.items():
            value=dirs.get(direction,-1.0)
            if value < floor:
                failures.append(f"{direction}:{value}<{floor}")

        candidates=gate.get("candidates")
        if (not isinstance(candidates,list) or len(candidates)!=1 or
                not isinstance(candidates[0],dict)):
            candidate_info={}
        else:
            candidate_info=candidates[0]
        # The independently evaluated holdout must belong to the exact
        # candidate admitted by the preceding dev/validation gate.
        if (gate.get("selected") != name or
                candidate_info.get("name") != name or
                not isinstance(candidates,list) or len(candidates)!=1 or
                candidate_info.get("pass") is not True):
            failures.append("gate_candidate_identity_mismatch")
        # All accuracy numbers are meaningless if separate checkpoints
        # were measured. The earlier gate already checks dev/validation
        # identity; bind its verified SHA to the independent holdout too.
        gate_ck=candidate_info.get("checkpoint_sha256")
        holdout_ck=holdout.get("checkpoint_sha256")
        if (not isinstance(gate_ck,str) or not SHA.fullmatch(gate_ck) or
                not isinstance(holdout_ck,str) or not SHA.fullmatch(holdout_ck) or
                gate_ck != holdout_ck):
            failures.append("checkpoint_identity_mismatch")
        actual_loss=loss(candidate_info.get("validation_loss"))
        if actual_loss is None:
            failures.append("invalid_validation_loss")
        candidate_loss=actual_loss if actual_loss is not None else 1e9
        dev_overall=pct(candidate_info.get("overall_accuracy_pct"))
        raw_categories=candidate_info.get("categories")
        dev_bilingual=pct(raw_categories.get("bilingual") if isinstance(raw_categories,dict) else None)
        if dev_overall is None or dev_bilingual is None:
            failures.append("invalid_dev_metric")

        rows.append({
            "name":name,
            "pass":not failures,
            "failures":failures,
            "validation_loss":candidate_loss,
            "old_dev_overall_accuracy_pct":dev_overall if dev_overall is not None else -1.0,
            "old_dev_bilingual_accuracy_pct":dev_bilingual if dev_bilingual is not None else -1.0,
            "holdout_overall_accuracy_pct":overall,
            "holdout_directions":dirs,
            "out_dir":out_dir,
        })

    passing=[row for row in rows if row["pass"]]
    # Holdout scores are only a NON-REGRESSION VETO, never a ranking or
    # tie-break signal: optimizing on held-out answers leaks test information.
    passing.sort(key=lambda row:(
        row["validation_loss"],-row["old_dev_overall_accuracy_pct"],row["name"]))
    selected=passing[0]["name"] if passing else None
    result={
        "schema":1,
        "purpose":"development-only V6 cross-lingual recovery selection",
        "baseline_holdout_overall_accuracy_pct":base_overall,
        "baseline_holdout_directions":base_dirs,
        "selected":selected,
        "selection_allowed":selected is not None,
        "candidates":rows,
        "policy":(
            "Fail closed: finite valid scores, immutable holdout digest, "
            "dev/validation gate and both independent language directions "
            "must not regress. Holdout is a veto, never a ranking signal. "
            "Never promote based on NaN/Infinity."
        ),
    }
    Path(args.out).write_text(json.dumps(result,indent=2,ensure_ascii=False,
                                         allow_nan=False),encoding="utf8")
    print(json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)


if __name__ == "__main__":
    main()
