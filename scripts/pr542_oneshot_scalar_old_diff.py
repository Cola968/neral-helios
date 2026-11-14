"""Scalar-only comparison of OLD PR542 local raw model ASTs with source clauses.

Does not open prospective 100. Never exports raw model JSON, generated text,
file paths, counterexample assignments or prompt content to CI artifacts.
"""
from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path

from neral_nl_schema_bridge import _parse_candidate, _signature, _source_core_witnesses
from neral_nl_verified_route import strict_source_certificate
from neral_typed_constraints_v2 import _type_predicate, _predicate_value
from scripts.neral_nl_schema_probe import old_case_question

_OLD = frozenset(("21453", "logic-02", "logic-06"))
_OLD_ROOT = "NERAL-PR542-GPU-RAW"
_CLAUSE_LABELS = {
    "21453": ("all_different", "nonadjacent_before", "symmetric_between",
              "immediate_right", "not_edge_left", "not_edge_right"),
    "logic-02": (), "logic-06": (),
}


def scalar_diff(case: str, raw: str) -> dict:
    """Report which source constraints a candidate fails to imply.

    NOTE: A witness is a necessary condition, not proof a model literally
    omitted that clause. The same output count can hide different errors.
    """
    if case not in _OLD or type(raw) is not str:
        raise ValueError("Only previously published development cases")
    source = strict_source_certificate(old_case_question(case))
    if source is None:
        return {"status": "canonical_unavailable"}
    try:
        candidate = _parse_candidate(raw)
    except (ValueError, TypeError):
        return {"status": "invalid_candidate"}
    if _signature(candidate) != _signature(source):
        return {"status": "signature_mismatch"}

    names = tuple(source["variables"])
    domains = [source["variables"][n]["domain"] for n in names]
    types = {name: source["variables"][name]["type"] for name in names}
    compile_rules = lambda rules: [
        _type_predicate(rule, types, [0], 0) for rule in rules
    ]
    source_rules = compile_rules(source["constraints"])
    candidate_rules = compile_rules(candidate["constraints"])

    candidates = 0
    sources = 0
    false_positive = 0
    false_negative = 0
    violated = [0 for _ in source_rules]
    for values in product(*domains):
        assignment = dict(zip(names, values))
        hits = [_predicate_value(rule, assignment) for rule in source_rules]
        original_ok = all(hits)
        candidate_ok = all(_predicate_value(rule, assignment)
                           for rule in candidate_rules)
        sources += int(original_ok)
        candidates += int(candidate_ok)
        false_positive += int(candidate_ok and not original_ok)
        false_negative += int(original_ok and not candidate_ok)
        if candidate_ok:
            for i, hit in enumerate(hits):
                violated[i] += int(not hit)
    labels = _CLAUSE_LABELS[case]
    if not labels:
        labels = tuple("source_clause_" + str(i + 1)
                       for i in range(len(source_rules)))
    if len(labels) != len(violated):
        return {"status": "internal_source_label_mismatch"}
    return {
        "status": "compared",
        "source_count": sources,
        "candidate_count": candidates,
        "candidate_not_satisfying_full_source": false_positive,
        "source_solutions_omitted_by_candidate": false_negative,
        "source_clauses_violated_by_candidate": {
            key: n for key, n in zip(labels, violated) if n > 0
        },
        "first_coverage_rejection": _source_core_witnesses(
            old_case_question(case), candidate),
    }


def previous_run_diff(run_id: int) -> dict:
    if type(run_id) is not int or not 1 <= run_id <= 99999999999:
        raise ValueError("Invalid old run id")
    root = Path(os.path.expanduser("~")) / _OLD_ROOT / ("run-" + str(run_id))
    out = {"old_run_id": run_id, "cases": {}}
    for case in ("21453", "logic-02", "logic-06"):
        path = root / ("raw_models_" + case + ".json")
        if not path.is_file():
            out["cases"][case] = {"status": "local_raw_not_found"}
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        model_results = {}
        for model in ("qlora", "base"):
            row = (data.get("models", {}).get(model, {})
                   if type(data) is dict else {})
            raw = row.get("raw_json") if type(row) is dict else None
            model_results[model] = (scalar_diff(case, raw)
                                    if type(raw) is str
                                    else {"status": "no_complete_raw_json"})
        out["cases"][case] = model_results
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--previous-run", type=int, required=True)
    args = p.parse_args()
    # No paths or prompts included in printed scalar output.
    print("PR542_OLD_RAW_CLAUSE_DIFF=" +
          json.dumps(previous_run_diff(args.previous_run), separators=(",", ":"),
                     sort_keys=True))
