"""NERAL v6/v7 offline statistical release diagnostics (NEVER grants release).

Inputs are paired *upstream verified* boolean scores, not model responses.
This script cannot validate the original grading oracle, source independence,
pre-registration chronology, model identity, or unseen-test guarantees.
Only aggregate counts leave the local process; raw prompts and answers stay local.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re

DOMAINS = ("math", "code", "reasoning")
EXPECTED_N = 600
DOMAIN_N = 200
Z95 = 1.959963984540054
HEX40 = re.compile(r"[0-9a-f]{40}\Z")
HEX64 = re.compile(r"[0-9a-f]{64}\Z")
ID = re.compile(r"[A-Za-z0-9_-]{1,80}\Z")
FIELDS = {
    "id", "domain", "family",
    "base_correct", "neral_correct",
    "base_strict", "neral_strict",
    "base_format_valid", "neral_format_valid",
}
REG_FIELDS = {
    "schema", "version", "attempt_id", "registered_at_utc",
    "case_sha256", "scorer_sha256", "baseline_prompt_sha256",
    "decoding_sha256", "foundation_revision", "adapter_sha256",
    "grader_contract_sha256", "registration_commit_sha",
    "pre_run_independence_review_sha256",
}


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    """Exact two-sided conditional binomial test, tied pairs excluded."""
    if (type(improved) is not int or type(regressed) is not int or
            improved < 0 or regressed < 0 or improved + regressed > 10000):
        raise ValueError("Invalid discordant counts")
    n = improved + regressed
    if n == 0:
        return 1.0
    # Integer combinatorics avoid asymptotic McNemar approximations.
    numerator = 2 * sum(math.comb(n, k) for k in range(min(improved, regressed) + 1))
    return min(1.0, numerator / (1 << n))


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if not 0 <= successes <= total or total <= 0:
        raise ValueError("Invalid Wilson binomial")
    proportion = successes / total
    z2 = Z95 * Z95
    denom = 1 + z2 / total
    center = (proportion + z2 / (2 * total)) / denom
    radius = Z95 * math.sqrt(
        proportion * (1 - proportion) / total + z2 / (4 * total * total)
    ) / denom
    return max(0., center - radius), min(1., center + radius)


def paired_newcombe_ci(
    improved: int, regressed: int, total: int, *, both_correct: int,
) -> tuple[float, float]:
    """Newcombe (1998) METHOD 10 for paired binary proportions, with corrected phi.

    Table: a=both correct, b=NERAL only, c=baseline only, d=both wrong.
    Unlike a Wilson interval on the discordant proportions alone, this uses
    BOTH marginal success proportions and their corrected correlation.
    """
    if (type(total) is not int or not 1 <= total <= 10000 or
            any(type(z) is not int or z < 0 for z in
                (improved, regressed, both_correct)) or
            improved + regressed + both_correct > total):
        raise ValueError("Invalid paired contingency table")
    a, b, c = both_correct, improved, regressed
    d = total - a - b - c
    p1 = (a + b) / total
    p2 = (a + c) / total
    l1, u1 = _wilson(a + b, total)
    l2, u2 = _wilson(a + c, total)
    num = a * d - b * c
    denominator = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    if denominator == 0:
        phi = 0.0
    else:
        # Newcombe's half-observation continuity correction for positive phi.
        phi = (max(0, num - total / 2) if num > 0 else num) / denominator
    diff = p1 - p2
    lower = diff - math.sqrt(max(0.0,
        (p1 - l1)**2 - 2*phi*(p1 - l1)*(u2 - p2) + (u2 - p2)**2))
    upper = diff + math.sqrt(max(0.0,
        (u1 - p1)**2 - 2*phi*(u1 - p1)*(p2 - l2) + (p2 - l2)**2))
    return max(-1.0, lower), min(1.0, upper)


def _check_rows(rows: list[dict]) -> dict[str, list[dict]]:
    if not isinstance(rows, list) or len(rows) != EXPECTED_N:
        raise ValueError("Release gate requires exactly 600 paired tasks")
    used = set()
    groups = {d: [] for d in DOMAINS}
    for row in rows:
        if not isinstance(row, dict) or set(row) != FIELDS:
            raise ValueError("Unexpected score fields")
        ident, domain, family = row["id"], row["domain"], row["family"]
        if (not isinstance(ident, str) or not ID.fullmatch(ident) or ident in used or
                domain not in DOMAINS or not isinstance(family, str) or
                not ID.fullmatch(family)):
            raise ValueError("Invalid task identity, domain or family")
        used.add(ident)
        for field in FIELDS - {"id", "domain", "family"}:
            if type(row[field]) is not bool:
                raise ValueError("Boolean verified scoring fields required")
        for model in ("base", "neral"):
            if row[model + "_strict"] and (not row[model + "_correct"]
                                           or not row[model + "_format_valid"]):
                raise ValueError("Strict correctness cannot exceed verified correctness")
        groups[domain].append(row)
    for domain, group in groups.items():
        if len(group) != DOMAIN_N:
            raise ValueError("Exactly 200 tasks needed in each domain")
        if len({row["family"] for row in group}) < 5:
            raise ValueError("At least five independent method families per domain")
    return groups


def _percentile(sorted_numbers: list[float], q: float) -> float:
    x = q * (len(sorted_numbers) - 1)
    low, high = math.floor(x), math.ceil(x)
    return sorted_numbers[low] * (1 - (x - low)) + sorted_numbers[high] * (x - low)


def stratified_cluster_bootstrap_ci(
    rows: list[dict], *, seed: int = 20261107, draws: int = 2000,
) -> tuple[float, float]:
    """Domain-stratified FAMILY-cluster paired bootstrap (diagnostic).

    Resamples whole task families within each domain, keeps domain weights fixed.
    Family clusters can be correlated internally; do not resample model scores
    independently. At five families/domain uncertainty is still approximate.
    """
    groups = _check_rows(rows)
    if (type(seed) is not int or type(draws) is not int or
            not 1000 <= draws <= 20000):
        raise ValueError("Invalid bootstrap settings")
    family_groups = []
    for domain in DOMAINS:
        mapping = defaultdict(list)
        for row in groups[domain]:
            mapping[row["family"]].append(int(row["neral_correct"]) -
                                           int(row["base_correct"]))
        # sorted ensures generator seed independent of JSONL row order.
        family_groups.append([mapping[k] for k in sorted(mapping)])
    rng = random.Random(seed)
    differences = []
    for _ in range(draws):
        domain_gap = []
        for clusters in family_groups:
            total_diff = 0
            total_rows = 0
            for _ in range(len(clusters)):
                sampled = clusters[rng.randrange(len(clusters))]
                total_diff += sum(sampled)
                total_rows += len(sampled)
            domain_gap.append(total_diff / total_rows)
        differences.append(sum(domain_gap) / len(DOMAINS))
    differences.sort()
    return _percentile(differences, 0.025), _percentile(differences, 0.975)


def _domain_report(group: list[dict]) -> dict:
    n = len(group)
    better = sum(not r["base_correct"] and r["neral_correct"] for r in group)
    worse = sum(r["base_correct"] and not r["neral_correct"] for r in group)
    both = sum(r["base_correct"] and r["neral_correct"] for r in group)
    lower, upper = paired_newcombe_ci(better, worse, n, both_correct=both)
    return {
        "total": n,
        "base_correct": sum(r["base_correct"] for r in group),
        "neral_correct": sum(r["neral_correct"] for r in group),
        "base_strict_correct": sum(r["base_strict"] for r in group),
        "neral_strict_correct": sum(r["neral_strict"] for r in group),
        "base_format_valid": sum(r["base_format_valid"] for r in group),
        "neral_format_valid": sum(r["neral_format_valid"] for r in group),
        "families": len({r["family"] for r in group}),
        "gap_pp": round(100 * (better - worse) / n, 4),
        "newcombe_95_ci_pp": [round(100 * lower, 4), round(100 * upper, 4)],
    }


def analyze_trial(
    rows: list[dict], version: str, *,
    seed: int = 20261107, draws: int = 2000,
) -> dict:
    """Pure statistics only; caller must separately verify grading/provenance."""
    if version not in ("v6", "v7"):
        raise ValueError("Only v6/v7 gates are defined")
    groups = _check_rows(rows)
    better = sum(not r["base_correct"] and r["neral_correct"] for r in rows)
    worse = sum(r["base_correct"] and not r["neral_correct"] for r in rows)
    pvalue = exact_mcnemar_p(better, worse)
    both = sum(r["base_correct"] and r["neral_correct"] for r in rows)
    lower, upper = paired_newcombe_ci(better, worse, len(rows), both_correct=both)
    bootstrap_lower, bootstrap_upper = stratified_cluster_bootstrap_ci(
        rows, seed=seed, draws=draws,
    )
    domains = {domain: _domain_report(group) for domain, group in groups.items()}
    gain = 100 * (better - worse) / len(rows)
    required_gap = 8.0 if version == "v6" else 15.0
    tolerated_drop = -3.0 if version == "v6" else -2.0
    tolerated_ci = -10.0 if version == "v6" else -7.0
    improved_domain_count = sum(stats["gap_pp"] > 0 for stats in domains.values())
    checks = {
        "gap_target_met": gain + 1e-10 >= required_gap,
        "exact_mcnemar_p_le_0_01": pvalue <= 0.01,
        "paired_newcombe_95_lower_positive": lower > 0,
        "family_cluster_bootstrap_lower_positive": bootstrap_lower > 0,
        "domain_improvement_count_met": improved_domain_count >=
            (2 if version == "v6" else 3),
        "no_major_domain_point_regression": all(
            stats["gap_pp"] >= tolerated_drop for stats in domains.values()
        ),
        "domain_uncertainty_bound_met": all(
            stats["newcombe_95_ci_pp"][0] > tolerated_ci for stats in domains.values()
        ),
    }
    return {
        "schema": 1, "status": "NERAL_RELEASE_STATISTICS_ONLY",
        "version": version, "total": len(rows),
        "paired": {
            "improvements": better,
            "regressions": worse,
            "both_correct": sum(r["base_correct"] and r["neral_correct"] for r in rows),
            "both_incorrect": sum(not r["base_correct"] and not r["neral_correct"] for r in rows),
            "gap_pp": round(gain, 4),
            "exact_mcnemar_p": pvalue,
            "paired_newcombe_95_ci_pp": [round(100 * lower, 4), round(100 * upper, 4)],
            "domain_cluster_bootstrap_95_ci_pp": [
                round(100 * bootstrap_lower, 4), round(100 * bootstrap_upper, 4)],
            "cluster_bootstrap_seed": seed,
            "cluster_bootstrap_draws": draws,
        },
        "by_domain": domains,
        "gate_checks": checks,
        "statistical_gate_passed": all(checks.values()),
        "requires_second_fresh_suite": version == "v7",
        "external_independence_verified": False,
        "grader_external_verified": False,
        "release_validated": False,
        "weights_promoted": False,
        "limitations": (
            "Statistical evidence from provided boolean scores only; this script "
            "cannot verify grader correctness, suite freshness, preregistration "
            "chronology, training contamination or tool/decoder parity. "
            "The family bootstrap is diagnostic, not a replacement for independence review."
        ),
    }


def _unique_json(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicated JSON key")
        result[key] = value
    return result


def _load_file(path: Path, maximum: int) -> bytes:
    path = Path(path)
    if (path.is_symlink() or not path.is_file() or
            not 0 < path.stat().st_size <= maximum):
        raise ValueError("Untrusted, missing or oversized local input")
    return path.read_bytes()


def _load_jsonl(raw: bytes) -> list[dict]:
    try:
        lines = raw.decode("utf-8").splitlines()
        if len(lines) != EXPECTED_N or any(not line.strip() for line in lines):
            raise ValueError("Invalid task count")
        rows = [json.loads(line, object_pairs_hook=_unique_json) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid JSONL") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError("Expected JSON objects")
    return rows


def analyze_files(cases: Path, scores: Path, registration: Path, output: Path,
                  *, seed: int = 20261107, draws: int = 2000) -> dict:
    """Local SHA-pinned scoring; refuses overwrites and never exports raw rows."""
    cases, scores, registration, output = map(Path, (cases, scores, registration, output))
    if (output.exists() or output.is_symlink() or not output.parent.is_dir()
            or output.parent.is_symlink() or
            output.resolve() in {cases.resolve(), scores.resolve(), registration.resolve()}):
        raise ValueError("New output file in existing folder required")
    case_bytes = _load_file(cases, 3_000_000)
    score_bytes = _load_file(scores, 3_000_000)
    raw_registration = _load_file(registration, 100_000)
    try:
        reg = json.loads(raw_registration.decode("utf-8"), object_pairs_hook=_unique_json)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Invalid registration JSON") from exc
    if (not isinstance(reg, dict) or set(reg) != REG_FIELDS or
            type(reg["schema"]) is not int or reg["schema"] != 1 or
            reg["version"] not in ("v6", "v7") or
            not isinstance(reg["attempt_id"], str) or not ID.fullmatch(reg["attempt_id"]) or
            not isinstance(reg["registered_at_utc"], str) or
            not re.fullmatch(r"20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", reg["registered_at_utc"]) or
            not all(isinstance(reg[k], str) and HEX64.fullmatch(reg[k])
                    for k in ("case_sha256", "scorer_sha256",
                              "baseline_prompt_sha256", "decoding_sha256",
                              "adapter_sha256", "grader_contract_sha256",
                              "pre_run_independence_review_sha256")) or
            not all(isinstance(reg[k], str) and HEX40.fullmatch(reg[k])
                    for k in ("foundation_revision", "registration_commit_sha"))):
        raise ValueError("Invalid pre-registered release contract")
    if hashlib.sha256(case_bytes).hexdigest() != reg["case_sha256"]:
        raise ValueError("Case SHA differs from pre-registration")
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != reg["scorer_sha256"]:
        raise ValueError("Scorer script changed after pre-registration")
    cases_rows = _load_jsonl(case_bytes)
    score_rows = _load_jsonl(score_bytes)
    if (len({r.get("id") for r in cases_rows}) != EXPECTED_N or
            {r["id"] for r in cases_rows} != {r["id"] for r in score_rows}):
        raise ValueError("Paired scoring IDs do not match registered cases")
    case_meta = {r["id"]: (r["domain"], r["family"]) for r in cases_rows}
    if any((row["domain"], row["family"]) != case_meta[row["id"]] for row in score_rows):
        raise ValueError("Paired scoring family/domain differs from registered suite")
    report = analyze_trial(score_rows, reg["version"], seed=seed, draws=draws)
    report["preregistration"] = {
        "attempt_id": reg["attempt_id"],
        "registration_commit_sha": reg["registration_commit_sha"],
        "case_sha256": reg["case_sha256"],
        "score_sha256": hashlib.sha256(score_bytes).hexdigest(),
        "scorer_sha256": reg["scorer_sha256"],
        "foundation_revision": reg["foundation_revision"],
        "adapter_sha256": reg["adapter_sha256"],
        "chronology_and_unique_suite_not_verified_automatically": True,
    }
    # This is a diagnostic, not an authorization to ship a checkpoint.
    from scripts.neral_ext import write_report
    write_report(output, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--registration", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20261107)
    parser.add_argument("--draws", type=int, default=2000)
    opts = parser.parse_args(argv)
    result = analyze_files(opts.cases, opts.scores, opts.registration,
                           opts.out, seed=opts.seed, draws=opts.draws)
    print(json.dumps({
        "status": result["status"], "statistical_gate_passed":
            result["statistical_gate_passed"],
        "release_validated": False, "paired": result["paired"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
