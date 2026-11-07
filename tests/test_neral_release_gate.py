"""CPU-only release statistics. RED before implementation."""
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from scripts.neral_release_gate import (
    exact_mcnemar_p, paired_newcombe_ci, stratified_cluster_bootstrap_ci,
    analyze_trial, analyze_files,
)

DOMAINS = ("math", "code", "reasoning")


def samples(total=600, wins_per_domain=40):
    rows = []
    for domain in DOMAINS:
        for index in range(total // 3):
            win = index < wins_per_domain
            rows.append({
                "id": f"{domain}-{index:03d}", "domain": domain,
                "family": f"{domain}-method-{index % 10}",
                "base_correct": False, "neral_correct": win,
                "base_strict": False, "neral_strict": win,
                "base_format_valid": True, "neral_format_valid": True,
            })
    return rows


class ReleaseGateContract(unittest.TestCase):
    def test_exact_two_sided_mcnemar_ties_and_large_discordance(self):
        self.assertEqual(exact_mcnemar_p(0, 0), 1.0)
        self.assertEqual(exact_mcnemar_p(5, 0), 0.0625)
        self.assertAlmostEqual(exact_mcnemar_p(116, 84), 0.028125407232926402, places=14)
        self.assertLess(exact_mcnemar_p(40, 8), 0.0001)
        for invalid in ((-1, 0), (1.0, 0), (True, 0)):
            with self.assertRaises(ValueError):
                exact_mcnemar_p(*invalid)

    def test_newcombe_is_paired_and_has_correct_direction(self):
        lower, upper = paired_newcombe_ci(60, 0, 600)
        self.assertGreater(lower, 0)
        self.assertLess(lower, upper)
        low2, high2 = paired_newcombe_ci(0, 60, 600)
        self.assertAlmostEqual(lower, -high2)
        self.assertAlmostEqual(upper, -low2)
        tie_lower, tie_upper = paired_newcombe_ci(0, 0, 600)
        self.assertLessEqual(tie_lower, 0)
        self.assertGreaterEqual(tie_upper, 0)
        with self.assertRaises(ValueError):
            paired_newcombe_ci(20, 20, 30)

    def test_stratified_cluster_bootstrap_deterministic_and_family_aware(self):
        rows = samples(600, 35)
        a = stratified_cluster_bootstrap_ci(rows, seed=20261107, draws=1000)
        b = stratified_cluster_bootstrap_ci(rows, seed=20261107, draws=1000)
        self.assertEqual(a, b)
        self.assertEqual(a, (a[0], a[1]))
        self.assertLessEqual(a[0], 35/200)
        self.assertGreaterEqual(a[1], 35/200)
        broken = rows[:]
        broken = [dict(r, family="same") if r["domain"] == "math" else r for r in broken]
        with self.assertRaises(ValueError):
            stratified_cluster_bootstrap_ci(broken, seed=7, draws=1000)

    def test_version_gate_and_explicit_not_release_approved(self):
        report = analyze_trial(samples(), "v6", seed=1234, draws=1000)
        self.assertTrue(report["statistical_gate_passed"])
        self.assertFalse(report["release_validated"])
        self.assertFalse(report["external_independence_verified"])
        self.assertEqual(report["total"], 600)
        self.assertEqual(report["paired"]["improvements"], 120)
        self.assertEqual(report["paired"]["regressions"], 0)
        self.assertAlmostEqual(report["paired"]["gap_pp"], 20.0)
        self.assertEqual(set(report["by_domain"]), set(DOMAINS))
        self.assertEqual(report["by_domain"]["math"]["neral_strict_correct"], 40)
        self.assertEqual(report["by_domain"]["math"]["base_format_valid"], 200)
        report_v7 = analyze_trial(samples(), "v7", seed=1234, draws=1000)
        self.assertTrue(report_v7["statistical_gate_passed"])
        self.assertTrue(report_v7["requires_second_fresh_suite"])
        rows = samples(wins_per_domain=10)
        self.assertFalse(analyze_trial(rows, "v6", draws=1000)["statistical_gate_passed"])

    def test_missing_domain_repeated_id_wrong_boolean_and_no_families_rejected(self):
        rows = samples()
        for altered in (
            rows[:-1],
            [rows[0], *rows],
            [dict(rows[0], neral_correct=1), *rows[1:]],
            [dict(rows[0], base_strict=True), *rows[1:]],
            [dict(rows[0], family="math-method-0"), *rows[1:]],
        ):
            if len(altered) == len(rows) and altered[0]["family"] == rows[0]["family"]:
                # Ignore one identical row in family test case.
                continue
            with self.assertRaises(ValueError):
                analyze_trial(altered, "v6", draws=1000)
        with self.assertRaises(ValueError):
            analyze_trial(rows, "v8", draws=1000)

    def test_preregistered_file_hashes_are_verified_and_raw_never_exported(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            cases = root / "cases.jsonl"
            scores = root / "scores.jsonl"
            config = root / "registration.json"
            output = root / "result.json"
            rows = samples()
            case_rows = [{"id": r["id"], "domain": r["domain"], "family": r["family"],
                          "prompt": "PRIVATE UNKNOWN TASK", "answer": "PRIVATE LABEL"}
                         for r in rows]
            cases.write_text("".join(json.dumps(r) + "\n" for r in case_rows), encoding="utf8")
            scores.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
            script = Path(__file__).resolve().parents[1] / "scripts" / "neral_release_gate.py"
            reg = {
                "schema": 1, "version": "v6", "attempt_id": "v6-release-01",
                "registered_at_utc": "2026-11-07T17:00:00Z",
                "case_sha256": hashlib.sha256(cases.read_bytes()).hexdigest(),
                "scorer_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
                "baseline_prompt_sha256": "a"*64,
                "decoding_sha256": "b"*64,
                "foundation_revision": "c"*40,
                "adapter_sha256": "d"*64,
                "grader_contract_sha256": "e"*64,
                "registration_commit_sha": "f"*40,
                "pre_run_independence_review_sha256": "0"*64,
            }
            config.write_text(json.dumps(reg), encoding="utf8")
            report = analyze_files(cases, scores, config, output, draws=1000)
            self.assertTrue(report["statistical_gate_passed"])
            self.assertTrue(output.exists())
            self.assertFalse(report["release_validated"])
            exported = output.read_text(encoding="utf8")
            self.assertNotIn("PRIVATE", exported)
            self.assertNotIn("math-000", exported)
            with self.assertRaises(ValueError):
                analyze_files(cases, scores, config, output, draws=1000)
            bad = dict(reg, case_sha256="1"*64)
            config.write_text(json.dumps(bad), encoding="utf8")
            with self.assertRaises(ValueError):
                analyze_files(cases, scores, config, root/"other.json", draws=1000)


if __name__ == "__main__":
    unittest.main()
