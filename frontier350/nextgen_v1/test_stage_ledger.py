"""CPU-only adversarial contract tests: synthetic fixtures do not prove training."""
from __future__ import annotations

import hashlib
import json
import unittest

from stage_ledger import audit_ledger
from test_admission import fixture


def synthetic_events():
    checkpoint = "e" * 64
    chain = "a" * 64
    events = []
    for index, next_checkpoint in enumerate(("2" * 64, "3" * 64, "1" * 64)):
        next_chain = str(index + 2) * 64
        events.append({
            "schema": 1,
            "stage_index": index,
            "token_start": index * 5_000_000,
            "token_end": (index + 1) * 5_000_000,
            "monotonic_start_ns": index * 80_000_000_000,
            "monotonic_end_ns": (index + 1) * 80_000_000_000,
            "checkpoint_start_sha256": checkpoint,
            "checkpoint_end_sha256": next_checkpoint,
            "input_chain_start_sha256": chain,
            "input_chain_end_sha256": next_chain,
            "rejected_tokens": 0,
            "health_pass": True,
            "includes_eval_checkpoint_recovery": True,
        })
        checkpoint, chain = next_checkpoint, next_chain
    return events


def encode(events):
    raw = ("".join(json.dumps(row, separators=(",", ":")) + "\n" for row in events)).encode("utf-8")
    return raw, hashlib.sha256(raw).hexdigest()


def run(events, report=None):
    raw, digest = encode(events)
    return audit_ledger(fixture() if report is None else report, raw, digest)


class StageLedgerTests(unittest.TestCase):
    def test_consistent_synthetic_never_proves_training(self):
        result = run(synthetic_events())
        self.assertTrue(result["ledger_consistent"])
        self.assertTrue(result["contract_passed"])
        self.assertEqual(result["accepted_tokens"], 15_000_000)
        self.assertFalse(result["independent_training_verified"])
        self.assertFalse(result["eligible_for_production"])

    def test_external_digest_mismatch(self):
        raw, _ = encode(synthetic_events())
        with self.assertRaisesRegex(ValueError, "pinned SHA"):
            audit_ledger(fixture(), raw, "f" * 64)

    def test_missing_or_reordered_stages(self):
        with self.assertRaises(ValueError):
            run(synthetic_events()[:2])
        rows = synthetic_events()
        rows[1], rows[2] = rows[2], rows[1]
        with self.assertRaises(ValueError):
            run(rows)

    def test_gapped_or_overlapping_token_counters(self):
        for offset in (1, -1):
            rows = synthetic_events()
            rows[1]["token_start"] += offset
            with self.assertRaisesRegex(ValueError, "continuity"):
                run(rows)

    def test_gapped_or_overlapping_monotonic_clock(self):
        for offset in (1, -1):
            rows = synthetic_events()
            rows[1]["monotonic_start_ns"] += offset
            with self.assertRaisesRegex(ValueError, "continuity"):
                run(rows)

    def test_checkpoint_and_data_chain_must_link(self):
        for field in ("checkpoint_start_sha256", "input_chain_start_sha256"):
            rows = synthetic_events()
            rows[1][field] = "f" * 64
            with self.assertRaisesRegex(ValueError, "continuity"):
                run(rows)

    def test_replayed_checkpoint_or_unadvanced_chain(self):
        rows = synthetic_events()
        rows[1]["checkpoint_end_sha256"] = rows[0]["checkpoint_end_sha256"]
        with self.assertRaises(ValueError):
            run(rows)
        rows = synthetic_events()
        rows[0]["input_chain_end_sha256"] = rows[0]["input_chain_start_sha256"]
        with self.assertRaises(ValueError):
            run(rows)

    def test_first_and_last_checkpoint_must_match_model_reports(self):
        rows = synthetic_events()
        rows[0]["checkpoint_start_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "baseline"):
            run(rows)
        rows = synthetic_events()
        rows[-1]["checkpoint_end_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "candidate"):
            run(rows)

    def test_counter_or_speed_inflation_is_rejected(self):
        report = fixture()
        report["stages"][0]["accepted_tokens"] += 1_000_000
        with self.assertRaisesRegex(ValueError, "counters"):
            run(synthetic_events(), report)
        report = fixture()
        report["stages"][0]["wall_seconds"] = 1.0
        with self.assertRaisesRegex(ValueError, "wall time"):
            run(synthetic_events(), report)

    def test_report_flags_must_match_and_sha_format_is_strict(self):
        report = fixture()
        report["stages"][0]["health_pass"] = False
        with self.assertRaisesRegex(ValueError, "health_pass"):
            run(synthetic_events(), report)
        raw, _ = encode(synthetic_events())
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            audit_ledger(fixture(), raw, "z" * 64)

    def test_good_ledger_does_not_override_quality_regression(self):
        report = fixture()
        report["candidate"]["dev_categories"]["math"] = 0
        result = run(synthetic_events(), report)
        self.assertTrue(result["ledger_consistent"])
        self.assertFalse(result["contract_passed"])
        self.assertIn("regression_dev_math", result["contract_failures"])

    def test_canonical_jsonl_required(self):
        raw, digest = encode(synthetic_events())
        with self.assertRaisesRegex(ValueError, "LF-terminated"):
            audit_ledger(fixture(), raw.rstrip(b"\n"), hashlib.sha256(raw.rstrip(b"\n")).hexdigest())
        duplicate_key = raw.replace(b'"schema":1,', b'"schema":1,"schema":1,', 1)
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            audit_ledger(fixture(), duplicate_key, hashlib.sha256(duplicate_key).hexdigest())


if __name__ == "__main__":
    unittest.main()
