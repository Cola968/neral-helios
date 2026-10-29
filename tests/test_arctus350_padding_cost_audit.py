"""CPU-only regression tests for a length-only Arctus350 SFT padding cost audit."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from frontier350.arctus350_padding_cost_audit import (
    PaddingCostError,
    compare_padding_costs,
    read_lengths_json,
)


class PaddingCostAuditTests(unittest.TestCase):
    def test_uniform_lengths_have_no_artificial_gain(self):
        report = compare_padding_costs([32] * 32, batch_size=4, bucket_size=16, seeds=4)
        self.assertEqual(report["status"], "THEORETICAL_CPU_ONLY")
        self.assertEqual(report["model_quality"], "NOT_MEASURED")
        self.assertEqual(report["training_performed"], False)
        self.assertEqual(report["summary"]["padded_token_reduction_pct"], 0.0)
        self.assertEqual(report["summary"]["attention_proxy_reduction_pct"], 0.0)

    def test_heterogeneous_lengths_show_measurable_padding_cost(self):
        lengths = [4] * 32 + [128] * 32
        report = compare_padding_costs(lengths, batch_size=4, bucket_size=64, seeds=8)
        self.assertEqual(report["example_count"], 64)
        self.assertEqual(report["summary"]["real_tokens"], sum(lengths))
        self.assertGreater(report["summary"]["padded_token_reduction_pct"], 0.0)
        self.assertGreater(report["summary"]["attention_proxy_reduction_pct"], 0.0)
        self.assertGreater(report["summary"]["bucketed_padded_tokens_median"],
                           report["summary"]["compact_padded_tokens_median"])

    def test_same_input_and_seed_is_reproducible(self):
        args = ([7, 12, 3, 45, 15, 8, 27, 60] * 4,)
        a = compare_padding_costs(*args, batch_size=4, bucket_size=16, seeds=5)
        b = compare_padding_costs(*args, batch_size=4, bucket_size=16, seeds=5)
        self.assertEqual(a, b)
        self.assertRegex(a["lengths_sha256"], r"^[0-9a-f]{64}$")

    def test_rejects_invalid_lengths_and_budgets(self):
        invalid = ([True] * 8, [0] * 8, [2.5] * 8, [16385] * 8,
                   [1] * 7, ["2"] * 8, [])
        for lengths in invalid:
            with self.subTest(lengths=lengths):
                with self.assertRaises(PaddingCostError):
                    compare_padding_costs(lengths, batch_size=4, bucket_size=8, seeds=2)
        for params in ({"batch_size": True}, {"bucket_size": 3},
                       {"seeds": 0}, {"seeds": 100}, {"batch_size": 0}):
            with self.subTest(params=params):
                with self.assertRaises(PaddingCostError):
                    compare_padding_costs([4] * 16, batch_size=4,
                                          bucket_size=8, seeds=2, **params)

    def test_strict_input_json_rejects_extras_duplicates_and_nan(self):
        for payload in ('{"lengths":[4,4,4,4],"source":"private"}',
                        '{"lengths":[4,4,4,4],"lengths":[8,8,8,8]}',
                        '{"lengths":[4,4,NaN,4]}', '{"lengths":"private"}',
                        '[]'):
            with self.subTest(payload=payload):
                with self.assertRaises(PaddingCostError):
                    read_lengths_json(payload)
        self.assertEqual(read_lengths_json('{"lengths":[2,3,4,5]}'), [2,3,4,5])

    def test_no_raw_length_list_or_example_ids_exported(self):
        lengths = [3, 11, 29, 47, 2, 13, 31, 53]
        report = compare_padding_costs(lengths, batch_size=2, bucket_size=8, seeds=3)
        self.assertNotIn("lengths", report)
        self.assertNotIn("cases", report)
        self.assertNotIn("example_ids", report)
        self.assertNotIn(json.dumps(lengths), json.dumps(report))
        self.assertIn("NOT_MEASURED", json.dumps(report))

    def test_cli_writes_new_aggregate_report_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            inp, out = root / "lengths.json", root / "report.json"
            inp.write_text(json.dumps({"lengths": [4] * 16 + [40] * 16}))
            command = [sys.executable, "-m", "frontier350.arctus350_padding_cost_audit",
                       "--lengths-json", str(inp), "--out", str(out),
                       "--batch-size", "4", "--bucket-size", "32", "--seeds", "3"]
            first = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(first.returncode, 0, first.stderr)
            report = json.loads(out.read_text())
            self.assertEqual(report["example_count"], 32)
            self.assertEqual(report["status"], "THEORETICAL_CPU_ONLY")
            self.assertNotIn("lengths", report)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertEqual(json.loads(out.read_text()), report)


if __name__ == "__main__":
    unittest.main()
