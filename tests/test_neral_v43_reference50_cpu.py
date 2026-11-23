"""CPU-only benchmark protocol: 50 fresh, matched, locally sealed cases."""
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from scripts.neral_v43_reference50 import generate_cases, seal_cases, verify_seal, aggregate


class Reference50ContractCPU(unittest.TestCase):
    SEED="2f"*32

    def test_four_categories_and_exact_50_unique_cases(self):
        rows=generate_cases(self.SEED)
        self.assertEqual(len(rows),50)
        self.assertEqual(dict(Counter(x["category"] for x in rows)),
                         {"everyday":12,"knowledge":12,"math_logic":14,"programming":12})
        self.assertEqual(len({x["id"] for x in rows}),50)
        self.assertEqual(len({x["prompt"] for x in rows}),50)
        self.assertTrue(all(x["history"]==[] for x in rows))

    def test_paired_modes_use_same_hash_locked_prompts(self):
        self.assertEqual(generate_cases(self.SEED),generate_cases(self.SEED))
        self.assertNotEqual(generate_cases(self.SEED),generate_cases("3f"*32))
        with tempfile.TemporaryDirectory() as d:
            suite=Path(d)/"fresh50.jsonl"; manifest=Path(d)/"manifest.json"
            seal_cases(generate_cases(self.SEED),suite,manifest)
            checked=verify_seal(suite,manifest)
            self.assertEqual(len(checked),50)
            with self.assertRaises(FileExistsError):
                seal_cases(generate_cases(self.SEED),suite,manifest)
            original=suite.read_bytes()
            suite.write_bytes(original+b"\n")
            with self.assertRaises(ValueError):
                verify_seal(suite,manifest)

    def test_aggregate_rejects_missing_or_duplicate_and_ungrounded_scores(self):
        cases=generate_cases(self.SEED)
        rows=[{"id":c["id"],"engine":"raw_qwen35_9b","grade":"pass",
               "latency_ms":100,"vram_total_peak_mib":8500} for c in cases]
        rows += [{"id":c["id"],"engine":"neral10_obsidian","grade":"pass",
                  "latency_ms":120,"vram_total_peak_mib":8900} for c in cases]
        stats=aggregate(cases,rows)
        self.assertEqual(stats["observations"],100)
        self.assertEqual(stats["pairs"],50)
        with self.assertRaises(ValueError): aggregate(cases,rows[:-1])
        with self.assertRaises(ValueError): aggregate(cases,rows+[rows[0]])

    def test_reject_weak_seed(self):
        with self.assertRaises(ValueError): generate_cases("1234")


if __name__=="__main__":
    unittest.main()
