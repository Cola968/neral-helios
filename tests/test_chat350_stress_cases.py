from __future__ import annotations
import collections
import json
import tempfile
import unittest
from pathlib import Path
from frontier350.build_chat350_stress_cases import generate_cases, write_cases


class ProceduralStressTests(unittest.TestCase):
    def test_default_count_distribution(self):
        cases = generate_cases()
        self.assertEqual(len(cases), 56)
        self.assertEqual(collections.Counter(c["category"] for c in cases),
                         {"math":16,"instruction_exact":12,"context_memory":12,"code":8,"translation":8})

    def test_seed_changes_answers_and_maintains_ids(self):
        a=generate_cases(30); b=generate_cases(31)
        self.assertEqual([x["id"] for x in a],[x["id"] for x in b])
        self.assertNotEqual([x["prefix"] for x in a],[x["prefix"] for x in b])

    def test_reproducible_and_supported_scoring(self):
        a=generate_cases(320); b=generate_cases(320)
        self.assertEqual(a,b)
        self.assertEqual(len({x["prefix"] for x in a}),len(a))
        self.assertTrue(all(x["scoring"]["type"] in {"exact","integer","contains_all","contains_any"} for x in a))

    def test_write_fails_closed_on_existing(self):
        with tempfile.TemporaryDirectory() as td:
            dest=Path(td)/"probe.json"
            meta=write_cases(444,dest)
            self.assertEqual(meta["cases"],56)
            self.assertEqual(len(json.loads(dest.read_text(encoding="utf8"))),56)
            with self.assertRaises(FileExistsError):write_cases(444,dest)


if __name__=="__main__":
    unittest.main()
