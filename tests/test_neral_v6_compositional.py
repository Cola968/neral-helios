"""CPU-only v6 cross-template candidate: deterministic labels, no test leakage."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_v6_compositional import FAMILIES, ROWS, make_case, generate, build


class V6CandidateTests(unittest.TestCase):
    def test_800_examples_reproducible_and_distinct(self):
        one, meta = generate()
        two, meta2 = generate()
        self.assertEqual(one, two)
        self.assertEqual(meta, meta2)
        self.assertEqual(meta["counts"], {"train": 600, "validation": 100, "holdout": 100})
        self.assertEqual(len(FAMILIES), 5)
        assert not meta["training_performed"]
        assert not meta["training_allowed"]
        assert not meta["human_review_approved"]
        assert not meta["independent_external_benchmark"]
        prompts = set()
        for split, payload in one.items():
            self.assertEqual(hashlib.sha256(payload).hexdigest(), meta["sha256"][split])
            rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
            self.assertEqual(len(rows), ROWS[split])
            self.assertEqual(set(x["family"] for x in rows), set(FAMILIES))
            for row in rows:
                roles = [m["role"] for m in row["messages"]]
                self.assertEqual(roles, ["system", "user", "assistant"])
                self.assertEqual(len(row["oracle_sha256"]), 64)
                prompt = row["messages"][1]["content"].casefold()
                self.assertNotIn(prompt, prompts)
                prompts.add(prompt)
                self.assertTrue(row["messages"][2]["content"].strip())
        self.assertEqual(len(prompts), 800)

    def test_holdout_different_surface_templates_and_seeds(self):
        for family in FAMILIES:
            train = make_case(family, "train", 0)["messages"][1]["content"]
            validation = make_case(family, "validation", 0)["messages"][1]["content"]
            holdout = make_case(family, "holdout", 0)["messages"][1]["content"]
            self.assertNotEqual(train, validation)
            self.assertNotEqual(train, holdout)
            self.assertNotEqual(validation, holdout)

    def test_create_only_and_explicit_unapproved_training_gate(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "new"
            report = build(out)
            self.assertEqual({p.name for p in out.iterdir()},
                             {"train.jsonl", "validation.jsonl",
                              "holdout.jsonl", "manifest.json"})
            self.assertFalse(report["training_allowed"])
            self.assertTrue(report["holdout_never_for_training"])
            with self.assertRaises(ValueError):
                build(out)

    def test_bad_coordinates_rejected(self):
        for args in (("unknown", "train", 0), (FAMILIES[0], "nope", 0),
                     (FAMILIES[0], "train", -1), (FAMILIES[0], "train", 120)):
            with self.assertRaises(ValueError):
                make_case(*args)


if __name__ == "__main__":
    unittest.main()
