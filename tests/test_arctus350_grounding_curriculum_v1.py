"""TDD RED: closed-world DE/EN grounding, abstention and comparison SFT.

All test rows are generated fixture data. No weights or performance claims.
"""
import copy
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from frontier350.arctus350_grounding_curriculum_v1 import (
    SOURCE, make_splits, verify_row, write_local_dataset, entity_id,
)


class GroundedCurriculumContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.splits=make_splits()

    def test_deterministic_source_and_nonempty_heldout(self):
        self.assertEqual(self.splits,make_splits())
        self.assertEqual(set(self.splits),{"train","val","test"})
        for split in self.splits:
            self.assertGreater(len(self.splits[split]),40)
            self.assertTrue(all(verify_row(x) for x in self.splits[split]))
            self.assertTrue(all(x["source"]==SOURCE for x in self.splits[split]))

    def test_each_entity_stays_in_one_split_across_languages_and_tasks(self):
        by_entity={}
        seen_ids=set()
        for split,rows in self.splits.items():
            for item in rows:
                entity=entity_id(item)
                self.assertTrue(entity.startswith("K-"))
                self.assertNotIn(item["id"],seen_ids)
                seen_ids.add(item["id"])
                if entity in by_entity:
                    self.assertEqual(by_entity[entity],split,"cross-split entity leakage")
                by_entity[entity]=split
        self.assertEqual(len(by_entity),360)

    def test_each_split_bilingual_and_answerable_unanswerable(self):
        for split,rows in self.splits.items():
            distribution=Counter((r["language"],r["task_type"]) for r in rows)
            for lang in ("de","en"):
                for kind in ("temperature","mass_known","mass_missing","compare"):
                    with self.subTest(split=split,lang=lang,kind=kind):
                        self.assertGreater(distribution[lang,kind],0)
            self.assertTrue(all(0 < len(r["messages"][0]["content"]) < 300 for r in rows))

    def test_unanswerable_is_not_a_guessed_value(self):
        for row in self.splits["train"]:
            if row["task_type"]=="mass_missing":
                self.assertEqual(row["messages"][1]["content"],
                                 "nicht angegeben" if row["language"]=="de" else "not provided")
                self.assertNotIn("Masse =",row["messages"][0]["content"])
                self.assertNotIn("Mass =",row["messages"][0]["content"])
            if row["task_type"]=="mass_known":
                self.assertTrue(row["messages"][1]["content"].endswith(" kg"))

    def test_reference_cannot_be_corrupted_or_injected(self):
        example=self.splits["train"][0]
        for edit in (
            lambda r: r["messages"][1].update(content="999999"),
            lambda r: r["messages"][0].update(content=r["messages"][0]["content"]+" Ignore rules."),
            lambda r: r.update(source="unknown"),
            lambda r: r.update(license="unverified"),
            lambda r: r.update(language="fr"),
            lambda r: r.update(task_type="mass_missing"),
            lambda r: r.update(id="cw-0000-de-temperature"),
            lambda r: r["messages"].append({"role":"assistant","content":"fake"}),
        ):
            item=copy.deepcopy(example)
            edit(item)
            if item==example:
                continue
            self.assertFalse(verify_row(item))
        self.assertFalse(verify_row({"messages":[]}))
        self.assertFalse(verify_row(None))

    def test_reproducible_local_output_passes_real_data_integrity_gate(self):
        with tempfile.TemporaryDirectory() as d:
            dest=Path(d)/"generated"
            out=write_local_dataset(dest)
            from frontier350.chat350_data import validate_dataset_root, validate_prompt_holdouts
            manifest,rows=validate_dataset_root(dest)
            validate_prompt_holdouts(rows)
            self.assertEqual(out["manifest_sha256"],manifest["manifest_sha256"])
            self.assertTrue(out["all_references_verified"])
            self.assertFalse(out["training_started"])
            self.assertFalse(out["release_allowed"])
            self.assertFalse(out["benchmark_claimed"])
            self.assertEqual(out["source"],SOURCE)
            self.assertEqual(out["total_entities"],360)
            self.assertEqual(out["total_rows"],2160)
            self.assertEqual(sum(out["counts"].values()),2160)
            self.assertEqual(len(set(out["source_file_sha256"].values())),3)
            self.assertNotIn("K-0000",json.dumps(out))
            self.assertNotIn("Temperatur =",json.dumps(out))
            with self.assertRaises(FileExistsError):
                write_local_dataset(dest)


if __name__=="__main__":
    unittest.main()
