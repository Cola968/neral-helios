"""Independent CPU-only NERAL instruction curriculum; no Qwen, GPU or training."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from frontier350.arctus350_instruction_curriculum_v1 import (
    SOURCE, SCHEMA, make_splits, verify_row, write_local_dataset,
)
from frontier350.arctus350_teacher_gate import _independent_task


class NativeInstructionCurriculumTests(unittest.TestCase):
    def test_reproducible_disjoint_balanced_bilingual_compositional_splits(self):
        first = make_splits()
        self.assertEqual(first, make_splits())
        self.assertEqual(sum(map(len, first.values())), 1024)
        owners, ids = {}, set()
        for split, rows in first.items():
            self.assertGreaterEqual(len(rows), 60)
            per_task = {(lang, kind): 0 for lang in ("de", "en")
                        for kind in ("select_second", "sort_three")}
            for row in rows:
                self.assertTrue(verify_row(row))
                self.assertEqual(row["source"], SOURCE)
                self.assertEqual(row["domain"], "instruction")
                self.assertNotIn(row["id"], ids)
                ids.add(row["id"])
                parts = row["id"].split("-")
                group = (parts[1], parts[2])
                self.assertEqual(owners.setdefault(group, split), split)
                language, kind = row["language"], row["task_type"]
                per_task[language, kind] += 1
                prompt, answer = (m["content"] for m in row["messages"])
                expected, semantic = _independent_task(kind, prompt, language)
                self.assertEqual(answer, expected)
                self.assertIsNotNone(semantic)
            self.assertTrue(all(count >= 10 for count in per_task.values()))

    def test_fails_closed_on_falsified_labels_prompts_metadata(self):
        row = make_splits()["train"][0]
        for mutation in (
            ("messages", [
                {"role": "user", "content": row["messages"][0]["content"]},
                {"role": "assistant", "content": "WRONG"},
            ]),
            ("source", "Qwen/Qwen3.5-9B"),
            ("license", "unknown"),
            ("language", "fr"),
            ("id", "../unsafe"),
            ("domain", "math"),
        ):
            changed = copy.deepcopy(row)
            changed[mutation[0]] = mutation[1]
            with self.subTest(change=mutation[0]):
                self.assertFalse(verify_row(changed))

    def test_atomic_nonoverwriting_native_jsonl_export(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "native-instruction-v1"
            report = write_local_dataset(out)
            self.assertEqual(report["schema"], SCHEMA)
            self.assertFalse(report["training_started"])
            self.assertFalse(report["training_approved"])
            self.assertFalse(report["native_quality_improved"])
            self.assertFalse(report["public_benchmark"])
            self.assertEqual(sum(report["split_counts"].values()), 1024)
            from frontier350.chat350_data import validate_dataset_root, validate_prompt_holdouts
            manifest, splits = validate_dataset_root(out)
            self.assertEqual(report["dataset_manifest_sha256"], manifest["manifest_sha256"])
            self.assertEqual(validate_prompt_holdouts(splits)["all_user_turn_overlap"], 0)
            metadata = (out / "metadata_report.json").read_text(encoding="utf8")
            self.assertNotIn("Gib nur das", metadata)
            self.assertNotIn("Sortiere diese", metadata)
            self.assertNotIn("Return only", metadata)
            with self.assertRaises(FileExistsError):
                write_local_dataset(out)


if __name__ == "__main__":
    unittest.main()
