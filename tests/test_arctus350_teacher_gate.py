"""CPU-only RED/GREEN contracts for licensed, checked Arctus teacher data."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from frontier350.arctus350_teacher_gate import prepare_teacher_candidate


class TeacherCandidateTests(unittest.TestCase):
    HASH = "a" * 64

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input = self.root / "teacher.jsonl"
        self.test = self.root / "frozen.jsonl"
        self.dest = self.root / "candidate"
        self.train = self.row("train-a", "train", "Berechne 59 + 77.", "136")
        self.val = self.row("val-a", "val", "Berechne 86 + 54.", "140")
        self.rows = [self.train, self.val]
        from frontier350.arctus350_grounding_curriculum_v1 import make_splits
        self.heldout = make_splits()["test"][:3]
        self.write_inputs()

    def tearDown(self):
        self.temp.cleanup()

    @classmethod
    def row(cls, identity, split, prompt, answer):
        return {
            "id": identity, "split": split, "language": "de", "domain": "math",
            "task_type": "addition", "prompt": prompt, "answer": answer,
            "independent_reference": answer, "human_reviewed": True,
            "teacher_model": "Qwen/Qwen3.5-9B", "teacher_artifact_sha256": cls.HASH,
            "teacher_license": "Apache-2.0", "generation_mode": "model-native",
            "tools_used": [],
        }

    def write_inputs(self):
        self.input.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.rows), encoding="utf8")
        self.test.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in self.heldout), encoding="utf8")

    def test_creates_native_compatible_splits_and_hashes_without_training(self):
        report = prepare_teacher_candidate(self.input, self.test, self.dest)
        from frontier350.chat350_data import validate_dataset_root, validate_prompt_holdouts
        manifest, rows = validate_dataset_root(self.dest)
        self.assertEqual(report["dataset_manifest_sha256"], manifest["manifest_sha256"])
        self.assertEqual(report["split_counts"], {"train": 1, "val": 1, "test": 3})
        self.assertEqual(report["teacher_model"], "Qwen/Qwen3.5-9B")
        self.assertFalse(report["training_started"])
        self.assertFalse(report["training_approved"])
        self.assertFalse(report["native_quality_improved"])
        self.assertTrue(report["frozen_test_source_verified"])
        self.assertEqual(validate_prompt_holdouts(rows)["all_user_turn_overlap"], 0)
        self.assertEqual(rows["train"][0]["messages"][1]["content"], "136")
        self.assertNotIn("Berechne", json.dumps(report))
        with self.assertRaises(FileExistsError):
            prepare_teacher_candidate(self.input, self.test, self.dest)

    def test_rejects_unreviewed_wrong_answer_and_fake_tool_assistance(self):
        for change in (
            {"human_reviewed": False}, {"answer": "137"},
            {"independent_reference": "999"}, {"tools_used": ["calculator"]},
            {"generation_mode": "model+tools"}, {"teacher_model": "Giotto Alto"},
            {"teacher_license": "unknown"}, {"teacher_artifact_sha256": "not-a-hash"},
            {"split": "test"},
        ):
            with self.subTest(change=change):
                self.rows = [{**self.train, **change}, self.val]
                self.write_inputs()
                with self.assertRaises(ValueError):
                    prepare_teacher_candidate(self.input, self.test, self.dest)
                self.assertFalse(self.dest.exists())

    def test_rejects_test_contamination_and_modified_frozen_references(self):
        self.rows = [self.train, {**self.val, "prompt": self.train["prompt"].lower()}]
        self.write_inputs()
        with self.assertRaises(ValueError):
            prepare_teacher_candidate(self.input, self.test, self.dest)
        self.rows = [self.train, self.val]
        self.heldout[0] = copy.deepcopy(self.heldout[0])
        self.heldout[0]["messages"][1]["content"] = "FALSCH"
        self.write_inputs()
        with self.assertRaises(ValueError):
            prepare_teacher_candidate(self.input, self.test, self.dest)

    def test_rejects_injected_json_and_extra_fields(self):
        raw = self.input.read_text(encoding="utf8").replace(
            '"human_reviewed": true',
            '"human_reviewed": true, "human_reviewed": true', 1
        )
        self.input.write_text(raw, encoding="utf8")
        with self.assertRaises(ValueError):
            prepare_teacher_candidate(self.input, self.test, self.dest)
        self.rows = [self.train, {**self.val, "unexpected": "field"}]
        self.write_inputs()
        with self.assertRaises(ValueError):
            prepare_teacher_candidate(self.input, self.test, self.dest)

    def test_rejects_generator_train_row_disguised_as_frozen_test(self):
        from frontier350.arctus350_grounding_curriculum_v1 import make_splits
        self.heldout = [make_splits()["train"][0]]
        self.write_inputs()
        with self.assertRaises(ValueError):
            prepare_teacher_candidate(self.input, self.test, self.dest)
        self.assertFalse(self.dest.exists())

    def test_accepts_verified_two_step_and_transitive_reasoning(self):
        self.rows = [
            {**self.train, "id": "math-chain", "task_type": "two_step",
             "prompt": "Berechne (59 + 77) - 14.", "answer": "122",
             "independent_reference": "122"},
            {**self.val, "id": "logic-chain", "task_type": "ordered_logic",
             "domain": "reasoning", "language": "en",
             "prompt": "Mara is faster than Lena. Lena is faster than Theo. Who is slowest?",
             "answer": "Theo", "independent_reference": "Theo"},
        ]
        self.write_inputs()
        report = prepare_teacher_candidate(self.input, self.test, self.dest)
        self.assertEqual(report["split_counts"]["train"], 1)
        self.assertEqual(report["split_counts"]["val"], 1)

    def test_rejects_invalid_multistep_and_cyclic_logic(self):
        variants = [
            {"task_type": "two_step", "prompt": "Berechne (59 + 77) - 14.",
             "answer": "123", "independent_reference": "123"},
            {"task_type": "ordered_logic", "domain": "reasoning",
             "prompt": "Mara ist schneller als Lena. Lena ist schneller als Theo. Wer ist am schnellsten?",
             "answer": "Lena", "independent_reference": "Lena"},
            {"task_type": "ordered_logic", "domain": "reasoning",
             "prompt": "Mara ist schneller als Lena. Lena ist schneller als Mara. Wer ist am schnellsten?",
             "answer": "Mara", "independent_reference": "Mara"},
        ]
        for item in variants:
            with self.subTest(item=item):
                self.rows = [{**self.train, **item}, self.val]
                self.write_inputs()
                with self.assertRaises(ValueError):
                    prepare_teacher_candidate(self.input, self.test, self.dest)
                self.assertFalse(self.dest.exists())

    def test_disallows_commutative_math_train_val_leakage(self):
        self.rows = [self.train, {**self.val, "prompt": "Calculate 77 + 59.",
                                  "answer": "136", "independent_reference": "136",
                                  "language": "en"}]
        self.write_inputs()
        with self.assertRaises(ValueError):
            prepare_teacher_candidate(self.input, self.test, self.dest)
        self.assertFalse(self.dest.exists())

    def test_disallows_order_logic_train_val_leakage(self):
        prompt = "Mara is faster than Lena. Lena is faster than Theo. Who is fastest?"
        reversed_prompt = "Lena is faster than Theo. Mara is faster than Lena. Who is fastest?"
        self.rows = [
            {**self.train, "task_type": "ordered_logic", "domain": "reasoning",
             "prompt": prompt, "answer": "Mara", "independent_reference": "Mara"},
            {**self.val, "task_type": "ordered_logic", "domain": "reasoning",
             "prompt": reversed_prompt, "answer": "Mara", "independent_reference": "Mara"},
        ]
        self.write_inputs()
        with self.assertRaises(ValueError):
            prepare_teacher_candidate(self.input, self.test, self.dest)
        self.assertFalse(self.dest.exists())

    def test_rejects_mixed_teacher_artifacts_and_symlinked_inputs(self):
        self.rows = [self.train, {**self.val, "teacher_artifact_sha256": "b" * 64}]
        self.write_inputs()
        with self.assertRaises(ValueError):
            prepare_teacher_candidate(self.input, self.test, self.dest)
        self.rows = [self.train, self.val]
        self.write_inputs()
        link = self.root / "link.jsonl"
        link.symlink_to(self.input)
        with self.assertRaises(ValueError):
            prepare_teacher_candidate(link, self.test, self.dest)


if __name__ == "__main__":
    unittest.main()
