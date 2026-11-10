"""Synthetic, offline OASST2 lineage checks. No Hub downloads, Torch or GPU."""
from __future__ import annotations

import gzip
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import phoenix_oasst2_bridge as bridge
from sft_contract import prompt_hash, sha


class OASST2BridgeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.archive = self.root / "fixture.messages.jsonl.gz"
        self.holdout = self.root / "holdout_hashes.json"
        self.holdout.write_text(json.dumps([prompt_hash("independent frozen holdout")]), encoding="utf-8")
        self.messages = []
        self.pairs = {"de": [], "en": []}
        # Choose real hash buckets so synthetic fixture is stable and contains
        # at least nine distinct message trees in EACH split.
        found = {"train": 0, "validation": 0}
        count = 0
        while min(found.values()) < 9:
            count += 1
            tree = f"synthetic-tree-{count}"
            split = bridge.tree_split(tree)
            if found[split] >= 9:
                continue
            found[split] += 1
            language = "de" if count % 2 == 0 else "en"
            parent_id, assistant_id = f"parent-{count}", f"reply-{count}"
            user = f"Unique public test question {count}?"
            answer = f"A separate synthetic answer for question {count}."
            common = {"message_tree_id": tree, "lang": language,
                      "synthetic": False, "review_result": True, "deleted": False,
                      "model_name": None}
            self.messages.extend([
                {**common, "message_id": parent_id, "parent_id": None,
                 "role": "prompter", "text": user},
                {**common, "message_id": assistant_id, "parent_id": parent_id,
                 "role": "assistant", "text": answer, "rank": 0},
            ])
            self.pairs[language].append({"id": "oasst2-" + assistant_id,
                                         "upstream_parent_id": parent_id,
                                         "prompt": user, "answer": answer})
        self.write_archive()

    def write_archive(self):
        with gzip.open(self.archive, "wt", encoding="utf-8") as stream:
            for row in self.messages:
                stream.write(json.dumps(row) + "\n")

    def test_real_tree_unit_produces_disjoint_training_and_validation(self):
        splits, report = bridge.select_tree_disjoint(self.pairs, self.messages, set())
        self.assertGreaterEqual(min(map(len, splits.values())), 8)
        self.assertTrue(report["split_disjoint"])
        trees_by_id = {"oasst2-" + row["message_id"]: row["message_tree_id"]
                       for row in self.messages if row["role"] == "assistant"}
        train = {trees_by_id[row["id"]] for row in splits["train"]}
        valid = {trees_by_id[row["id"]] for row in splits["validation"]}
        self.assertFalse(train & valid)
        self.assertEqual(sum(report["rows"].values()), 18)

    def test_second_turn_from_same_tree_stays_in_same_split(self):
        original = self.messages[1]
        second_id = "new-answer-in-same-tree"
        self.messages.append({**original, "message_id": second_id,
                              "text": "Second independent answer with distinct prompt."})
        parent = self.messages[0]
        self.messages.append({**parent, "message_id": "new-parent-in-same-tree",
                              "text": "Another distinct question in same conversation."})
        self.messages[-2]["parent_id"] = "new-parent-in-same-tree"
        self.pairs[original["lang"]].append({"id": "oasst2-" + second_id,
                                             "upstream_parent_id": "new-parent-in-same-tree",
                                             "prompt": self.messages[-1]["text"],
                                             "answer": self.messages[-2]["text"]})
        splits, _ = bridge.select_tree_disjoint(self.pairs, self.messages, set())
        containing = [name for name, rows in splits.items()
                      if "oasst2-" + original["message_id"] in {row["id"] for row in rows}]
        self.assertEqual(len(containing), 1)
        self.assertIn("oasst2-" + second_id, {row["id"] for row in splits[containing[0]]})

    def test_missing_or_inconsistent_tree_rejected(self):
        self.messages[1]["message_tree_id"] = "different-tree"
        with self.assertRaisesRegex(ValueError, "tree ID"):
            bridge.select_tree_disjoint(self.pairs, self.messages, set())

    def test_mismatched_converted_answer_rejected(self):
        pair = self.pairs["en"][0]
        pair["answer"] = "Unverified substituted answer."
        with self.assertRaisesRegex(ValueError, "reviewed human source"):
            bridge.select_tree_disjoint(self.pairs, self.messages, set())

    def test_mismatched_converted_prompt_rejected(self):
        pair = self.pairs["de"][0]
        pair["prompt"] = "Mismatched user question?"
        with self.assertRaisesRegex(ValueError, "reviewed human source"):
            bridge.select_tree_disjoint(self.pairs, self.messages, set())

    def test_unreviewed_or_wrong_language_message_rejected(self):
        self.messages[1]["review_result"] = False
        with self.assertRaisesRegex(ValueError, "reviewed human source"):
            bridge.select_tree_disjoint(self.pairs, self.messages, set())
        self.messages[1]["review_result"] = True
        self.messages[1]["lang"] = "unverified"
        with self.assertRaisesRegex(ValueError, "reviewed human source"):
            bridge.select_tree_disjoint(self.pairs, self.messages, set())

    def test_frozen_prompt_overlap_fails_before_export(self):
        frozen = {prompt_hash(self.pairs["en"][0]["prompt"])} if self.pairs["en"] else {
            prompt_hash(self.pairs["de"][0]["prompt"])}
        with self.assertRaisesRegex(ValueError, "frozen benchmark"):
            bridge.select_tree_disjoint(self.pairs, self.messages, frozen)

    def test_prompt_duplicate_is_dropped_and_disclosed(self):
        duplicated = self.pairs["en"][0] if self.pairs["en"] else self.pairs["de"][0]
        other = self.pairs["de"][0] if duplicated is not self.pairs["de"][0] else self.pairs["en"][0]
        # Change BOTH the selected pair and its local source message. Otherwise
        # the stronger raw-provenance check rightly rejects the mismatch.
        other["prompt"] = duplicated["prompt"]
        for message in self.messages:
            if message["message_id"] == other["upstream_parent_id"]:
                message["text"] = other["prompt"]
        _, report = bridge.select_tree_disjoint(self.pairs, self.messages, set())
        self.assertEqual(report["duplicate_prompts_skipped"], 1)

    def test_local_preparation_requires_pinned_bytes_and_never_approves(self):
        out = self.root / "new-export"
        with patch.object(bridge, "UPSTREAM_SHA256", sha(self.archive)):
            report = bridge.prepare(self.root, self.archive, self.holdout, out,
                                    max_en=50, max_de=50)
        self.assertFalse(report["approved"])
        self.assertTrue(report["human_review_required"])
        self.assertTrue((out / "source_report.json").is_file())
        self.assertEqual(report["train_sha256"], sha(out / "train.jsonl"))
        self.assertEqual(report["validation_sha256"], sha(out / "validation.jsonl"))
        self.assertEqual(report["rows"]["train"] + report["rows"]["validation"], 18)
        for split in ("train", "validation"):
            for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines():
                self.assertEqual(set(json.loads(line)), {"id", "source_id", "user", "assistant"})
        self.assertNotIn("Unique public test question", (out / "source_report.json").read_text())

    def test_wrong_archive_digest_fails_before_any_directory_creation(self):
        out = self.root / "no-export"
        with self.assertRaisesRegex(ValueError, "pinned upstream SHA"):
            bridge.prepare(self.root, self.archive, self.holdout, out,
                           max_en=50, max_de=50)
        self.assertFalse(out.exists())

    def test_symlink_archive_refused(self):
        link = self.root / "alias.gz"
        try:
            link.symlink_to(self.archive)
        except OSError:
            self.skipTest("symlink creation unsupported")
        with self.assertRaisesRegex(ValueError, "symlinks"):
            bridge.prepare(self.root, link, self.holdout, self.root / "unused",
                           max_en=50, max_de=50)


if __name__ == "__main__":
    unittest.main()
