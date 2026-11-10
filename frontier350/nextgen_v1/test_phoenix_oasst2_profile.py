"""Synthetic source-selection diagnostic contracts; no real data, network or GPU."""
from __future__ import annotations

import unittest

from phoenix_oasst2_bridge import tree_split
from phoenix_oasst2_profile import profile_messages


class SourceProfileContracts(unittest.TestCase):
    def setUp(self):
        self.messages = []
        found = {"train": 0, "validation": 0}
        i = 0
        while min(found.values()) < 9:
            i += 1
            tree = f"public-synthetic-profile-tree-{i}"
            split = tree_split(tree)
            if found[split] >= 9:
                continue
            found[split] += 1
            lang = "de" if i % 2 else "en"
            common = {"message_tree_id": tree, "lang": lang,
                      "synthetic": False, "review_result": True,
                      "deleted": False, "model_name": None}
            parent = f"synthetic-parent-{i}"
            self.messages.extend([
                {**common, "message_id": parent, "parent_id": None,
                 "role": "prompter", "text": f"Independent synthetic public question {i}?"},
                {**common, "message_id": f"synthetic-answer-{i}", "parent_id": parent,
                 "role": "assistant", "text": f"Synthetic public reply number {i}.", "rank": 0},
            ])

    def test_real_chooser_and_tree_dry_run_never_claim_holdout_or_approval(self):
        report = profile_messages(self.messages)
        self.assertEqual(report["status"], "real_source_selector_dry_run_not_approved")
        self.assertEqual(report["source_messages"], len(self.messages))
        self.assertEqual(report["dry_run_rows"], {"train": 9, "validation": 9})
        self.assertTrue(report["tree_disjoint"])
        self.assertFalse(report["frozen_holdout_checked"])
        self.assertFalse(report["training_dataset_exported"])
        self.assertFalse(report["training_approved"])
        self.assertFalse(report["training_started"])
        self.assertNotIn("Independent synthetic public question", str(report))

    def test_wrong_source_lineage_is_refused_even_in_dry_run(self):
        self.messages[1]["message_tree_id"] = "unexpected-other-tree"
        with self.assertRaisesRegex(ValueError, "tree ID"):
            profile_messages(self.messages)

    def test_bad_caps_refused_before_chooser(self):
        with self.assertRaisesRegex(ValueError, "caps"):
            profile_messages(self.messages, max_en=1)


if __name__ == "__main__":
    unittest.main()
