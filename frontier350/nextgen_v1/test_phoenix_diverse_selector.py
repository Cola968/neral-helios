"""CPU-only synthetic test of a proposed sampler, NEVER approval or training."""
import json
import unittest

from phoenix_diverse_selector import choose_tree_diverse


class DiverseSelectorContracts(unittest.TestCase):
    def setUp(self):
        self.messages = []
        for n in range(35):
            # Independent trees in each language, plus shared trees across languages.
            for lang in ("de", "en"):
                tree = f"shared-artificial-tree-{n}" if n < 8 else f"{lang}-artificial-tree-{n}"
                prefix = f"{lang}-{n}"
                common = {"message_tree_id": tree, "lang": lang, "synthetic": False,
                          "review_result": True, "deleted": False, "model_name": None}
                self.messages.extend([
                    {**common, "message_id": f"parent-{prefix}", "parent_id": None,
                     "role": "prompter", "text": f"Question from fictional {prefix}?"},
                    {**common, "message_id": f"answer-{prefix}",
                     "parent_id": f"parent-{prefix}", "role": "assistant",
                     "text": f"Fictional answer from {prefix} only.", "rank": 0},
                ])

    def test_one_pair_per_tree_and_language_caps(self):
        pairs, meta = choose_tree_diverse(self.messages, max_de=12, max_en=12)
        self.assertEqual((len(pairs["de"]), len(pairs["en"])), (12, 12))
        by_id = {row["message_id"]: row for row in self.messages}
        trees = [by_id[p["id"][len("oasst2-"):]]["message_tree_id"]
                 for lang in ("de", "en") for p in pairs[lang]]
        self.assertEqual(len(set(trees)), 24)
        self.assertEqual(meta["distinct_trees_selected"], 24)
        self.assertFalse(meta["training_approved"])
        self.assertFalse(meta["holdout_checked"])
        self.assertNotIn("Fictional answer", json.dumps(meta))

    def test_reordering_source_does_not_change_selection_or_digest(self):
        first, report = choose_tree_diverse(self.messages, max_de=12, max_en=12)
        second, again = choose_tree_diverse(list(reversed(self.messages)), max_de=12, max_en=12)
        self.assertEqual(first, second)
        self.assertEqual(report, again)
        self.assertEqual(len(report["selected_id_inventory_sha256"]), 64)

    def test_invalid_caps_or_missing_distinct_trees_fail(self):
        with self.assertRaises(ValueError):
            choose_tree_diverse(self.messages, max_de=0, max_en=12)
        with self.assertRaisesRegex(ValueError, "insufficient independent"):
            choose_tree_diverse(self.messages, max_de=35, max_en=35)

    def test_wrong_tree_lineage_refused(self):
        self.messages[1]["message_tree_id"] = "wrong-tree"
        with self.assertRaisesRegex(ValueError, "conversation tree"):
            choose_tree_diverse(self.messages, max_de=12, max_en=12)

    def test_duplicate_upstream_message_id_refused(self):
        self.messages.append(dict(self.messages[0]))
        with self.assertRaisesRegex(ValueError, "duplicate upstream"):
            choose_tree_diverse(self.messages, max_de=12, max_en=12)


if __name__ == "__main__":
    unittest.main()
