"""Synthetic contracts for risk-aware OASST2 selection; no real data/GPU."""
from __future__ import annotations

import copy
import unittest

from phoenix_risk_aware_selector import choose_risk_aware


class RiskAwareSelectorContracts(unittest.TestCase):
    def build(self, per_lang=12):
        rows = []
        for lang in ("de", "en"):
            for i in range(per_lang):
                tree = f"{lang}-tree-{i}"
                parent = f"{lang}-p-{i}"
                common = {"message_tree_id": tree, "lang": lang,
                          "synthetic": False, "review_result": True,
                          "deleted": False, "model_name": None}
                rows.extend([
                    {**common, "message_id": parent, "parent_id": None,
                     "role": "prompter", "text": f"A sufficiently distinct public synthetic question {lang} {i}?"},
                    {**common, "message_id": f"{lang}-a-{i}", "parent_id": parent,
                     "role": "assistant", "text": f"This is a sufficiently long reviewed synthetic answer number {i} in {lang} for selector testing.", "rank": 0},
                ])
        return rows

    def test_selects_one_clean_pair_per_tree_and_is_deterministic(self):
        rows = self.build()
        first, a = choose_risk_aware(rows, max_de=8, max_en=8)
        second, b = choose_risk_aware(list(reversed(rows)), max_de=8, max_en=8)
        self.assertEqual([r["id"] for r in first["de"]], [r["id"] for r in second["de"]])
        self.assertEqual([r["id"] for r in first["en"]], [r["id"] for r in second["en"]])
        self.assertEqual(a["selected_id_inventory_sha256"], b["selected_id_inventory_sha256"])
        self.assertEqual(a["distinct_trees_selected"], 16)
        self.assertFalse(a["training_approved"])

    def test_contact_and_short_answer_candidates_are_excluded(self):
        rows = self.build(per_lang=14)
        # Change one answer in each language into a hard warning; enough clean trees remain.
        for row in rows:
            if row.get("message_id") == "de-a-0":
                row["text"] = "mail me at test@example.com for a private reply that should be excluded"
            if row.get("message_id") == "en-a-0":
                row["text"] = "too short"
        _, report = choose_risk_aware(rows, max_de=8, max_en=8)
        rejected = report["hard_warning_rejections_before_tree_sampling"]
        self.assertGreaterEqual(rejected["contact_email"], 1)
        self.assertGreaterEqual(rejected["answer_too_short"], 1)

    def test_insufficient_clean_trees_fails_instead_of_relaxing_filter(self):
        rows = self.build(per_lang=8)
        for row in rows:
            if row.get("role") == "assistant" and row.get("lang") == "de":
                row["text"] = "short"
        with self.assertRaisesRegex(ValueError, "insufficient clean independent"):
            choose_risk_aware(rows, max_de=8, max_en=8)

    def test_duplicate_upstream_id_is_refused(self):
        rows = self.build()
        duplicate = copy.deepcopy(rows[0])
        rows.append(duplicate)
        with self.assertRaisesRegex(ValueError, "duplicate upstream message ID"):
            choose_risk_aware(rows, max_de=8, max_en=8)


if __name__ == "__main__":
    unittest.main()
