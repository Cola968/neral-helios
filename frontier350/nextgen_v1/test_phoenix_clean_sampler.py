"""Synthetic-only contracts for experimental conservative sampling; no GPU/data."""
from __future__ import annotations

import unittest

from phoenix_clean_sampler import choose_screened_diverse, screen_reason
from phoenix_oasst2_bridge import select_tree_disjoint
from phoenix_oasst2_content_profile import screen_selected


class ScreenedSamplerContracts(unittest.TestCase):
    def setUp(self):
        self.messages = []
        for language in ("de", "en"):
            for index in range(65):
                tree = f"synthetic-screened-{language}-{index}"
                parent = f"source-parent-{language}-{index}"
                common = {"message_tree_id": tree, "lang": language,
                          "synthetic": False, "review_result": True,
                          "deleted": False, "model_name": None}
                prompt = f"Original independent question about test item {language} {index}?"
                answer = f"An entirely invented and clearly sufficient answer for item {language} {index}, with additional context."
                self.messages += [
                    {**common, "message_id": parent, "parent_id": None,
                     "role": "prompter", "text": prompt},
                    {**common, "message_id": f"source-answer-{language}-{index}",
                     "parent_id": parent, "role": "assistant", "text": answer, "rank": 0},
                ]

    def test_screen_rejects_contact_length_short_and_role_patterns(self):
        good = "This synthetic answer is sufficiently descriptive for an illustrative unit test."
        self.assertIsNone(screen_reason("Question about physics?", good))
        self.assertEqual(screen_reason("Contact me at test@example.org", good), "email_pattern")
        self.assertEqual(screen_reason("Phone: +49 151 12345678", good), "phone_pattern")
        self.assertEqual(screen_reason("x" * 513, good), "long_prompt")
        self.assertEqual(screen_reason("Brief?", "Yes."), "short_answer")
        self.assertEqual(screen_reason("Brief?", "x" * 2501), "long_answer")
        self.assertEqual(screen_reason("Brief?", "Assistant: " + good), "role_marker")
        self.assertEqual(screen_reason("Brief?", "<|eos|>" + good), "reserved_role_tokens")

    def test_selected_pairs_have_no_screen_flags_unique_trees_and_prompts(self):
        pairs, metadata = choose_screened_diverse(self.messages, max_en=50, max_de=50)
        self.assertEqual(metadata["selected_distinct_trees"], 100)
        self.assertFalse(metadata["training_approved"])
        self.assertFalse(metadata["holdout_checked"])
        splits, summary = select_tree_disjoint(pairs, self.messages, frozen=set())
        self.assertTrue(summary["split_disjoint"])
        languages = {item["id"]: lang for lang in pairs for item in pairs[lang]}
        if min(len(splits[name]) for name in ("train", "validation")) >= 8:
            screened = screen_selected(splits, languages)
            self.assertTrue(all(value == 0 for value in screened["flag_counts"].values()))
        self.assertNotIn("invented", str(metadata).lower())

    def test_determinism_across_message_reordering(self):
        left, report_left = choose_screened_diverse(self.messages, 50, 50)
        right, report_right = choose_screened_diverse(list(reversed(self.messages)), 50, 50)
        self.assertEqual(report_left, report_right)
        self.assertEqual(left, right)

    def test_rejects_duplicate_upstream_ids(self):
        self.messages.append(self.messages[0].copy())
        with self.assertRaisesRegex(ValueError, "duplicate upstream"):
            choose_screened_diverse(self.messages, 50, 50)

    def test_refuses_insufficient_eligible_data_after_exclusions(self):
        for item in self.messages:
            if item["role"] == "assistant" and item["lang"] == "de":
                item["text"] = "No."
        with self.assertRaisesRegex(ValueError, "insufficient distinct eligible"):
            choose_screened_diverse(self.messages, 50, 50)

    def test_duplicate_normalized_prompt_is_not_selected_twice(self):
        parent = next(item for item in self.messages if item["message_id"] == "source-parent-de-1")
        duplicate = next(item for item in self.messages if item["message_id"] == "source-parent-de-0")
        parent["text"] = duplicate["text"]
        pairs, meta = choose_screened_diverse(self.messages, 50, 50)
        prompts = [item["prompt"].casefold() for language in pairs for item in pairs[language]]
        self.assertEqual(len(prompts), len(set(prompts)))
        self.assertEqual(meta["selected_distinct_trees"], 100)


if __name__ == "__main__":
    unittest.main()
