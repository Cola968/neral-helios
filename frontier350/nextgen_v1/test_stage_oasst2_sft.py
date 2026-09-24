"""CPU-only tests for Phoenix OASST2 human-only staging."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage_oasst2_sft as stage


def split_tree(prefix: str, validation: bool) -> str:
    for i in range(10_000):
        value = f"{prefix}-{i}"
        is_valid = int(hashlib.sha256(value.encode()).hexdigest()[:8], 16) % 10 == 0
        if is_valid == validation:
            return value
    raise AssertionError("could not construct deterministic split id")


def msg(*, ident: str, tree: str, role: str, lang: str, text: str,
        parent=None, rank=None, synthetic=False, model_name=None) -> dict:
    return {
        "message_id": ident,
        "message_tree_id": tree,
        "parent_id": parent,
        "role": role,
        "lang": lang,
        "text": text,
        "rank": rank,
        "review_result": True,
        "review_count": 1,
        "deleted": False,
        "synthetic": synthetic,
        "tree_state": "ready_for_export",
        "model_name": model_name,
        "labels": {},
    }


class Oasst2StageTests(unittest.TestCase):
    def test_model_named_and_synthetic_rows_are_rejected(self):
        base = msg(ident="x", tree="t", role="assistant", lang="en",
                   text="A valid answer.", parent="p", rank=0)
        self.assertTrue(stage.permitted_oasst2(base, "assistant"))
        self.assertFalse(stage.permitted_oasst2(
            dict(base, model_name="external-model"), "assistant"))
        self.assertFalse(stage.permitted_oasst2(
            dict(base, synthetic=True), "assistant"))

    def test_selection_is_human_only_bilingual_and_split_disjoint(self):
        rows = []
        frozen = {stage.prompt_hash("This exact prompt is frozen.")}
        counter = 0
        for validation in (False, True):
            for lang in ("de", "en"):
                for j in range(3):
                    counter += 1
                    tree = split_tree(f"{lang}-{validation}-{j}", validation)
                    root_id = f"p-{counter}"
                    answer_id = f"a-{counter}"
                    prompt = f"{'Deutsche' if lang == 'de' else 'English'} Frage Nummer {counter}"
                    answer = f"{'Deutsche' if lang == 'de' else 'English'} Antwort Nummer {counter}"
                    rows.extend([
                        msg(ident=root_id, tree=tree, role="prompter", lang=lang,
                            text=prompt),
                        msg(ident=answer_id, tree=tree, role="assistant", lang=lang,
                            text=answer, parent=root_id, rank=0),
                    ])

        # These attractive-looking rows must never enter the staged output.
        bad_tree = split_tree("bad-synth", False)
        rows.extend([
            msg(ident="bad-p", tree=bad_tree, role="prompter", lang="en",
                text="Synthetic source prompt"),
            msg(ident="bad-a", tree=bad_tree, role="assistant", lang="en",
                text="Synthetic source answer", parent="bad-p", rank=0, synthetic=True),
            msg(ident="model-p", tree=bad_tree+"x", role="prompter", lang="en",
                text="Model source prompt"),
            msg(ident="model-a", tree=bad_tree+"x", role="assistant", lang="en",
                text="Model source answer", parent="model-p", rank=0,
                model_name="external-model"),
        ])

        # Frozen exact prompt also must be excluded.
        frozen_tree = split_tree("frozen", False)
        rows.extend([
            msg(ident="frozen-p", tree=frozen_tree, role="prompter", lang="en",
                text="This exact prompt is frozen."),
            msg(ident="frozen-a", tree=frozen_tree, role="assistant", lang="en",
                text="Do not stage me.", parent="frozen-p", rank=0),
        ])

        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / stage.SOURCE_NAME
            with gzip.open(source, "wt", encoding="utf-8") as stream:
                for row in rows:
                    stream.write(json.dumps(row) + "\n")
            with patch.object(stage, "sha_file", return_value=stage.SOURCE_SHA256):
                selected, report = stage.select(
                    source, frozen, train_limit=4, valid_limit=4,
                    min_de_train=1, min_de_valid=1,
                )

        self.assertGreaterEqual(report["candidate_counts"]["train"]["de"], 1)
        self.assertGreaterEqual(report["candidate_counts"]["train"]["en"], 1)
        self.assertGreaterEqual(report["candidate_counts"]["validation"]["de"], 1)
        self.assertGreaterEqual(report["candidate_counts"]["validation"]["en"], 1)

        all_rows = selected["train"] + selected["validation"]
        ids = {row["id"] for row in all_rows}
        self.assertNotIn("bad-a", ids)
        self.assertNotIn("model-a", ids)
        self.assertNotIn("frozen-a", ids)

        train_prompts = {stage.prompt_hash(row["user"]) for row in selected["train"]}
        valid_prompts = {stage.prompt_hash(row["user"]) for row in selected["validation"]}
        self.assertFalse(train_prompts & valid_prompts)


if __name__ == "__main__":
    unittest.main()
