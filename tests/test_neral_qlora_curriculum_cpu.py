"""CPU contracts for synthetic curriculum and untouched 24-case holdout."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from scripts.neral_qlora_dataset import generate, load_holdout
from scripts.neral_qlora_holdout_eval import (
    load_holdout as verify_holdout, assert_disjoint)
from scripts.neral_qlora_train import validated_training_rows

HOLDOUT=Path(__file__).resolve().parents[1]/"datasets/neral_qwen35_holdout_v1.jsonl"


class QloraCurriculumCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.holdout=verify_holdout(HOLDOUT)
        cls.excluded,_=load_holdout(HOLDOUT)
        cls.rows=generate(pairs=250,exclusion_questions=cls.excluded)

    def test_holdout_has_24_real_oracle_verified_disjoint_questions(self):
        self.assertEqual(len(self.holdout),24)
        self.assertEqual(len({r["question"] for r in self.holdout}),24)
        self.assertEqual(sum("bool" in r["family"] for r in self.holdout),12)
        self.assertEqual(sum("ordering" in r["family"] for r in self.holdout),12)

    def test_family_mix_preserves_ordering_retention(self):
        # 360 paired questions -> 270 Boolean, 90 complete-order;
        # each question has a Turn-0 and a Turn-1 answer.
        from scripts.neral_qlora_dataset import generate
        rows=generate(pairs=360,exclusion_questions=self.excluded)
        self.assertEqual(len(rows),720)
        self.assertEqual(sum(r["meta"]["family"]=="boolean" for r in rows),540)
        self.assertEqual(sum(r["meta"]["family"]=="ordering" for r in rows),180)
        for kind in ("turn0","turn1"):
            self.assertEqual(sum(r["meta"]["family"]=="ordering" and
                                 r["meta"]["mode"]==kind for r in rows),90)

    def test_generate_500_oracle_examples_with_balance_and_no_leak(self):
        self.assertEqual(len(self.rows),500)
        self.assertEqual([sum(r["meta"]["mode"]==mode for r in self.rows)
                          for mode in ("turn0","turn1")],[250,250])
        ids={}
        for row in self.rows:
            self.assertEqual([m["role"] for m in row["messages"]],
                             ["system","user","assistant"])
            answer=json.loads(row["messages"][2]["content"])
            self.assertTrue(answer["complete"])
            self.assertEqual(answer["kind"],"formal")
            key=row["meta"]["question_sha256"]
            self.assertNotIn(key,self.excluded)
            ids.setdefault(key,set()).add(row["meta"]["mode"])
        self.assertEqual(len(ids),250)
        self.assertTrue(all(modes=={"turn0","turn1"} for modes in ids.values()))

    def test_train_loader_checks_holdout_and_schema_without_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            train=Path(tmp)/"train.jsonl"
            train.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n"
                                     for r in self.rows),encoding="utf-8")
            assert_disjoint(self.holdout,train)
            self.assertEqual(len(validated_training_rows(train,HOLDOUT)),500)

    def test_holdout_prompt_injected_into_training_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            train=Path(tmp)/"train.jsonl"
            wrong=json.loads(json.dumps(self.rows))
            wrong[0]["meta"]["question_sha256"]=__import__("hashlib").sha256(
                self.holdout[0]["question"].encode("utf-8")).hexdigest()
            train.write_text("".join(json.dumps(r)+"\n" for r in wrong),
                             encoding="utf-8")
            with self.assertRaises(ValueError):
                validated_training_rows(train,HOLDOUT)

    def test_native_chat_template_masks_only_assistant_tokens(self):
        from scripts.neral_qlora_train import encode_answer_only
        class NativeFakeTokenizer:
            def apply_chat_template(self, messages, *, tokenize,
                                    add_generation_prompt, enable_thinking):
                if not tokenize or enable_thinking:
                    raise AssertionError("Need native non-thinking tokens")
                text="".join(m["role"]+":"+m["content"]+"\n" for m in messages)
                if add_generation_prompt:
                    text+="assistant:"
                return [ord(ch) for ch in text]
        item=self.rows[0]
        output=encode_answer_only([item],NativeFakeTokenizer(),max_seq_length=65535)
        self.assertEqual(len(output),1)
        labels=output[0]["labels"]
        self.assertIn(-100,labels)
        self.assertGreater(sum(x>=0 for x in labels),10)
        self.assertEqual(len(labels),len(output[0]["input_ids"]))
        self.assertEqual(sum(x>=0 for x in labels),len(item["messages"][-1]["content"])+1)

    def test_native_template_refuses_truncation_or_malformed_alignment(self):
        from scripts.neral_qlora_train import encode_answer_only
        class BadTokenizer:
            def apply_chat_template(self, messages, *, tokenize,
                                    add_generation_prompt, enable_thinking):
                return [1] if add_generation_prompt else [2,3,4]
        with self.assertRaises(RuntimeError):
            encode_answer_only(self.rows[:1],BadTokenizer(),2048)

    def test_generation_deterministic(self):
        a=generate(pairs=250,seed=918,exclusion_questions=self.excluded)
        b=generate(pairs=250,seed=918,exclusion_questions=self.excluded)
        self.assertEqual(a,b)


if __name__=="__main__":
    unittest.main()
