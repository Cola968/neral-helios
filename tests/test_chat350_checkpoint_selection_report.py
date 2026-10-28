"""Independent-review regression: SFT final test must use selected best_chat.pt.

CPU-only structural tests. Never load a real model, pickle or GPU checkpoint.
"""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from frontier350.chat350_checkpoint_selection import evaluate_selected_checkpoint


class FakeModel:
    def __init__(self):
        self.identity = "last_step"
        self.load_calls = []

    def load_state_dict(self, state, strict):
        self.load_calls.append((state, strict))
        self.identity = state["label"]


class SelectedCheckpointTests(unittest.TestCase):
    config = {"native_model": "arctus350"}
    tokenizer = "locked-tokenizer"
    source = "immutable-foundation"
    data = "val-manifest"

    def payload(self, **overrides):
        result = {
            "model": {"label": "best_step"},
            "config": self.config,
            "tokenizer_sha256": self.tokenizer,
            "source_checkpoint_sha256": self.source,
            "data_manifest_sha256": self.data,
            "chat_step": 200,
            "validation": {"answer_nll": 1.8},
        }
        result.update(overrides)
        return result

    def invoke(self, model, checkpoint, payload, calls, **kw):
        def load(_):
            calls.append("load")
            return payload
        def ev(kind):
            def go():
                calls.append(f"eval_{kind}")
                return {"answer_nll": 1.8 if model.identity == "best_step" else 2.6,
                        "model_identity": model.identity}
            return go
        return evaluate_selected_checkpoint(
            model, checkpoint, loader=load, checkpoint_digest=lambda _: "c" * 64,
            expected_config=self.config,
            expected_tokenizer_sha256=self.tokenizer,
            expected_source_sha256=self.source,
            expected_data_manifest_sha256=self.data,
            last_step=1200, expected_best_step=200,
            eval_validation=ev("val"), eval_test=ev("test"),
            eval_replay=ev("replay"), **kw)

    def test_best_weights_are_loaded_before_any_test_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck=Path(tmp)/"best_chat.pt"
            ck.touch()
            model=FakeModel()
            calls=[]
            report=self.invoke(model,ck,self.payload(),calls)
            self.assertEqual(model.identity,"best_step")
            self.assertEqual(model.load_calls,[(self.payload()["model"],True)])
            self.assertEqual(calls,["load","eval_val","eval_test","eval_replay"])
            self.assertEqual(report["selected_checkpoint_step"],200)
            self.assertEqual(report["selected_checkpoint_test_sft"]["model_identity"],"best_step")
            self.assertEqual(report["selected_checkpoint_sha256"],"c"*64)

    def test_missing_best_checkpoint_never_leaks_test_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            model=FakeModel()
            calls=[]
            self.assertIsNone(self.invoke(model,Path(tmp)/"missing.pt",self.payload(),calls,
                                          expected_best_step=None))
            self.assertEqual(model.identity,"last_step")
            self.assertEqual(calls,[])

    def test_reject_wrong_checkpoint_lineage_before_test_eval(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck=Path(tmp)/"best_chat.pt"
            ck.touch()
            for key,val in (
                ("config",{"foreign_model":True}),
                ("tokenizer_sha256","wrong"),
                ("source_checkpoint_sha256","wrong"),
                ("data_manifest_sha256","wrong"),
                ("chat_step",100),
            ):
                model,calls=FakeModel(),[]
                with self.subTest(key=key),self.assertRaises(ValueError):
                    self.invoke(model,ck,self.payload(**{key:val}),calls)
                self.assertEqual(model.load_calls,[])
                self.assertNotIn("eval_test",calls)

    def test_reject_missing_validation_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck=Path(tmp)/"best_chat.pt"
            ck.touch()
            for validation in (None,{},{"answer_nll":float("nan")}):
                model,calls=FakeModel(),[]
                with self.subTest(value=validation),self.assertRaises(ValueError):
                    self.invoke(model,ck,self.payload(validation=validation),calls)
                self.assertNotIn("eval_test",calls)

    def test_selected_checkpoint_step_must_not_be_after_last_step(self):
        with tempfile.TemporaryDirectory() as tmp:
            ck=Path(tmp)/"best_chat.pt"
            ck.touch()
            model=FakeModel()
            calls=[]
            with self.assertRaises(ValueError):
                self.invoke(model,ck,self.payload(chat_step=1500),calls,expected_best_step=1500)
            self.assertNotIn("eval_test",calls)


if __name__ == "__main__":
    unittest.main()
