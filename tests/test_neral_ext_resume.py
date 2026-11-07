"""Offline regression: NERAL-EXT GPU runs must preserve checkpoints on interruption.

Only synthetic records; no real model, Windows runner, network or weights.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.neral_ext_run import build_messages
from scripts.neral_ext_resume import write_variant_checkpointed

CASES = [
    {"id": f"task-{i}", "kind": "integer", "domain": "math",
     "prompt": f"Compute 10+{i}.", "answer": str(10 + i)}
    for i in range(5)
]
KEY = "a" * 64


def fake(case, messages):
    assert messages[-1] == {"role": "user", "content": case["prompt"]}
    return {"text": case["answer"], "complete": True, "new_tokens": 7}


class RestartableEvaluationTests(unittest.TestCase):
    def test_interruption_keeps_completed_rows_and_resumes_remaining_only(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "base.jsonl"
            attempted = []
            def first(case, messages):
                attempted.append(case["id"])
                if len(attempted) == 3:
                    raise TimeoutError("simulation of a 6h GPU timeout")
                return fake(case, messages)
            with self.assertRaises(TimeoutError):
                write_variant_checkpointed(
                    CASES, "base", target, first, build_messages, run_key=KEY)
            self.assertFalse(target.exists())
            self.assertEqual(attempted, ["task-0", "task-1", "task-2"])
            self.assertTrue((Path(temp) / "base.jsonl.progress").exists())

            replay = []
            def second(case, messages):
                replay.append(case["id"])
                return fake(case, messages)
            report = write_variant_checkpointed(
                CASES, "base", target, second, build_messages, run_key=KEY)
            self.assertEqual(replay, ["task-2", "task-3", "task-4"])
            self.assertEqual(report["resumed_cases"], 2)
            self.assertEqual(report["generated_tokens"], 35)
            self.assertEqual(report["cases"], 5)
            self.assertEqual(report["incomplete"], 0)
            results = [json.loads(line) for line in target.read_text("utf8").splitlines()]
            self.assertEqual([r["id"] for r in results], [c["id"] for c in CASES])
            self.assertEqual([r["text"] for r in results], [c["answer"] for c in CASES])
            self.assertTrue(all(set(row) == {"id", "text", "complete"} for row in results))
            self.assertFalse((Path(temp) / "base.jsonl.progress").exists())
            with self.assertRaises(ValueError):
                write_variant_checkpointed(
                    CASES, "base", target, fake, build_messages, run_key=KEY)

    def test_identity_and_dataset_mismatch_fail_closed_before_generation(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "neral.jsonl"
            def partial(case, messages):
                if case["id"] == "task-1":
                    raise TimeoutError
                return fake(case, messages)
            with self.assertRaises(TimeoutError):
                write_variant_checkpointed(
                    CASES, "neral_v5", target, partial, build_messages, run_key=KEY)
            progress = (Path(temp) / "neral.jsonl.progress").read_bytes()
            def unexpected(case, messages):
                raise AssertionError("incompatible checkpoint must never be resumed")
            for variant, cases, key in (
                ("base", CASES, KEY),
                ("neral_v5", CASES, "b" * 64),
                ("neral_v5", CASES[:-1], KEY),
                ("neral_v5", [dict(CASES[0], prompt="new")] + CASES[1:], KEY),
            ):
                with self.subTest(variant=variant, key=key[:2], count=len(cases)):
                    with self.assertRaises(ValueError):
                        write_variant_checkpointed(
                            cases, variant, target, unexpected, build_messages, run_key=key)
                    self.assertEqual((Path(temp) / "neral.jsonl.progress").read_bytes(), progress)
                    self.assertFalse(target.exists())

    def test_corrupt_or_truncated_checkpoint_rejected_without_executing_model(self):
        for junk in ('{"bad":true}', "not-json\n", '{"schema":1}\n',
                     '{"schema":1}\n{"id":"task-0"'):
            with self.subTest(junk=junk):
                with tempfile.TemporaryDirectory() as temp:
                    target = Path(temp) / "base.jsonl"
                    (Path(temp) / "base.jsonl.progress").write_text(junk, encoding="utf8")
                    with self.assertRaises(ValueError):
                        write_variant_checkpointed(
                            CASES, "base", target,
                            lambda *_: self.fail("generator must not be called"),
                            build_messages, run_key=KEY)
                    self.assertFalse(target.exists())

    def test_incomplete_answer_is_empty_and_counted_on_resume(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "base.jsonl"
            def first(case, messages):
                if case["id"] == "task-1":
                    raise TimeoutError
                return {"text": "x" * 4097, "complete": True, "new_tokens": 512}
            with self.assertRaises(TimeoutError):
                write_variant_checkpointed(
                    CASES, "base", target, first, build_messages, run_key=KEY)
            result = write_variant_checkpointed(
                CASES, "base", target, fake, build_messages, run_key=KEY)
            self.assertEqual(result["incomplete"], 1)
            self.assertEqual(result["resumed_cases"], 1)
            first_row = json.loads(target.read_text("utf8").splitlines()[0])
            self.assertEqual(first_row["text"], "")
            self.assertIs(first_row["complete"], False)

    def test_rejects_invalid_key_duplicate_ids_and_symlink_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp) / "base.jsonl"
            for cases, key in ((CASES, "no"), ([CASES[0]] * 2, KEY), ([], KEY)):
                with self.assertRaises(ValueError):
                    write_variant_checkpointed(
                        cases, "base", target, fake, build_messages, run_key=key)
            link = Path(temp) / "base.jsonl.progress"
            outside = Path(temp) / "secret.txt"
            outside.write_text("untouched", encoding="utf8")
            try:
                link.symlink_to(outside)
            except (OSError, NotImplementedError):
                self.skipTest("symlinks unavailable")
            with self.assertRaises(ValueError):
                write_variant_checkpointed(
                    CASES, "base", target, fake, build_messages, run_key=KEY)
            self.assertEqual(outside.read_text("utf8"), "untouched")


if __name__ == "__main__":
    unittest.main()
