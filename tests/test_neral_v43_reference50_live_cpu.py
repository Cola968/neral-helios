"""Non-GPU reference runner security and score contract tests."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.neral_v43_reference50_live import (
    validate_engine_identity, make_record, store_private, verify_docker_grader,
)
from scripts.neral_v43_reference50 import generate_cases


class LiveReferenceCPU(unittest.TestCase):
    def setUp(self):
        self.cases=generate_cases("a7"*32)

    def test_only_identical_pinned_9b_models_and_valid_modes(self):
        self.assertEqual(validate_engine_identity(
            "raw_qwen35_9b",checkpoint="D:/neral-qlora/checkpoint",
            adapter_loaded=False), "raw_qwen35_9b")
        self.assertEqual(validate_engine_identity(
            "neral10_obsidian",checkpoint="D:/neral-qlora/checkpoint",
            adapter_loaded=True), "neral10_obsidian")
        with self.assertRaises(ValueError):
            validate_engine_identity("neral10_obsidian",
                                     checkpoint="D:/neral-qlora/checkpoint",
                                     adapter_loaded=False)
        with self.assertRaises(ValueError):
            validate_engine_identity("raw_qwen35_9b",
                                     checkpoint="C:/different",
                                     adapter_loaded=False)

    def test_no_raw_prompts_or_answers_in_public_record(self):
        source=self.cases[0]
        record=make_record(source,"raw_qwen35_9b",{"text":"PRIVATE ANSWER"},
                           grade="ungraded",reason="blind_human_review",
                           elapsed=2.5,total_peak_mib=8700,cuda_reserved_mib=7600)
        serialized=json.dumps(record)
        self.assertNotIn("PRIVATE ANSWER",serialized)
        self.assertNotIn(source["prompt"],serialized)
        self.assertEqual(record["grade"],"ungraded")
        self.assertEqual(record["latency_ms"],2500)

    def test_private_file_exclusive_and_unmodified(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/"neral.jsonl"
            with store_private(path) as sink:
                sink.write('{"id":"test"}\n')
            with self.assertRaises(FileExistsError):
                with store_private(path):
                    pass

    def test_docker_grade_image_must_exist_before_any_gpu(self):
        absent=Mock(return_value=subprocess.CompletedProcess(
            args=[],returncode=1,stdout="",stderr=""))
        with self.assertRaises(RuntimeError):
            verify_docker_grader(runner=absent)
        self.assertEqual(absent.call_count,1)


if __name__=="__main__":
    unittest.main()
