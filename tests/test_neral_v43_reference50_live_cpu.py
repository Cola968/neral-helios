"""Non-GPU reference runner security and score contract tests."""
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from scripts.neral_v43_reference50_live import (
    validate_engine_identity, make_record, store_private, verify_docker_grader,
    ensure_port_unbound, finish, verify_result_seal, classify_raw_answer_failure,
)
from unittest.mock import patch
from hashlib import sha256

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

    def test_port_9690_requires_exclusive_bind_even_when_http_unhealthy(self):
        # A failed HTTP/connect probe cannot prove that the TCP port is free.
        with patch("socket.socket") as constructor:
            probe=constructor.return_value.__enter__.return_value
            self.assertTrue(ensure_port_unbound())
            probe.bind.assert_called_once_with(("127.0.0.1",9690))
            probe.bind.side_effect=OSError(10048,"local address in use")
            with self.assertRaisesRegex(RuntimeError,"Port 9690"):
                ensure_port_unbound()

    def test_windows_exclusive_address_use_is_set_if_supported(self):
        with patch("socket.SO_EXCLUSIVEADDRUSE",0xFFFFFFFB,create=True), \
             patch("socket.socket") as constructor:
            probe=constructor.return_value.__enter__.return_value
            self.assertTrue(ensure_port_unbound())
            probe.setsockopt.assert_called_once_with(
                __import__("socket").SOL_SOCKET,0xFFFFFFFB,1)

    def test_complete_results_sha_receipt_is_tamper_evident(self):
        with tempfile.TemporaryDirectory() as temp:
            folder=Path(temp)
            from scripts.neral_v43_reference50 import seal_cases
            seal_cases(self.cases, folder/"sealed50.jsonl",folder/"manifest.json")
            for engine,filename in (
                ("raw_qwen35_9b","raw9b-50.jsonl"),
                ("neral10_obsidian","neral10-50.jsonl")):
                with store_private(folder/filename) as output:
                    for case in self.cases:
                        row=make_record(case,engine,{"text":"SECRET"},
                            grade="ungraded",reason="blind_human_review",
                            elapsed=1,total_peak_mib=8000)
                        row["source_commit"]="c0a6be607a0797908e101db460e3ac297a70f088"
                        row["model_text"]="SECRET"
                        output.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
            summary=finish(folder)
            self.assertEqual(summary["pairs"],50)
            self.assertTrue(verify_result_seal(folder))
            with self.assertRaises(FileExistsError):finish(folder)
            output=folder/"raw9b-50.jsonl"
            output.write_bytes(output.read_bytes()+b" ")
            with self.assertRaises(ValueError):verify_result_seal(folder)

    def test_raw_token_budget_exhaustion_is_a_scored_failure_not_a_gpu_rerun(self):
        self.assertEqual(
            classify_raw_answer_failure(RuntimeError("Raw output incomplete or exceeded fixed budget")),
            "raw_answer_incomplete")
        self.assertEqual(
            classify_raw_answer_failure(RuntimeError("Raw response invalid")),
            "raw_answer_invalid")
        with self.assertRaises(RuntimeError):
            classify_raw_answer_failure(RuntimeError("Postload GPU total/headroom safety exceeded"))

    def test_raw_tokenizer_accepts_mapping_not_just_builtin_dict(self):
        from collections import UserDict
        from scripts.neral_v43_reference50_live import encode_raw_prompt
        tokenizer=Mock()
        for value in ([12,34], {"input_ids":[12,34]},
                      UserDict({"input_ids":[12,34]})):
            tokenizer.apply_chat_template.return_value=value
            self.assertEqual(encode_raw_prompt(tokenizer,"synthetic"),[12,34])
        for invalid in ([],[1]*3201,[[1,2]],{"other":[1]},[True],[-1]):
            tokenizer.apply_chat_template.return_value=invalid
            with self.assertRaisesRegex(RuntimeError,"Raw input context invalid"):
                encode_raw_prompt(tokenizer,"synthetic")

    def test_raw_smoke_precedes_result_creation(self):
        from scripts import neral_v43_reference50_live as live
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            torch=Mock()
            with patch.object(live,"_root",return_value=([{"prompt":"sealed"}],{})), \
                 patch.object(live,"verify_docker_grader"), \
                 patch.object(live,"ensure_port_unbound"), \
                 patch.object(live,"_raw_model",return_value=(torch,Mock(),Mock())), \
                 patch.object(live,"encode_raw_prompt",return_value=[1]), \
                 patch.object(live,"_raw_answer",side_effect=RuntimeError("synthetic smoke failure")) as generate:
                with self.assertRaisesRegex(RuntimeError,"synthetic smoke failure"):
                    live.raw(out)
                self.assertEqual(generate.call_count,1)
                self.assertNotEqual(generate.call_args.args[-1],"sealed")
                self.assertFalse((out/"raw9b-50.jsonl").exists())

    def test_docker_grade_image_must_exist_before_any_gpu(self):
        absent=Mock(return_value=subprocess.CompletedProcess(
            args=[],returncode=1,stdout="",stderr=""))
        with self.assertRaises(RuntimeError):
            verify_docker_grader(runner=absent)
        self.assertEqual(absent.call_count,1)


if __name__=="__main__":
    unittest.main()
