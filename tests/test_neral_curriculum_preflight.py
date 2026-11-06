"""Preflight gate must prove every row fits local Qwen completion-only training."""
import shutil
from pathlib import Path
import pytest

from scripts.neral_curriculum_preflight import prepare_report, verify_pinned_files

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"datasets"/"neral_curriculum_v1"


class FakeTokenizer:
    def apply_chat_template(self, messages, *, chat_template, tokenize, add_generation_prompt):
        assert tokenize is True and chat_template == "PINNED_TEMPLATE"
        # This is deliberately only a protocol fake; GPU tokenizer is a
        # separate mandatory manual verification.
        return list(range(11 if add_generation_prompt else 17))


def test_all_1000_rows_have_completion_only_training_boundary():
    report=prepare_report(DATA, FakeTokenizer(), "PINNED_TEMPLATE", max_length=512)
    assert report["status"] == "TOKENIZER_INTERFACE_PASSED_NOT_MODEL_VERIFIED"
    assert report["local_qwen_tokenizer_verified"] is False
    assert report["rows"] == {"train":800,"val":100,"test":100}
    assert report["tokenizer_contract_checked"] is True
    assert report["completion_only_mask_checked"] is True
    assert report["trainable_examples"] == 800
    assert report["training_performed"] is False
    assert report["adapter_modified"] is False
    assert report["new_adapter_quality_proven"] is False
    assert len(report["files_sha256"]["train"]) == 64
    assert report["token_stats"]["train"]["completion_tokens_min"] == 6


def test_corrupt_source_file_fails_sha_pin_before_tokenizer(tmp_path):
    for filename in ("train.jsonl","val.jsonl","test.jsonl","manifest.json"):
        shutil.copyfile(DATA/filename,tmp_path/filename)
    with (tmp_path/"train.jsonl").open("a",encoding="utf8") as f:
        f.write("corrupted\n")
    with pytest.raises(ValueError, match="pin"):
        verify_pinned_files(tmp_path)


def test_mask_boundary_failure_rejected():
    class BadTokenizer(FakeTokenizer):
        def apply_chat_template(self,messages,*,chat_template,tokenize,add_generation_prompt):
            if add_generation_prompt: return [1,2,3]
            return [9,9,9,4,5]
    with pytest.raises(ValueError,match="boundary"):
        prepare_report(DATA,BadTokenizer(),"PINNED_TEMPLATE",max_length=512)


def test_token_overflow_refused_instead_of_silently_truncated():
    class LongTokenizer(FakeTokenizer):
        def apply_chat_template(self,messages,*,chat_template,tokenize,add_generation_prompt):
            return list(range(511 if add_generation_prompt else 600))
    with pytest.raises(ValueError,match="length"):
        prepare_report(DATA,LongTokenizer(),"PINNED_TEMPLATE",max_length=512)


def test_no_checkout_or_gpu_load_in_cpu_preflight_module():
    source=(ROOT/"scripts"/"neral_curriculum_preflight.py").read_text(encoding="utf8")
    assert "trainer.train(" not in source
    assert "model.save_pretrained(" not in source
    assert "torch.cuda" not in source
    assert "HF_HUB_OFFLINE" in source
