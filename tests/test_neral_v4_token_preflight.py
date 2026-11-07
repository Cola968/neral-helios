"""CPU-only NERAL v4 tokenization contract tests. No model, GPU or training."""
import json
import tempfile
import unittest
from pathlib import Path
from scripts.neral_v4_curriculum import build
from scripts.neral_v4_token_preflight import prepare_report


class FakeTokenizer:
    def apply_chat_template(self, messages, *, chat_template, tokenize, add_generation_prompt):
        assert tokenize is True and chat_template == "PINNED_TEMPLATE"
        return list(range(11 if add_generation_prompt else 17))


class V4TokenizerContract(unittest.TestCase):
    def test_all_2560_candidate_rows_check_completion_only_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            build(Path(tmp))
            report=prepare_report(Path(tmp), FakeTokenizer(), "PINNED_TEMPLATE",max_length=512)
        self.assertEqual(report["rows"],{"train":1920,"val":320,"test":320})
        self.assertEqual(report["total_rows"],2560)
        self.assertEqual(report["token_stats"]["train"]["completion_tokens_min"],6)
        self.assertTrue(report["completion_only_mask_checked"])
        self.assertFalse(report["local_qwen_tokenizer_verified"])
        self.assertFalse(report["training_allowed"])
        self.assertFalse(report["training_performed"])
        self.assertFalse(report["adapter_modified"])
        self.assertFalse(report["quality_improvement_proven"])

    def test_mutated_candidate_rejected_before_loading_tokenizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            build(root)
            p=root/"train.jsonl"
            data=p.read_text(encoding="utf8").splitlines()
            row=json.loads(data[0])
            row["completion"][0]["content"]="mismatched_target"
            data[0]=json.dumps(row)
            p.write_text("\n".join(data)+"\n",encoding="utf8")
            with self.assertRaises(ValueError):
                prepare_report(root,FakeTokenizer(),"PINNED_TEMPLATE")

    def test_overflow_is_a_hard_failure_not_truncation(self):
        class Overflow(FakeTokenizer):
            def apply_chat_template(self,messages,*,chat_template,tokenize,add_generation_prompt):
                return list(range(500 if add_generation_prompt else 600))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            build(root)
            with self.assertRaisesRegex(ValueError,"length"):
                prepare_report(root,Overflow(),"PINNED_TEMPLATE")

    def test_local_gpu_workflow_never_trains_or_uploads_private_text(self):
        path=Path(".github/workflows/neral-v4-token-preflight.yml")
        w=path.read_text(encoding="utf8")
        for text in ("RUN_NERAL_V4_TOKEN_PREFLIGHT","self-hosted","neral-gpu",
                     "HF_HUB_OFFLINE","--model-dir","--trainer-root",
                     "completion_only_mask_checked","--summary","upload-artifact"):
            self.assertIn(text,w)
        for unsafe in ("--run-local-gpu","TRAIN_NERAL_V3_R16_R32_80_STEPS","--steps 80"):
            self.assertNotIn(unsafe,w)


if __name__=="__main__":
    unittest.main()
