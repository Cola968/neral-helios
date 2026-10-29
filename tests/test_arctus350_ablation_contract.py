"""RED -> GREEN: pre-register one-variable matched Arctus350 experiments.

All fixtures are synthetic metadata. No weights/PC/GPU/training/evaluations.
"""
import copy
import unittest
from frontier350.arctus350_ablation_contract import (
    AblationError, freeze_ablation, read_ablation_json
)

TOK="4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"

def fixture():
    return {
        "schema":"arctus350-preregistered-ablation-v1",
        "model_family":"neral-arctus350",
        "architecture":"frontier350-v1",
        "source_checkpoint_sha256":"a"*64,
        "tokenizer_sha256":TOK,
        "train_manifest_sha256":"b"*64,
        "val_manifest_sha256":"c"*64,
        "benchmark_sha256":"d"*64,
        "evaluation_split":"val",
        "protected_test_access":False,
        "seed":350726,
        "budget":{
            "optimizer_updates":64,"sequence_length":256,
            "micro_batch_size":4,"gradient_accumulation":1,
            "planned_assistant_target_tokens":32000,
            "max_selection_rounds":1,
        },
        "baseline":{"peak_lr":0.00002,"eos_weight":1.5,"sampling_mode":"bucketed"},
        "candidate":{"peak_lr":0.00002,"eos_weight":1.5,"sampling_mode":"compact"},
        "primary_metric":"heldout_exact_match",
        "secondary_metrics":["eos_rate","repeat_trigram_rate","german_exact_match"],
        "intent":"Compare equal-budget compact batching with bucketed batching, not model quality claims.",
    }

class AblationContractTests(unittest.TestCase):
    def test_valid_mono_variable_pre_registration_has_deterministic_digest(self):
        x=freeze_ablation(fixture())
        self.assertEqual(x["changed_variable"],"sampling_mode")
        self.assertEqual(x["status"],"FROZEN_PLAN_NOT_RUN")
        self.assertFalse(x["native_inference_verified"])
        self.assertFalse(x["training_executed"])
        self.assertEqual(x["release_authorization"],"none")
        self.assertEqual(len(x["contract_sha256"]),64)
        raw=fixture()
        shuffled=dict(reversed(list(raw.items())))
        self.assertEqual(freeze_ablation(shuffled)["contract_sha256"],x["contract_sha256"])
        self.assertEqual(x["budget"]["optimizer_updates"],64)

    def test_only_one_independent_variable_may_change(self):
        data=fixture()
        data["candidate"]["peak_lr"]=1e-5
        with self.assertRaises(AblationError):
            freeze_ablation(data)
        same=fixture()
        same["candidate"]=copy.deepcopy(same["baseline"])
        with self.assertRaises(AblationError):
            freeze_ablation(same)

    def test_pinned_tokenizer_source_and_holdout_validation(self):
        for key,bad in (
            ("source_checkpoint_sha256","x"*64),("tokenizer_sha256","f"*64),
            ("train_manifest_sha256","weird"),
            ("val_manifest_sha256","0"*63),
            ("benchmark_sha256","d"*65),
            ("evaluation_split","test"),
            ("protected_test_access",True),
            ("model_family","qwen"),("architecture","unknown"),
        ):
            # Different valid source checkpoints are allowed: only syntax is
            # checked. Source identity is pinned *within* a frozen contract.
            if key=="source_checkpoint_sha256":bad="x"*63
            data=fixture()
            data[key]=bad
            with self.subTest(key=key),self.assertRaises(AblationError):
                freeze_ablation(data)

    def test_locked_equal_compute_budget_is_required(self):
        badbudgets=(
            {"optimizer_updates":0},{"optimizer_updates":True},
            {"optimizer_updates":100001},{"sequence_length":0},
            {"sequence_length":8193},{"micro_batch_size":0},
            {"gradient_accumulation":0},{"planned_assistant_target_tokens":0},
            {"max_selection_rounds":2},{"max_selection_rounds":True},
        )
        for change in badbudgets:
            v=fixture();v["budget"].update(change)
            with self.subTest(budget=change),self.assertRaises(AblationError):
                freeze_ablation(v)
        v=fixture();v["budget"]["separate_candidate_steps"]=9999
        with self.assertRaises(AblationError):freeze_ablation(v)

    def test_untrusted_model_settings_and_broad_search_are_rejected(self):
        variants=(
            {"candidate":{"peak_lr":"2e-5","eos_weight":1.5,"sampling_mode":"compact"}},
            {"candidate":{"peak_lr":0.0,"eos_weight":1.5,"sampling_mode":"compact"}},
            {"candidate":{"peak_lr":2e-5,"eos_weight":float("nan"),"sampling_mode":"compact"}},
            {"candidate":{"peak_lr":2e-5,"eos_weight":1.5,"sampling_mode":"adaptive"}},
            {"candidate":{"peak_lr":2e-5,"eos_weight":1.5,"sampling_mode":"compact","teacher":"api"}},
        )
        for change in variants:
            v=fixture();v.update(change)
            with self.subTest(change=str(change)),self.assertRaises(AblationError):
                freeze_ablation(v)

    def test_unknown_fields_and_duplicate_secondary_metrics_refused(self):
        for attr,val in (
            ("answer","I am already intelligent"),
            ("secondary_metrics",["eos_rate","eos_rate"]),
            ("secondary_metrics",["protected_test_score"]),
            ("primary_metric","train_loss"),
            ("intent","x"),
        ):
            v=fixture();v[attr]=val
            with self.subTest(attr=attr),self.assertRaises(AblationError):
                freeze_ablation(v)

    def test_no_modification_of_input_or_pretend_results(self):
        original=fixture();before=copy.deepcopy(original)
        r=freeze_ablation(original)
        self.assertEqual(before,original)
        r["budget"]["optimizer_updates"]=987
        self.assertEqual(original["budget"]["optimizer_updates"],64)
        self.assertNotIn("accuracy",str(r))

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_extra_arrays(self):
        for raw in (
            '{"schema":"a","schema":"b"}',
            '{"v":NaN}',
            '[]',
            '{"missing":1}',
            'x'*16385,
        ):
            with self.subTest(raw=raw[:20]),self.assertRaises(AblationError):
                read_ablation_json(raw)

    def test_json_roundtrip_and_multiple_seeds_not_permitted(self):
        import json
        obj=fixture()
        self.assertEqual(freeze_ablation(read_ablation_json(json.dumps(obj)))["changed_variable"],"sampling_mode")
        obj["seed"]=[1,2,3]
        with self.assertRaises(AblationError):freeze_ablation(obj)

if __name__=="__main__":
    unittest.main()
