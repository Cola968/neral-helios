"""No CUDA or real weights required: benchmark safety and accounting contracts."""
import pytest
from scripts.neral_generalization_gpu import validate_report, build_aggregate
from scripts.neral_generalization_probe import CASES


def test_report_allows_only_bounded_scalar_and_verified_models():
    base={c["id"]:c["expected"] for c in CASES}
    adapter={c["id"]:c["expected"] for c in CASES}
    r=build_aggregate(base,adapter,base_revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
                      adapter_sha256="a"*64, max_new_tokens=160)
    assert r["base_correct"]==r["adapter_correct"]==32
    assert r["model_quality_improvement_proven"] is False
    assert r["source_training_changed"] is False
    assert r["adapter_modified"] is False
    assert r["independent_external_holdout"] is False
    assert r["data_sent_to_external_model_provider"] is False
    assert validate_report(r)
    assert len(str(r))<12000
    assert "Schreibe" not in str(r)
    assert "def multiply" not in str(r)


def test_incomplete_or_forged_report_fails_closed():
    responses={c["id"]:c["expected"] for c in CASES}
    report=build_aggregate(responses,responses,
                           base_revision="c03e6d358207e414f1eca0bb1891e29f1db0e242",
                           adapter_sha256="b"*64,max_new_tokens=160)
    for key,val in (("question_count",31),("training_performed",True),
                    ("independent_external_holdout",True),("max_new_tokens",4096),
                    ("model_quality_improvement_proven",True)):
        changed=dict(report,**{key:val})
        with pytest.raises(ValueError):
            validate_report(changed)
    with pytest.raises(ValueError):
        validate_report(dict(report,raw_generation="private text"))


def test_model_loading_is_lazy_and_script_does_not_train():
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/"scripts"/"neral_generalization_gpu.py").read_text()
    assert "trainer.train(" not in source
    assert ".save_pretrained(" not in source
    assert "disable_adapter()" in source
    assert "HF_HUB_OFFLINE" in source
