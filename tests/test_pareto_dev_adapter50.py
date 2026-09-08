from pareto_dev_adapter50 import build_dev_vector


def sft_report():
    return {
        "macro_answer_nll": 1.0,
        "slices": {
            "de/math": {"answer_nll": 1.1},
            "en/math": {"answer_nll": 0.9},
            "de/code": {"answer_nll": 0.8},
            "en/code": {"answer_nll": 1.0},
            "de/reasoning": {"answer_nll": 1.2},
            "en/reasoning": {"answer_nll": 1.0},
        },
    }


def free_report():
    return {
        "utf8_valid_pct": 100.0,
        "eos_pct": 25.0,
        "repeated_trigram_pct": 5.0,
        "health_score": 20.0,
    }


def test_builds_flat_vector_from_existing_dev_reports():
    result = build_dev_vector(
        id_math={"nll": .7}, transfer_math={"nll": 1.3},
        id_exact={"overall": {"exact_pct": 12.0}},
        transfer_exact={"overall": {"exact_pct": 2.0}},
        sft=sft_report(), lm={"macro": 1.1},
        free=free_report(),
        context={"accuracy_pct": 83.3333333333},
    )
    assert result["id_exact_pct"] == 12.0
    assert result["transfer_exact_pct"] == 2.0
    assert result["sft_math_nll"] == 1.0
    assert result["sft_code_nll"] == .9
    assert result["sft_reasoning_nll"] == 1.1
    assert result["sft_de_nll"] == (1.1 + .8 + 1.2) / 3
    assert result["free_utf8_pct"] == 100.0
    assert result["context_dev_pct"] > 83.3


def test_rejects_missing_required_domain():
    sft = sft_report()
    sft["slices"] = {k: v for k, v in sft["slices"].items() if not k.endswith("/code")}
    try:
        build_dev_vector(
            id_math={"nll": .7}, transfer_math={"nll": 1.3},
            id_exact={"overall": {"exact_pct": 12.0}}, transfer_exact={"overall": {"exact_pct": 2.0}},
            sft=sft, lm={"macro": 1.1},
            free=free_report(),
            context={"accuracy_pct": 80.0},
        )
    except ValueError as exc:
        assert "code" in str(exc)
    else:
        raise AssertionError("expected ValueError")
