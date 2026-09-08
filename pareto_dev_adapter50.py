"""Convert existing Neral-50M development evaluators into one Pareto-guard vector.

The adapter is intentionally release-benchmark blind. It consumes only training-time held-out
reports: ID/transfer math NLL and exact-match, screened SFT validation, document-bounded LM
validation, free-generation dev health and the independent context-dev suite.
"""
from __future__ import annotations


def _f(value, name: str) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid {name}: {value!r}") from exc
    if out != out or out in (float("inf"), float("-inf")):
        raise ValueError(f"non-finite {name}: {out}")
    return out


def domain_nll(sft: dict, domain: str) -> float:
    values = [
        _f(row.get("answer_nll"), f"SFT {key} answer_nll")
        for key, row in sft.get("slices", {}).items()
        if key.endswith("/" + domain)
    ]
    if not values:
        raise ValueError(f"SFT validation has no {domain!r} domain")
    return sum(values) / len(values)


def german_nll(sft: dict) -> float:
    values = [
        _f(row.get("answer_nll"), f"SFT {key} answer_nll")
        for key, row in sft.get("slices", {}).items()
        if key.startswith("de/")
    ]
    if not values:
        raise ValueError("SFT validation has no German slices")
    return sum(values) / len(values)


def build_dev_vector(
    *,
    id_math: dict,
    transfer_math: dict,
    id_exact: dict,
    transfer_exact: dict,
    sft: dict,
    lm: dict,
    free: dict,
    context: dict,
) -> dict:
    return {
        "id_exact_pct": _f(id_exact["overall"]["exact_pct"], "ID exact_pct"),
        "transfer_exact_pct": _f(transfer_exact["overall"]["exact_pct"], "transfer exact_pct"),
        "id_nll": _f(id_math["nll"], "ID NLL"),
        "transfer_nll": _f(transfer_math["nll"], "transfer NLL"),
        "sft_macro_nll": _f(sft["macro_answer_nll"], "SFT macro NLL"),
        "sft_math_nll": domain_nll(sft, "math"),
        "sft_code_nll": domain_nll(sft, "code"),
        "sft_reasoning_nll": domain_nll(sft, "reasoning"),
        "sft_de_nll": german_nll(sft),
        "lm_macro": _f(lm["macro"], "LM macro"),
        "free_utf8_pct": _f(free["utf8_valid_pct"], "free UTF-8 validity"),
        "free_eos_pct": _f(free["eos_pct"], "free EOS"),
        "free_repetition_pct": _f(free["repeated_trigram_pct"], "free repetition"),
        "free_health_score": _f(free["health_score"], "free health"),
        "context_dev_pct": _f(context["accuracy_pct"], "context dev accuracy"),
    }
