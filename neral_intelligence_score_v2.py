"""Neral Intelligence Score 2.0 (NIS 2.0).

NIS 2.0 is a project-local 0-100 raw-model capability score. It expands NIS 1.0 with
math transfer/OOD, long-context, generation health, instruction following, multi-turn state,
truthfulness, robustness and compositional reasoning. Safety policy, tools, RAG and Helios
product features remain separate readiness gates and never inflate this model score.

The score combines a weighted arithmetic mean (85%) with a weighted geometric balance term
(15%). The balance term makes a model with a severe capability hole score lower than a model
with the same arithmetic average but a more even capability profile. A floor of 5 is used only
inside the balance term so a single zero does not collapse the entire score to zero.
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

from neral_intelligence_score import nis_v1
from summarize_strong50 import metrics as direct_metrics
from weakness_repair_gate50 import extract

SCORE_VERSION = "NIS-2.0"
WEIGHTS = {
    "language_knowledge": 0.10,
    "math_generalization": 0.15,
    "code": 0.10,
    "science_domain": 0.08,
    "reasoning_compositional": 0.12,
    "bilingual_translation": 0.05,
    "context_dialogue": 0.12,
    "instruction_following": 0.10,
    "truthfulness_robustness": 0.08,
    "generation_health": 0.10,
}


def _clamp(x: float) -> float:
    # NaN used to be silently converted into a perfect 100% component by
    # min/max float ordering. This would artificially inflate NIS2.
    if isinstance(x, bool):
        raise ValueError("NIS2 components must be finite numeric values, not booleans")
    try:
        value = float(x)
    except (TypeError, ValueError) as exc:
        raise ValueError("NIS2 components must be finite numeric values") from exc
    if not math.isfinite(value):
        raise ValueError("NIS2 components must be finite: NaN and infinity are invalid")
    return max(0.0, min(100.0, value))


def _mean(*values: float) -> float:
    return sum(float(v) for v in values) / len(values)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def generation_health(metrics: dict) -> float:
    """Map generation diagnostics to 0-100 against explicit project quality targets.

    EOS reaches full credit at the formal 60% target. Repetition receives full credit at zero
    and zero credit at >=20% repeated-trigram fraction. UTF-8 remains a hard quality signal.
    """
    eos = _clamp(float(metrics["free_eos_pct"]) / 60.0 * 100.0)
    repetition = _clamp(100.0 * (1.0 - float(metrics["free_repeated_trigram_pct"]) / 20.0))
    utf8 = _clamp(float(metrics["utf8_pct"]))
    return 0.30 * eos + 0.30 * repetition + 0.40 * utf8


NIS2_REQUIRED_FILES = (
    "free.json", "v3.json", "v4.json", "rawmath.json",
    "targeted.json", "v5.json", "nis2_chat.json",
)
NIS2_CHAT_KEYS = frozenset((
    "instruction_following_pct", "dialogue_state_pct", "truthfulness_pct",
    "robustness_pct", "compositional_reasoning_pct",
))


def _validate_chat_evidence(chat: dict) -> None:
    """Verify provenance and completeness before incorporating frozen chat evidence."""
    if not isinstance(chat, dict):
        raise ValueError("NIS2 frozen chat report must be a JSON object")
    if chat.get("benchmark") != "nis2-chat-core-v1":
        raise ValueError("NIS2 frozen chat suite name mismatch")
    if chat.get("foreign_model_or_tool_assistance") is not False:
        raise ValueError("NIS2 forbids model assistance and tools")
    if chat.get("used_for_checkpoint_selection") is not False:
        raise ValueError("NIS2 holdout may not be used for checkpoint selection")
    for name in ("checkpoint_sha256", "suite_sha256"):
        value = chat.get(name)
        if not isinstance(value, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError("NIS2 frozen chat evidence missing SHA-256: " + name)
    scores = chat.get("scores")
    if not isinstance(scores, dict) or NIS2_CHAT_KEYS - scores.keys():
        raise ValueError("NIS2 frozen chat scores are incomplete")
    for key in NIS2_CHAT_KEYS:
        _clamp(scores[key])


def _validate_checkpoint_lineage(directory: Path, chat: dict) -> None:
    """Every NIS2 source report must belong to exactly the same raw checkpoint."""
    origin = chat["checkpoint_sha256"].lower()
    for filename in NIS2_REQUIRED_FILES:
        if filename == "nis2_chat.json":
            continue
        item = _load(directory / filename)
        if not isinstance(item, dict):
            raise ValueError("NIS2 report is not an object: " + filename)
        digest = item.get("checkpoint_sha256")
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError("NIS2 report is missing checkpoint lineage: " + filename)
        if digest.lower() != origin:
            raise ValueError("NIS2 mixed checkpoints forbidden: " + filename)


def nis2_evidence_status(report_dir: str | Path) -> dict:
    """Return a truthful score status: never infer a score from partial evidence.

    Checks the frozen chat provenance and mandatory report format. Structural
    checks cannot prove benchmark independence or the authenticity of a
    checkpoint: that requires a separate review of run IDs and hashes.
    """
    directory = Path(report_dir)
    missing = sorted(name for name in NIS2_REQUIRED_FILES if not (directory / name).is_file())
    invalid = []
    if "nis2_chat.json" not in missing:
        try:
            frozen = _load(directory / "nis2_chat.json")
            _validate_chat_evidence(frozen)
        except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError):
            invalid.append("nis2_chat")
        else:
            if not missing:
                try:
                    _validate_checkpoint_lineage(directory, frozen)
                except (ValueError, TypeError, KeyError, OSError, json.JSONDecodeError):
                    invalid.append("checkpoint_lineage")
    if missing or invalid:
        return {
            "version": SCORE_VERSION, "status": "not_measured",
            "nis2_score": None, "can_report_nis2": False,
            "missing_files": missing, "invalid_evidence": invalid,
            "methodology": "raw-model only; NIS 1.0 and Qwen-based product metrics excluded",
        }
    try:
        report = score_report_dir(directory)
    except (ValueError, KeyError, TypeError, OSError, json.JSONDecodeError) as exc:
        return {
            "version": SCORE_VERSION, "status": "not_measured",
            "nis2_score": None, "can_report_nis2": False,
            "missing_files": [], "invalid_evidence": ["required benchmark metrics incomplete"],
            "methodology": "raw-model only; NIS 1.0 and Qwen-based product metrics excluded",
        }
    return {
        "version": SCORE_VERSION, "status": "evidence_complete_unreviewed",
        "nis2_score": report["score"], "can_report_nis2": True,
        "missing_files": [], "invalid_evidence": [],
        "bottleneck": report["bottleneck"],
        "methodology": "structurally complete raw-model reports; run identity and heldout independence require review",
    }


def components_from_reports(report_dir: str | Path) -> dict[str, float]:
    report_dir = Path(report_dir)
    chat_path = report_dir / "nis2_chat.json"
    if not chat_path.is_file():
        raise FileNotFoundError(f"NIS 2.0 requires {chat_path}; partial scores are forbidden")

    free = _load(report_dir / "free.json")
    v3 = _load(report_dir / "v3.json")
    v4 = _load(report_dir / "v4.json")
    raw = _load(report_dir / "rawmath.json")
    old = nis_v1(direct_metrics(free, v3, v4, raw))["components"]
    metrics = extract(report_dir)
    chat_report = _load(chat_path)
    _validate_chat_evidence(chat_report)
    _validate_checkpoint_lineage(report_dir, chat_report)
    chat = chat_report["scores"]

    math_generalization = (
        0.45 * float(metrics["rawmath_exact_pct"])
        + 0.30 * float(metrics["rawmath_transfer_pct"])
        + 0.25 * float(metrics["rawmath_ood_pct"])
    )
    code = 0.60 * float(old["code"]) + 0.40 * float(metrics["targeted_code_pct"])
    reasoning = (
        0.50 * float(old["logic_induction"])
        + 0.25 * float(metrics["targeted_reasoning_pct"])
        + 0.25 * float(chat["compositional_reasoning_pct"])
    )
    bilingual = 0.60 * float(old["translation"]) + 0.40 * float(metrics["targeted_de_pct"])
    context_dialogue = (
        0.45 * float(old["context_use"])
        + 0.20 * float(metrics["v5_single_pct"])
        + 0.15 * float(metrics["v5_multi_pct"])
        + 0.20 * float(chat["dialogue_state_pct"])
    )
    truth_robust = _mean(float(chat["truthfulness_pct"]), float(chat["robustness_pct"]))

    components = {
        "language_knowledge": float(old["language_knowledge"]),
        "math_generalization": math_generalization,
        "code": code,
        "science_domain": float(old["science_focus"]),
        "reasoning_compositional": reasoning,
        "bilingual_translation": bilingual,
        "context_dialogue": context_dialogue,
        "instruction_following": float(chat["instruction_following_pct"]),
        "truthfulness_robustness": truth_robust,
        "generation_health": generation_health(metrics),
    }
    return {k: _clamp(v) for k, v in components.items()}


def score_from_components(components: dict[str, float]) -> dict:
    missing = sorted(set(WEIGHTS) - set(components))
    extra = sorted(set(components) - set(WEIGHTS))
    if missing or extra:
        raise ValueError(f"NIS2 component mismatch: missing={missing}, extra={extra}")
    if not math.isclose(sum(WEIGHTS.values()), 1.0, abs_tol=1e-12):
        raise AssertionError("NIS2 weights must sum to 1")

    values = {k: _clamp(components[k]) for k in WEIGHTS}
    arithmetic = sum(WEIGHTS[k] * values[k] for k in WEIGHTS)
    balance = math.exp(sum(WEIGHTS[k] * math.log(max(5.0, values[k])) for k in WEIGHTS))
    score = 0.85 * arithmetic + 0.15 * balance
    bottleneck_key = min(values, key=values.get)
    return {
        "version": SCORE_VERSION,
        "score": score,
        "arithmetic_component": arithmetic,
        "balance_component": balance,
        "bottleneck": {"component": bottleneck_key, "value": values[bottleneck_key]},
        "weights": WEIGHTS,
        "components": values,
        "coverage_pct": 100.0,
        "definition": "Project-local raw-model score; not IQ and not cross-model standardized. 85% weighted mean + 15% balance term.",
    }


def score_report_dir(report_dir: str | Path) -> dict:
    return score_from_components(components_from_reports(report_dir))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("report_dir")
    ap.add_argument("--out")
    args = ap.parse_args()
    result = score_report_dir(args.report_dir)
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
