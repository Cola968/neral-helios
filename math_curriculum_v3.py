from __future__ import annotations

"""Neral Math Curriculum V3: split-safe curriculum contract, not a training run.

It composes existing project-generated RawMath/transfer infrastructure with future audited math
corpora. Public benchmarks (GSM8K etc.) are evaluation-only and never curriculum sources.
"""

from dataclasses import dataclass, asdict
import json

from holdout_registry_v3 import assert_dataset_ids_not_holdout


@dataclass(frozen=True)
class MathPhase:
    level: int
    name: str
    skills: tuple[str, ...]
    training_sources: tuple[str, ...]
    min_transfer_share: float
    min_language_replay_share: float
    gate: str


PHASES = (
    MathPhase(1, "fundamentals", ("integer arithmetic", "fractions", "percent", "signed numbers"), ("project_rawmath_generated",), 0.10, 0.30, "ID exact/NLL improve; language no regression"),
    MathPhase(2, "algebra_manipulation", ("simplification", "equivalent forms", "formula rearrangement"), ("project_math_v3_generated",), 0.15, 0.30, "equivalence exact + held-out algebra improve"),
    MathPhase(3, "equations", ("linear equations", "systems basics", "unknown isolation"), ("project_rawmath_generated", "project_math_v3_generated"), 0.20, 0.30, "equation exact + transfer NLL improve"),
    MathPhase(4, "functions", ("function evaluation", "slope", "linear/quadratic patterns", "trigonometric identities basics"), ("project_math_v3_generated",), 0.25, 0.30, "held-out function exact and transfer non-regression"),
    MathPhase(5, "multistep", ("two-to-four step calculations", "unit conversions", "word problems"), ("project_math_v3_generated",), 0.30, 0.30, "multi-step exact + reasoning dev improve"),
    MathPhase(6, "reasoning", ("constraint reasoning", "proof-like short chains", "logic", "probability"), ("project_math_v3_generated",), 0.35, 0.30, "reasoning + math transfer jointly improve"),
    MathPhase(7, "transfer", ("range-disjoint arithmetic", "range-disjoint algebra", "novel phrasing"), ("project_rawmath_transfer_generated",), 0.50, 0.30, "transfer exact grows OR transfer NLL >=0.5% better; ID protected"),
    MathPhase(8, "ood", ("unseen operand bands", "novel composition", "unseen templates"), ("project_math_v3_ood_generated",), 0.50, 0.30, "OOD exact/NLL must improve; ID and transfer protected"),
    MathPhase(9, "physics_math", ("formula rearrangement", "mechanics", "gravity", "waves", "basic astronomy calculations"), ("project_physics_math_generated",), 0.40, 0.35, "physics-math transfer + general math protected"),
)

PUBLIC_EVAL_NEVER_TRAIN = (
    "openai/gsm8k", "google-research-datasets/mbpp", "deepmind/code_contests",
    "google/IFEval", "truthfulqa/truthful_qa", "google/boolq", "allenai/ai2_arc",
)

SPLITS = {
    "train": {"selection_use": True, "frozen": False, "purpose": "gradient updates only"},
    "development": {"selection_use": True, "frozen": False, "purpose": "checkpoint selection / early stop; never gradients"},
    "frozen_test": {"selection_use": False, "frozen": True, "purpose": "post-training release evidence only"},
    "transfer_test": {"selection_use": False, "frozen": True, "purpose": "range/template-disjoint post-training generalization"},
    "ood_test": {"selection_use": False, "frozen": True, "purpose": "out-of-distribution post-training generalization"},
}


def validate_curriculum() -> dict:
    levels = [p.level for p in PHASES]
    if levels != list(range(1, 10)):
        raise ValueError("math levels must be contiguous 1..9")
    if any(not 0 <= p.min_transfer_share <= 1 for p in PHASES):
        raise ValueError("invalid transfer share")
    if any(p.min_language_replay_share < 0.25 for p in PHASES):
        raise ValueError("language replay floor is too low")
    if any(set(p.training_sources) & set(PUBLIC_EVAL_NEVER_TRAIN) for p in PHASES):
        raise ValueError("public evaluation source leaked into math curriculum")
    # Fail-closed sanity check against the authoritative HF holdout registry.
    try:
        assert_dataset_ids_not_holdout(PUBLIC_EVAL_NEVER_TRAIN)
    except ValueError:
        pass
    else:
        raise ValueError("public evaluation registry unexpectedly considered trainable")
    return {
        "levels": len(PHASES),
        "phases": [asdict(p) for p in PHASES],
        "splits": SPLITS,
        "public_eval_never_train": list(PUBLIC_EVAL_NEVER_TRAIN),
        "selection_principle": "ID-only improvement is insufficient; transfer/OOD evidence is mandatory for final math promotion",
    }


def recommended_mix(stage: str) -> dict:
    """Planning ratios only; actual runs remain evidence-gated."""
    if stage == "50m_repair":
        return {"general_language_replay": 0.35, "human_sft_replay": 0.20, "math_id": 0.15, "math_transfer": 0.20, "math_ood_physics": 0.10}
    if stage == "orion100_pretrain_math_lane":
        return {"general_language": 0.70, "educational_stem": 0.15, "admitted_math_web": 0.10, "project_generated_transfer_ood": 0.05}
    raise ValueError("unknown planning stage")


if __name__ == "__main__":
    print(json.dumps(validate_curriculum(), indent=2))
