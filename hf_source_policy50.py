"""Executable Hugging Face source policy for Neral-50M.

This file separates three concerns that must not be conflated:
1. sources already approved for Neral training,
2. future candidates that still need a dedicated contamination/license audit, and
3. public evaluation datasets that are permanently reserved from training.

The registry records only project policy and Hugging Face metadata. It downloads nothing and
never calls another model. Training builders may call ``assert_training_allowed`` before using
an upstream dataset so evaluation corpora cannot accidentally leak into future training.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict

REGISTRY_VERSION = 2


@dataclass(frozen=True)
class SourcePolicy:
    dataset_id: str
    role: str
    license: str
    content_origin: str
    allow_training: bool
    current_i1: bool
    reason: str


SOURCES = {
    # Current approved pretraining/SFT sources.
    "HuggingFaceFW/fineweb": SourcePolicy(
        "HuggingFaceFW/fineweb", "pretrain", "ODC-By-1.0", "web-human",
        True, True, "Approved English web pretraining source already used by Neral."),
    "HuggingFaceFW/fineweb-2": SourcePolicy(
        "HuggingFaceFW/fineweb-2", "pretrain", "ODC-By-1.0", "web-human",
        True, True, "Approved multilingual source; Neral currently uses the German subset."),
    "HuggingFaceFW/fineweb-edu": SourcePolicy(
        "HuggingFaceFW/fineweb-edu", "pretrain", "ODC-By-1.0", "web-human-filtered",
        True, True, "Approved English educational web source already represented in B3."),
    "OpenAssistant/oasst1": SourcePolicy(
        "OpenAssistant/oasst1", "sft-human", "Apache-2.0", "human-crowdsourced",
        True, False, "Approved legacy human assistant corpus; retained for reproducibility."),
    "OpenAssistant/oasst2": SourcePolicy(
        "OpenAssistant/oasst2", "sft-human", "Apache-2.0", "human-crowdsourced",
        True, True, "Preferred I1 human SFT source; synthetic rows are explicitly rejected."),

    # Public benchmarks are permanently reserved. They can be evaluated, never trained on.
    "openai/gsm8k": SourcePolicy(
        "openai/gsm8k", "public-eval-holdout", "MIT", "human-crowdsourced",
        False, False, "Reserve as an external math benchmark; training would invalidate future GSM8K claims."),
    "google-research-datasets/mbpp": SourcePolicy(
        "google-research-datasets/mbpp", "public-eval-holdout", "CC-BY-4.0", "human-crowdsourced/expert-verified",
        False, False, "Reserve as an external Python benchmark; do not train on prompts, solutions or tests."),
    "deepmind/code_contests": SourcePolicy(
        "deepmind/code_contests", "public-eval-holdout", "CC-BY-4.0", "human-found",
        False, False, "Reserve for future coding evaluation rather than specialization data."),
    "google/IFEval": SourcePolicy(
        "google/IFEval", "public-eval-holdout", "Apache-2.0", "human-authored-verifiable-instructions",
        False, False, "Reserve as an external instruction-following benchmark; never expose its prompts to training."),
    "truthfulqa/truthful_qa": SourcePolicy(
        "truthfulqa/truthful_qa", "public-eval-holdout", "Apache-2.0", "expert-generated",
        False, False, "Reserve as an external truthfulness/misconception benchmark."),
    "google/boolq": SourcePolicy(
        "google/boolq", "public-eval-holdout", "CC-BY-SA-3.0", "human-found/crowdsourced",
        False, False, "Reserve as an external reading-comprehension benchmark."),
    "allenai/ai2_arc": SourcePolicy(
        "allenai/ai2_arc", "public-eval-holdout", "CC-BY-SA-4.0", "human-found",
        False, False, "Reserve ARC as an external science/reasoning benchmark."),

    # Potentially useful, but deliberately blocked until a separate next-campaign audit.
    "HuggingFaceTB/finemath": SourcePolicy(
        "HuggingFaceTB/finemath", "future-pretrain-candidate", "ODC-By-1.0", "web-human-filtered",
        False, False, "Promising math corpus, but adding it mid-I1 would change the frozen curriculum; audit before I2."),
    "HuggingFaceTB/stack-edu": SourcePolicy(
        "HuggingFaceTB/stack-edu", "future-pretrain-candidate", "UNVERIFIED-MIXED", "code-found-filtered",
        False, False, "Useful code corpus, but file-level/source licensing must be resolved before Neral training."),
    "HuggingFaceTB/smollm-corpus": SourcePolicy(
        "HuggingFaceTB/smollm-corpus", "blocked-external-synthetic", "ODC-By-1.0", "mixed-including-model-generated",
        False, False, "Contains large model-generated components; excluded from the current own-data policy."),
    "allenai/tulu-3-sft-mixture": SourcePolicy(
        "allenai/tulu-3-sft-mixture", "blocked-mixed-sft", "MIXED", "human+machine-generated",
        False, False, "Mixture contains machine-generated and differently licensed/non-commercial subsets."),
    "codeparrot/codeparrot-clean": SourcePolicy(
        "codeparrot/codeparrot-clean", "future-pretrain-candidate", "UNVERIFIED", "github-code-found",
        False, False, "Dataset card does not provide a sufficiently clear training license policy for this project."),
}


ALLOWED_ROLES = {
    "pretrain", "sft-human", "public-eval-holdout", "future-pretrain-candidate",
    "blocked-external-synthetic", "blocked-mixed-sft",
}


def validate_registry():
    if not SOURCES:
        raise ValueError("empty Hugging Face source registry")
    for key, row in SOURCES.items():
        if key != row.dataset_id:
            raise ValueError(f"registry key mismatch: {key} != {row.dataset_id}")
        if row.role not in ALLOWED_ROLES:
            raise ValueError(f"{key}: unsupported role {row.role}")
        if not row.license or not row.content_origin or not row.reason:
            raise ValueError(f"{key}: incomplete policy metadata")
        if row.role == "public-eval-holdout" and row.allow_training:
            raise ValueError(f"{key}: evaluation holdout cannot be trainable")
        if row.current_i1 and not row.allow_training:
            raise ValueError(f"{key}: current I1 source must be training-approved")
    return {
        "version": REGISTRY_VERSION,
        "sources": len(SOURCES),
        "training_allowed": sorted(k for k, v in SOURCES.items() if v.allow_training),
        "public_eval_holdouts": sorted(k for k, v in SOURCES.items() if v.role == "public-eval-holdout"),
        "future_or_blocked": sorted(k for k, v in SOURCES.items() if not v.allow_training and v.role != "public-eval-holdout"),
    }


def assert_training_allowed(dataset_id: str) -> SourcePolicy:
    row = SOURCES.get(dataset_id)
    if row is None:
        raise ValueError(f"Unregistered Hugging Face dataset: {dataset_id}")
    if not row.allow_training:
        raise ValueError(f"Training source blocked by policy: {dataset_id} ({row.role}) - {row.reason}")
    return row


def assert_not_reserved_eval(dataset_ids):
    blocked = sorted(d for d in dataset_ids if d in SOURCES and SOURCES[d].role == "public-eval-holdout")
    if blocked:
        raise ValueError("Public evaluation datasets present in training inputs: " + ", ".join(blocked))
    return True


def public_registry():
    return {key: asdict(value) for key, value in sorted(SOURCES.items())}


if __name__ == "__main__":
    print(json.dumps({"audit": validate_registry(), "registry": public_registry()}, indent=2, ensure_ascii=False))
