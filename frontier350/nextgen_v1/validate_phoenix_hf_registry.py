"""Validate the Phoenix Hugging Face source-family registry against repo policy.

Read-only CPU check. It authorizes nothing and downloads nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from hf_source_policy50 import SOURCES

REGISTRY = HERE / "phoenix_hf_source_registry.json"
EXPECTED_RESERVED = {
    "openai/gsm8k",
    "google-research-datasets/mbpp",
    "deepmind/code_contests",
    "google/IFEval",
    "truthfulqa/truthful_qa",
    "google/boolq",
    "allenai/ai2_arc",
}


def validate(path: Path = REGISTRY) -> dict:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("schema") != 1:
        raise ValueError("unsupported Phoenix HF registry schema")
    reserved = set(map(str, obj.get("reserved_public_eval_dataset_ids", [])))
    if reserved != EXPECTED_RESERVED:
        raise ValueError(f"reserved public-eval registry drift: {sorted(reserved ^ EXPECTED_RESERVED)}")

    rows = obj.get("sources")
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty sources list required")
    seen = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("source row must be an object")
        dataset_id = str(row.get("dataset_id", ""))
        if not dataset_id or dataset_id in seen:
            raise ValueError(f"duplicate/empty dataset id: {dataset_id!r}")
        seen.add(dataset_id)
        allowed = row.get("source_family_allowed_by_global_policy")
        if type(allowed) is not bool:
            raise ValueError(f"{dataset_id}: explicit global-policy boolean required")
        if row.get("external_model_generated_training_rows") is True and allowed:
            raise ValueError(f"{dataset_id}: external-model-generated rows cannot be marked allowed")
        global_row = SOURCES.get(dataset_id)
        if allowed:
            if global_row is None or not global_row.allow_training:
                raise ValueError(f"{dataset_id}: Phoenix says allowed but global policy does not")
        elif global_row is not None and global_row.allow_training:
            raise ValueError(f"{dataset_id}: Phoenix blocks family but global policy currently allows it")
        if dataset_id in EXPECTED_RESERVED:
            raise ValueError(f"{dataset_id}: eval holdout must not appear in training-source rows")

    # Phoenix default human SFT direction: OASST2 preferred; OASST1 legacy.
    roles = {row["dataset_id"]: row.get("phoenix_role") for row in rows}
    if roles.get("OpenAssistant/oasst2") != "preferred_human_sft_family":
        raise ValueError("OASST2 must remain the preferred human SFT family")
    if roles.get("OpenAssistant/oasst1") == "preferred_human_sft_family":
        raise ValueError("OASST1 and OASST2 must not both be default SFT families")

    # FineMath must stay blocked until the central executable policy changes.
    fm = next(row for row in rows if row["dataset_id"] == "HuggingFaceTB/finemath")
    if fm["source_family_allowed_by_global_policy"]:
        raise ValueError("FineMath registry must be updated only together with explicit policy admission")

    return {
        "schema": obj["schema"],
        "sources": len(rows),
        "reserved_public_eval": len(reserved),
        "globally_allowed_families": sorted(
            row["dataset_id"] for row in rows if row["source_family_allowed_by_global_policy"]
        ),
        "blocked_or_research_only": sorted(
            row["dataset_id"] for row in rows if not row["source_family_allowed_by_global_policy"]
        ),
        "training_authorized_by_this_check": False,
    }


if __name__ == "__main__":
    print(json.dumps(validate(), indent=2, ensure_ascii=False))
