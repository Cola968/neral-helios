"""Schema-only validation for a NEW private 180-case benchmark manifest.

No original prompts, answers, historical Blind200 artifacts, training data,
GPU calls, network access or dataset generation in this module.
"""
from __future__ import annotations

from collections import Counter
import re

MANIFEST_VERSION = "neral-frontier-blind180-v1"
CATEGORY_COUNTS = {"everyday_facts": 60, "math": 60, "code": 60}
_HEX = re.compile(r"[0-9a-f]{64}\Z", re.ASCII)
_ID = re.compile(r"(?:everyday_facts|math|code)-[0-9]{3}\Z", re.ASCII)
_KEYS = {"schema_version", "case_count", "data_origin", "cases"}
_CASE_KEYS = {"case_id", "category", "prompt_sha256", "grader_sha256"}


def validate_manifest(manifest: object) -> None:
    """Reject plaintext and invalid allocations; NOT a contamination audit."""
    if (type(manifest) is not dict or set(manifest) != _KEYS or
            manifest.get("schema_version") != MANIFEST_VERSION or
            manifest.get("data_origin") != "new_independent_private_blind180" or
            type(manifest.get("case_count")) is not int or
            manifest["case_count"] != 180 or
            type(manifest.get("cases")) is not list or
            len(manifest["cases"]) != 180):
        raise ValueError("Invalid Blind180 manifest metadata")
    seen_ids = set()
    seen_prompts = set()
    counts = Counter()
    for case in manifest["cases"]:
        if type(case) is not dict or set(case) != _CASE_KEYS:
            raise ValueError("Unexpected fields or plaintext in sealed case metadata")
        case_id, category = case["case_id"], case["category"]
        if (type(case_id) is not str or not _ID.fullmatch(case_id) or
                type(category) is not str or category not in CATEGORY_COUNTS or
                not case_id.startswith(category + "-")):
            raise ValueError("Invalid category or case id")
        if case_id in seen_ids:
            raise ValueError("Duplicate case id")
        seen_ids.add(case_id)
        counts[category] += 1
        for key in ("prompt_sha256", "grader_sha256"):
            digest = case[key]
            if type(digest) is not str or not _HEX.fullmatch(digest) or digest == "0" * 64:
                raise ValueError("Invalid sealed SHA-256")
        if case["prompt_sha256"] in seen_prompts:
            raise ValueError("Repeated prompt digest")
        seen_prompts.add(case["prompt_sha256"])
    if counts != Counter(CATEGORY_COUNTS):
        raise ValueError("Expected exactly 60 everyday, 60 math and 60 code")
