"""Sealed final holdout for the verified-step comparison.

Selection, the phase-2 training gate and the experiment decision must not
import this module. It returns item ids only to a caller that imports it
directly. Aggregate scoring below drops ids, prompts and completions.

The items are disjoint from retention and from the development transfer
slice. They are not a release suite and not a measured score.
"""
from __future__ import annotations

from hashlib import sha256
import json

from scripts.neral_reasoning_train_candidate import candidate
from scripts.neral_v6_measurement_slices import (
    MEASUREMENT_SEED, PROTOCOL_ID, RETENTION_PAIR_COUNT, grade_exact,
    safe_integer_expression, slice_sha256,
)

HOLDOUT_FAMILIES = (
    "odd_median",
    "population_bitcount",
    "distinct_values",
    "parenthesized_product_sum",
)
_MEDIANS = (
    (4, 1, 9),
    (8, 2, 5, 1, 7),
    (3, 9, 4, 1, 6, 2, 8),
    (12, 4, 7, 9, 1),
    (6, 2, 11, 5, 3, 14, 8),
    (20, 1, 15, 4, 9),
    (2, 10, 6, 3, 8, 1, 7),
    (5, 13, 4, 9, 1, 8, 2),
)
_BITCOUNTS = (1, 2, 7, 16, 31, 42, 100, 255)
_DISTINCT = (
    (1, 1, 1),
    (2, 3, 2, 4),
    (5, 5, 6, 6, 7),
    (8, 1, 8, 1, 8),
    (4, 9, 4, 2, 9, 2),
    (1, 2, 3, 4, 5, 1),
    (7, 7, 7, 8),
    (3, 1, 4, 1, 5, 9, 2),
)
_EXPRESSIONS = (
    "((2 + 3) * (4 + 1))",
    "((1 + 2) + (3 * 4))",
    "((6 * 2) + (3 * 3))",
    "((8 + 1) * (2 + 2))",
    "(((2 + 2) + 3) * 4)",
    "((5 * 5) + (1 + 2))",
    "((7 + 3) * (1 + 1))",
    "(((4 * 2) + 1) + 6)",
)


def _case(stratum: str, family: str, index: int, prompt: str, answer: str, **extra) -> dict:
    row = {
        "id": f"locked-holdout-{stratum}-{family}-{index:02d}",
        "role": "locked_holdout",
        "stratum": stratum,
        "family": family,
        "index": index,
        "prompt": prompt,
        "answer": str(answer),
        "source": PROTOCOL_ID,
        "required_prompt_protocol": "matched_fewshot",
        "decoding": "greedy",
        "max_new_tokens": 160,
    }
    row.update(extra)
    return row


def _unseen_case(family: str, index: int) -> dict:
    if family == "odd_median":
        values = _MEDIANS[index]
        ordered = tuple(sorted(values))
        if len(ordered) % 2 != 1 or len(ordered) != len(set(ordered)):
            raise ValueError("Median list must be odd and distinct")
        answer = ordered[len(ordered) // 2]
        prompt = (f"The odd-length list {list(values)} has distinct entries. "
                  f"After sorting it ascending, which entry sits in the middle?")
    elif family == "population_bitcount":
        number = _BITCOUNTS[index]
        answer = number.bit_count()
        prompt = (f"Write {number} in binary and count the one-bits. "
                  f"Return that population count as an integer.")
    elif family == "distinct_values":
        values = _DISTINCT[index]
        answer = len(set(values))
        prompt = (f"How many different numbers appear in {list(values)}, "
                  f"ignoring repeated copies of the same number?")
    elif family == "parenthesized_product_sum":
        expression = _EXPRESSIONS[index]
        answer = safe_integer_expression(expression)
        prompt = (f"Compute the integer value of this fully parenthesized "
                  f"sum and product: {expression}. Use the parentheses as written.")
    else:
        raise ValueError("Unknown sealed family")
    return _case("unseen_family", family, index, prompt, answer)


def locked_holdout_cases() -> list[dict]:
    """Item ids for the untouched final slice. Do not import this from a gate."""
    cases = []
    rows, proofs = candidate(MEASUREMENT_SEED)
    for proof, row in zip(proofs, rows):
        family, index_text, variant_text = proof["id"].split("-")
        pair_index = int(index_text)
        variant = int(variant_text)
        if pair_index < RETENTION_PAIR_COUNT:
            continue
        cases.append(_case(
            "counterexample", family, pair_index * 2 + variant,
            row["messages"][1]["content"], proof["answer"],
            pair_index=pair_index, variant=variant,
            measurement_proof_id=f"{MEASUREMENT_SEED}:{proof['id']}",
        ))
    for family in HOLDOUT_FAMILIES:
        for index in range(8):
            cases.append(_unseen_case(family, index))
    ids = [case["id"] for case in cases]
    prompts = [case["prompt"] for case in cases]
    if (len(cases) != 64 + 32 or len(ids) != len(set(ids))
            or len(prompts) != len(set(prompts))):
        raise ValueError("Locked holdout is not a stable disjoint slice")
    return cases


def holdout_commitment() -> dict:
    """Counts and a hash. No item id, prompt or answer is included."""
    cases = locked_holdout_cases()
    digest = slice_sha256(cases)
    return {
        "schema": 1,
        "protocol_id": PROTOCOL_ID,
        "kind": "scorer_protocol_not_a_learning_change",
        "role": "locked_holdout_commitment",
        "count": len(cases),
        "sha256": digest,
        "family_counts": {
            family: sum(case["family"] == family for case in cases)
            for family in ("ordered_price", "segmented_fuel", *HOLDOUT_FAMILIES)
        },
        "ids_included": False,
        "prompts_included": False,
        "training_authorized": False,
        "claimed_score_gain": False,
        "release_validated": False,
        "consumed_for_selection": False,
    }


def sealed_holdout_counts(completions: list[dict]) -> dict:
    """Score aligned completions and return only the four outcome counts."""
    cases = locked_holdout_cases()
    if len(completions) != len(cases):
        raise ValueError("Sealed completions do not match the locked holdout")
    kinds = [grade_exact(case["answer"], row.get("text", ""), row.get("complete"))
             for case, row in zip(cases, completions)]
    counts = {name: kinds.count(name) for name in (
        "strict_correct", "wrong_readable_number", "unreadable", "incomplete")}
    if sum(counts.values()) != len(cases):
        raise ValueError("Sealed partition does not cover the holdout")
    payload = json.dumps(counts, sort_keys=True)
    return {
        "role": "locked_holdout",
        "total": len(cases),
        **counts,
        "case_sha256": slice_sha256(cases),
        "count_sha256": sha256(payload.encode("utf8")).hexdigest(),
        "ids_included": False,
        "training_authorized": False,
        "claimed_score_gain": False,
        "measurement_change": "scorer_protocol_not_a_learning_change",
    }
