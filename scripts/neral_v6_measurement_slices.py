"""Frozen retention and transfer slices for the verified-step comparison.

CPU measurement protocol only. This module does not train, does not load
weights, and does not return locked-holdout item ids. Those live in
``scripts.neral_v6_locked_holdout``. A green test of these slices is not
a capability gain.

Assignment is a pure function of the constants below. Same-family retention
reuses the compositional holdout and a second counterexample seed. Transfer
uses eight families that are not in the verified-step training mixture and
not in the already consumed dev72 screens. Four of them were added after
the 32-item draft because a paired test on 32 items cannot see a gap
below about 19 percentage points.
"""
from __future__ import annotations

import ast
from hashlib import sha256
import json
import re

from scripts.neral_reasoning_train_candidate import candidate
from scripts.neral_v6_compositional import FAMILIES, PER_FAMILY, make_case

PROTOCOL_ID = "NERAL-v6-verified-step-measurement-slices-20261108-v1"
MEASUREMENT_SEED = 20261108
RETENTION_PAIR_COUNT = 16
TRAINING_COUNTEREXAMPLE_SEED = 20261107
CONSUMED_SCREEN_SOURCES = (
    "NERAL-v6-structural-new-dev-72-20261107-v1",
    "NERAL-v61-new-development-only",
)
TRANSFER_FAMILIES = (
    "inclusive_range_sum",
    "nesting_depth",
    "inversion_pairs",
    "running_maximum_changes",
    "digit_sum",
    "adjacent_ascents",
    "alternating_sum",
    "capped_sum",
)
ASSIGNMENT_RULE = (
    "retention composition: make_case(family, 'holdout', index) for every "
    "composition family and every holdout index. "
    "retention counterexample: candidate(seed=20261108) pair index 0..15, "
    "both variants, ordered_price and segmented_fuel. "
    "transfer: the eight TRANSFER_FAMILIES, index 0..7, parameters in this "
    "file. The last four families were added before any GPU comparison so "
    "a paired gap near 19 percentage points is not the smallest visible "
    "shift. "
    "locked holdout is a different module and is not built here: "
    "candidate(seed=20261108) pair index 16..31, plus four other families. "
    "Consumed dev72 screens are in none of the three slices. "
    "The cut depends only on family and index, not on a score."
)

_BRACKETS = (
    "()", "(())", "(()())", "((()))", "(()(()))",
    "(((()))())", "(()()(()))", "((()())())",
)
_INVERSIONS = (
    (4, 1, 3, 2),
    (5, 1, 4, 2, 3),
    (2, 6, 1, 5, 3),
    (8, 3, 7, 1, 4, 2),
    (1, 2, 3, 4),
    (9, 8, 7, 6, 5),
    (3, 1, 4, 2, 6, 5),
    (7, 2, 8, 1, 5, 3, 4),
)
_RUNNING = (
    (1, 3, 2, 5, 4),
    (4, 4, 4, 2),
    (2, 1, 3, 3, 6, 5),
    (9, 1, 2, 3),
    (5, 6, 6, 7, 1),
    (3, 8, 2, 8, 9),
    (1, 1, 1),
    (6, 2, 7, 3, 8, 4),
)
_DIGIT_SUMS = (19, 208, 67, 1000, 845, 314159, 42, 9001)
_ASCENTS = (
    (1, 3, 2, 4),
    (5, 4, 4, 6, 1),
    (2, 2, 2),
    (9, 1, 8, 2, 7),
    (1, 2, 3, 4, 5),
    (8, 7, 6, 5),
    (3, 5, 4, 6, 6, 8),
    (4, 1, 1, 2),
)
_ALTERNATING = (
    (4, 1, 3, 2),
    (5, 5),
    (8, 1, 2),
    (3, 9, 1, 4, 2),
    (6,),
    (2, 8, 3, 1),
    (7, 2, 7, 2, 7),
    (1, 4, 1, 4),
)
_CAPPED = (
    ((3, 8, 1), 4),
    ((2, 2, 2), 5),
    ((9, 1, 7), 3),
    ((4, 10, 6, 1), 6),
    ((8,), 2),
    ((1, 5, 5, 1), 5),
    ((7, 0, 4), 4),
    ((6, 6, 9, 2), 1),
)


def _json_bytes(cases: list[dict]) -> bytes:
    return "".join(
        json.dumps(case, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        for case in cases
    ).encode("utf8")


def slice_sha256(cases: list[dict]) -> str:
    return sha256(_json_bytes(cases)).hexdigest()


def _final_answer(text: str) -> str:
    if "FINAL_ANSWER:" in text:
        return text.split("FINAL_ANSWER:", 1)[1].strip()
    return text.strip()


def _case(role: str, stratum: str, family: str, index: int, prompt: str, answer: str,
          **extra) -> dict:
    if not prompt.strip() or not str(answer).strip():
        raise ValueError("Slice case is missing a prompt or an answer")
    row = {
        "id": f"{role}-{stratum}-{family}-{index:02d}",
        "role": role,
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


def _check_unique(cases: list[dict]) -> list[dict]:
    ids = [case["id"] for case in cases]
    prompts = [case["prompt"] for case in cases]
    if len(ids) != len(set(ids)) or len(prompts) != len(set(prompts)):
        raise ValueError("Slice ids or prompts are not unique")
    return cases


def spoken_parentheses(brackets: str) -> str:
    """Word form so a punctuation-stripping overlap check still sees the string."""
    words = []
    for char in brackets:
        if char == "(":
            words.append("open")
        elif char == ")":
            words.append("close")
        else:
            raise ValueError("Bracket string contains a non-bracket")
    return " ".join(words)


def nesting_depth(brackets: str) -> int:
    depth = maximum = 0
    for char in brackets:
        if char == "(":
            depth += 1
            maximum = max(maximum, depth)
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced brackets")
        else:
            raise ValueError("Bracket string contains a non-bracket")
    if depth != 0:
        raise ValueError("Unbalanced brackets")
    return maximum


def inversion_count(values: tuple[int, ...]) -> int:
    return sum(1 for i, left in enumerate(values)
               for right in values[i + 1:] if left > right)


def running_maximum_changes(values: tuple[int, ...]) -> int:
    if not values:
        raise ValueError("Empty running-maximum list")
    best = values[0]
    changes = 1
    for value in values[1:]:
        if value > best:
            best = value
            changes += 1
    return changes


def retention_cases() -> list[dict]:
    """Same families as the verified-step mixture, items absent from train/val."""
    cases = []
    for family in FAMILIES:
        for index in range(PER_FAMILY["holdout"]):
            row = make_case(family, "holdout", index)
            cases.append(_case(
                "retention", "composition", family, index,
                row["messages"][1]["content"],
                _final_answer(row["messages"][2]["content"]),
                template_shift=True,
            ))
    rows, proofs = candidate(MEASUREMENT_SEED)
    if MEASUREMENT_SEED == TRAINING_COUNTEREXAMPLE_SEED:
        raise ValueError("Measurement seed collides with the training seed")
    for proof, row in zip(proofs, rows):
        family, index_text, variant_text = proof["id"].split("-")
        pair_index = int(index_text)
        variant = int(variant_text)
        if pair_index >= RETENTION_PAIR_COUNT:
            continue
        if family not in ("ordered_price", "segmented_fuel") or variant not in (0, 1):
            raise ValueError("Unexpected measurement counterexample id")
        cases.append(_case(
            "retention", "counterexample", family, pair_index * 2 + variant,
            row["messages"][1]["content"], proof["answer"],
            pair_index=pair_index, variant=variant,
            measurement_proof_id=f"{MEASUREMENT_SEED}:{proof['id']}",
        ))
    if len(cases) != 100 + RETENTION_PAIR_COUNT * 2 * 2:
        raise ValueError("Retention slice changed size")
    return _check_unique(cases)


def _transfer_case(family: str, index: int) -> dict:
    if type(index) is not int or not 0 <= index < 8 or family not in TRANSFER_FAMILIES:
        raise ValueError("Unexpected transfer coordinate")
    if family == "inclusive_range_sum":
        low, high = 3 + index, 11 + 2 * index
        answer = (high - low + 1) * (low + high) // 2
        if index % 2 == 0:
            prompt = (f"Sum every whole number from {low} through {high}, "
                      f"counting both ends once. Return that one integer.")
        else:
            prompt = (f"Add the closed integer interval that begins at {low} "
                      f"and stops at {high}. Return that one integer.")
    elif family == "nesting_depth":
        brackets = _BRACKETS[index]
        answer = nesting_depth(brackets)
        spoken = spoken_parentheses(brackets)
        prompt = (f"Read these parenthesis marks from left to right: {spoken}. "
                  f"Report the greatest nesting depth reached.")
    elif family == "inversion_pairs":
        values = _INVERSIONS[index]
        answer = inversion_count(values)
        prompt = (f"For the list {list(values)}, count index pairs i < j "
                  f"whose left value is strictly greater than the right value.")
    elif family == "running_maximum_changes":
        values = _RUNNING[index]
        answer = running_maximum_changes(values)
        prompt = (f"Read {list(values)} from left to right. Count how often the "
                  f"largest value seen so far strictly increases, including the first item.")
    elif family == "digit_sum":
        number = _DIGIT_SUMS[index]
        answer = sum(int(digit) for digit in str(number))
        prompt = (f"Tally the decimal digits of the integer {number} by adding "
                  f"them. Reply with that digit total.")
    elif family == "adjacent_ascents":
        values = _ASCENTS[index]
        answer = sum(1 for pos in range(1, len(values)) if values[pos] > values[pos - 1])
        prompt = (f"Across the sequence {list(values)}, count positions where "
                  f"the entry is strictly above the entry just before it.")
    elif family == "alternating_sum":
        values = _ALTERNATING[index]
        answer = sum(value if pos % 2 == 0 else -value
                     for pos, value in enumerate(values))
        prompt = (f"Start positive, then alternate minus and plus through "
                  f"{list(values)}. The signed total is the result.")
    elif family == "capped_sum":
        values, cap = _CAPPED[index]
        answer = sum(min(value, cap) for value in values)
        prompt = (f"Each value in {list(values)} is replaced by whichever is "
                  f"smaller, the value or the ceiling {cap}. Add the replacements.")
    else:
        raise ValueError("Unexpected transfer family")
    return _case("transfer", "unseen_family", family, index, prompt, answer)


def transfer_cases() -> list[dict]:
    """Families that do not occur in the verified-step training mixture."""
    cases = [_transfer_case(family, index)
             for family in TRANSFER_FAMILIES for index in range(8)]
    if len(cases) != len(TRANSFER_FAMILIES) * 8:
        raise ValueError("Transfer slice changed size")
    return _check_unique(cases)


def public_manifest() -> dict:
    retention = retention_cases()
    transfer = transfer_cases()
    return {
        "schema": 1,
        "protocol_id": PROTOCOL_ID,
        "kind": "scorer_protocol_not_a_learning_change",
        "assignment_rule": ASSIGNMENT_RULE,
        "assignment_depends_on_scores": False,
        "retention_count": len(retention),
        "transfer_count": len(transfer),
        "retention_sha256": slice_sha256(retention),
        "transfer_sha256": slice_sha256(transfer),
        "retention_families": sorted({case["family"] for case in retention}),
        "transfer_families": list(TRANSFER_FAMILIES),
        "consumed_screens_excluded": list(CONSUMED_SCREEN_SOURCES),
        "required_prompt_protocol": "matched_fewshot",
        "decoding": "greedy",
        "max_new_tokens": 160,
        "locked_holdout_ids_visible_here": False,
        "training_authorized": False,
        "claimed_score_gain": False,
        "release_validated": False,
        "general_improvement_proven": False,
    }


_DIGIT = re.compile(r"\d")


def grade_exact(answer: str, text: str, complete: bool) -> str:
    """Exact final-answer match for these slices.

    This is not ``parse_answer`` and it does not rescore dev72. Composition
    answers are currency, mass, clock and fraction strings, so an integer
    parser would mis-grade them.
    """
    if complete is not True:
        return "incomplete"
    if not isinstance(text, str) or not text.strip():
        return "unreadable"
    body = text.strip()
    if "FINAL_ANSWER:" in body:
        body = body.split("FINAL_ANSWER:")[-1].strip()
    if body == answer:
        return "strict_correct"
    if _DIGIT.search(body):
        return "wrong_readable_number"
    return "unreadable"


def partition_visible(cases: list[dict], completions: list[dict]) -> dict:
    if len(cases) != len(completions):
        raise ValueError("Visible slice completions do not match the frozen cases")
    kinds = [grade_exact(case["answer"], row.get("text", ""), row.get("complete"))
             for case, row in zip(cases, completions)]
    counts = {name: kinds.count(name) for name in (
        "strict_correct", "wrong_readable_number", "unreadable", "incomplete")}
    if sum(counts.values()) != len(cases):
        raise ValueError("Visible partition does not cover the slice")
    return {
        "role": cases[0]["role"] if cases else "",
        "total": len(cases),
        **counts,
        "training_authorized": False,
        "claimed_score_gain": False,
        "measurement_change": "scorer_protocol_not_a_learning_change",
    }


def safe_integer_expression(expression: str) -> int:
    """Evaluate a fully parenthesized sum/product of nonnegative integers.

    No ``eval``. Only ``+``, ``*`` and parentheses are accepted. The locked
    holdout uses this helper; the development transfer slice does not.
    """
    tree = ast.parse(expression, mode="eval")

    def walk(node):
        if isinstance(node, ast.Expression):
            return walk(node.body)
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Mult)):
            left, right = walk(node.left), walk(node.right)
            return left + right if isinstance(node.op, ast.Add) else left * right
        if isinstance(node, ast.Constant) and type(node.value) is int and node.value >= 0:
            return node.value
        raise ValueError("Expression is outside the sealed arithmetic subset")

    return walk(tree)
