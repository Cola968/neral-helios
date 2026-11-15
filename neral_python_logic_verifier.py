"""Strict independent finite-domain source checker for NERAL 1.1.

No model-generated code is executed here. The oracle parses EVERY sentence
in a deliberately small, unambiguous English grammar and independently counts
all assignments. Unsupported wording is rejected. This is not a general
natural-language verifier and cannot establish correctness outside its grammar.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations, product
import re

MAX_SOURCE_BYTES = 2400
MAX_CLAUSES = 16
MAX_ASSIGNMENTS = 120_000
MAX_CODE_BYTES = 4096


@dataclass(frozen=True)
class BoundedLogicTask:
    family: str
    variables: int
    colors_or_max: int
    clauses: tuple[tuple, ...]


def _match(pattern: str, text: str):
    return re.fullmatch(pattern, text, flags=re.IGNORECASE | re.ASCII)


def _positive_int(value: str) -> int:
    # Reject non-canonical integers and absurdly long numeric literals.
    if len(value) > 6 or not re.fullmatch(r"(?:0|[1-9][0-9]{0,5})", value):
        raise ValueError("invalid_integer")
    return int(value)


def parse_bounded_logic_source(source: str) -> BoundedLogicTask | None:
    """Require exact full sentence coverage, range validity and search caps.

    Return None on *any* unrecognized instruction, additional sentence,
    invalid reference, unsupported family, or oversized enumeration.
    """
    if (type(source) is not str or not 20 <= len(source.encode("utf8")) <= MAX_SOURCE_BYTES
            or not source.endswith(".") or "\n" in source or "\r" in source):
        return None
    sentences = source[:-1].split(". ")
    if not 2 <= len(sentences) <= MAX_CLAUSES + 1:
        return None
    header, body = sentences[0], sentences[1:]
    kind, n, k = None, None, None
    header_color = _match(
        r"Count valid colorings of a chain of ([0-9]+) positions with ([0-9]+) colors",
        header)
    header_perm = _match(r"Count permutations of numbers 1 through ([0-9]+)", header)
    header_tuple = _match(
        r"Count ([0-9]+)-tuples with entries from 0 through ([0-9]+)", header)
    try:
        if header_color:
            kind = "color_chain"
            n, k = map(_positive_int, header_color.groups())
            if not 2 <= n <= 10 or not 2 <= k <= 8 or k ** n > MAX_ASSIGNMENTS:
                return None
        elif header_perm:
            kind = "permutation"
            n = _positive_int(header_perm.group(1))
            k = n
            if not 2 <= n <= 8:
                return None
        elif header_tuple:
            kind = "modular_tuple"
            n, k = map(_positive_int, header_tuple.groups())
            if not 2 <= n <= 10 or not 1 <= k <= 9 or (k + 1) ** n > MAX_ASSIGNMENTS:
                return None
        else:
            return None

        rules = []
        for sentence in body:
            rule = None
            if kind == "color_chain":
                if sentence.lower() == "adjacent positions differ":
                    rule = ("adjacent_diff",)
                m = _match(r"Position ([0-9]+) has color ([0-9]+)", sentence)
                if m:
                    position, color = map(_positive_int, m.groups())
                    if not (1 <= position <= n and 0 <= color < k):
                        return None
                    rule = ("color_is", position - 1, color)
                m = _match(r"Position ([0-9]+) is not color ([0-9]+)", sentence)
                if m:
                    position, color = map(_positive_int, m.groups())
                    if not (1 <= position <= n and 0 <= color < k):
                        return None
                    rule = ("color_not", position - 1, color)
                m = _match(r"Positions ([0-9]+) and ([0-9]+) differ", sentence)
                if m:
                    a, b = map(_positive_int, m.groups())
                    if not (1 <= a <= n and 1 <= b <= n and a != b):
                        return None
                    rule = ("positions_diff", a - 1, b - 1)
            elif kind == "permutation":
                m = _match(
                    r"Number ([0-9]+) is (before|after) number ([0-9]+)", sentence)
                if m:
                    a, b = _positive_int(m.group(1)), _positive_int(m.group(3))
                    if not (1 <= a <= n and 1 <= b <= n and a != b):
                        return None
                    rule = ("before" if m.group(2).lower() == "before" else "after", a, b)
                m = _match(r"Number ([0-9]+) is (not )?(first|last)", sentence)
                if m:
                    a = _positive_int(m.group(1))
                    if not 1 <= a <= n:
                        return None
                    rule = ("not_" if m.group(2) else "") + m.group(3).lower(), a
                m = _match(
                    r"Numbers ([0-9]+) and ([0-9]+) are not adjacent", sentence)
                if m:
                    a, b = map(_positive_int, m.groups())
                    if not (1 <= a <= n and 1 <= b <= n and a != b):
                        return None
                    rule = ("not_adjacent", a, b)
            elif kind == "modular_tuple":
                m = _match(
                    r"Their sum is congruent to ([0-9]+) modulo ([0-9]+)", sentence)
                if m:
                    residue, modulus = map(_positive_int, m.groups())
                    if not 2 <= modulus <= 100 or not 0 <= residue < modulus:
                        return None
                    rule = ("sum_mod", residue, modulus)
                if sentence.lower() == "all entries are distinct":
                    rule = ("distinct",)
                m = _match(r"Exactly ([0-9]+) entries are even", sentence)
                if m:
                    count = _positive_int(m.group(1))
                    if not 0 <= count <= n:
                        return None
                    rule = ("even_count", count)
            if rule is None:
                return None
            rules.append(rule)
        # For chains, an explicit adjacency clause is mandatory: the word
        # "chain" alone never silently supplies a mathematical condition.
        if kind == "color_chain" and ("adjacent_diff",) not in rules:
            return None
        if len(set(rules)) != len(rules):
            return None
        return BoundedLogicTask(kind, n, k, tuple(rules))
    except (ValueError, TypeError, OverflowError):
        return None


def _satisfies(assignment: tuple[int, ...], clause: tuple) -> bool:
    op = clause[0]
    if op == "adjacent_diff":
        return all(a != b for a, b in zip(assignment, assignment[1:]))
    if op == "color_is":
        return assignment[clause[1]] == clause[2]
    if op == "color_not":
        return assignment[clause[1]] != clause[2]
    if op == "positions_diff":
        return assignment[clause[1]] != assignment[clause[2]]
    if op == "before":
        return assignment.index(clause[1]) < assignment.index(clause[2])
    if op == "after":
        return assignment.index(clause[1]) > assignment.index(clause[2])
    if op == "first":
        return assignment[0] == clause[1]
    if op == "last":
        return assignment[-1] == clause[1]
    if op == "not_first":
        return assignment[0] != clause[1]
    if op == "not_last":
        return assignment[-1] != clause[1]
    if op == "not_adjacent":
        return abs(assignment.index(clause[1]) - assignment.index(clause[2])) != 1
    if op == "sum_mod":
        return sum(assignment) % clause[2] == clause[1]
    if op == "distinct":
        return len(set(assignment)) == len(assignment)
    if op == "even_count":
        return sum(v % 2 == 0 for v in assignment) == clause[1]
    raise ValueError("unrecognized_trusted_rule")


def independently_count(task: BoundedLogicTask) -> int:
    """Exhaust the full bounded assignment space with EVERY parsed rule."""
    if type(task) is not BoundedLogicTask:
        raise ValueError("invalid_trusted_task")
    if task.family == "permutation":
        assignments = permutations(range(1, task.variables + 1))
    elif task.family == "color_chain":
        assignments = product(range(task.colors_or_max), repeat=task.variables)
    elif task.family == "modular_tuple":
        assignments = product(range(task.colors_or_max + 1), repeat=task.variables)
    else:
        raise ValueError("unsupported_trusted_family")
    return sum(1 for values in assignments
               if all(_satisfies(values, rule) for rule in task.clauses))


def independent_python_certificate(
    *, source: str, code: str, output: str, canonical: dict | None,
) -> dict:
    """Full-input trusted certificate; never evaluate model-authored code.

    The Docker execution is performed separately by the existing TIR sandbox.
    Only a scalar result matching an independent full-space count may pass.
    Return fixed diagnostic codes only (no source/code/results on failure).
    """
    failed = {"verified": False, "reason": "source_unsupported"}
    if (type(code) is not str or not 1 <= len(code.encode("utf8")) <= MAX_CODE_BYTES
            or canonical is not None):
        return failed
    task = parse_bounded_logic_source(source)
    if task is None:
        return failed
    if (type(output) is not str or
            re.fullmatch(r"(?:0|[1-9][0-9]{0,11})", output, flags=re.ASCII) is None):
        return {"verified": False, "reason": "independent_proof_missing"}
    try:
        verified_answer = independently_count(task)
    except (ValueError, OverflowError, MemoryError):
        return {"verified": False, "reason": "independent_proof_missing"}
    if output != str(verified_answer):
        return {"verified": False, "reason": "independent_proof_missing"}
    return {
        "verified": True, "full_source": True,
        "all_conditions_checked": True,
        "answer": verified_answer,
        "proof_backend": "independent_enumeration",
    }
