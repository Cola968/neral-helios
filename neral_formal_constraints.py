"""Bounded, deterministic finite-domain constraint solver for explicit schemas.

Not a natural-language classifier. Only fully declared NERAL_CONSTRAINTS_JSON
requests enter this route. No code generation, eval, network or model call.
"""
from __future__ import annotations

from itertools import product
import json
import math

PREFIX = "NERAL_CONSTRAINTS_JSON: "
MAX_INPUT_BYTES = 8192
MAX_ASSIGNMENTS = 100000
MAX_VARIABLES = 8
MAX_RULES = 32
_OPERATORS = {"==", "!=", "<", "<=", ">", ">="}


class InvalidFormalConstraints(ValueError):
    """Invalid or excessive schema; never includes the user's submitted text."""


def _scalar(value):
    return type(value) in (int, bool) and (
        type(value) is bool or abs(value) <= 1_000_000
    )


def _satisfies(left, op, right):
    if op == "==":
        return left == right
    if op == "!=":
        return left != right
    if op == "<":
        return left < right
    if op == "<=":
        return left <= right
    if op == ">":
        return left > right
    if op == ">=":
        return left >= right
    raise InvalidFormalConstraints("Unsupported operator")


def try_solve_formal_constraints(question: str) -> int | None:
    """Return exact count or 0/1 satisfiability, None for non-schema input.

    Explicit schema:
      NERAL_CONSTRAINTS_JSON: {
        "variables":{"A":[1,2,3],"B":[1,2,3]},
        "rules":[{"left":"A","op":"<","right":"B"}],
        "query":"count"
      }
    Rule RHS is exactly one of 'right' (declared variable) and 'value'
    (JSON integer / Boolean). Optional 'all_different' groups use names.
    """
    if type(question) is not str or not question.startswith(PREFIX):
        return None
    if len(question.encode("utf-8")) > MAX_INPUT_BYTES:
        raise InvalidFormalConstraints("Schema exceeds size limit")
    try:
        payload = json.loads(question[len(PREFIX):])
    except (TypeError, ValueError):
        raise InvalidFormalConstraints("Invalid JSON") from None
    if (type(payload) is not dict
            or not {"variables", "rules", "query"} <= set(payload)
            or set(payload) - {"variables", "rules", "query", "all_different"}):
        raise InvalidFormalConstraints("Invalid top-level schema")
    variables = payload["variables"]
    rules = payload["rules"]
    distinct = payload.get("all_different", [])
    query = payload["query"]
    if (type(variables) is not dict or not 1 <= len(variables) <= MAX_VARIABLES
            or type(rules) is not list or len(rules) > MAX_RULES
            or type(distinct) is not list or len(distinct) > MAX_RULES
            or query not in ("count", "satisfiable")):
        raise InvalidFormalConstraints("Invalid schema dimensions")
    for name, domain in variables.items():
        if (type(name) is not str or not name.isidentifier()
                or not 1 <= len(name) <= 32
                or type(domain) is not list or not 1 <= len(domain) <= 12
                or not all(_scalar(value) for value in domain)
                or len({(type(v).__name__, v) for v in domain}) != len(domain)
                or len({type(value) for value in domain}) != 1):
            raise InvalidFormalConstraints("Invalid variable domain")
    names = tuple(sorted(variables))
    max_states = math.prod(len(variables[name]) for name in names)
    if max_states > MAX_ASSIGNMENTS:
        raise InvalidFormalConstraints("Search space exceeds hard cap")
    prepared = []
    for rule in rules:
        if (type(rule) is not dict or "left" not in rule
                or "op" not in rule or rule["op"] not in _OPERATORS
                or (set(rule) not in ({"left", "op", "right"},
                                      {"left", "op", "value"}))):
            raise InvalidFormalConstraints("Malformed relation")
        left = rule["left"]
        if type(left) is not str or left not in variables:
            raise InvalidFormalConstraints("Unknown lhs variable")
        if "right" in rule:
            right = rule["right"]
            if type(right) is not str or right not in variables:
                raise InvalidFormalConstraints("Unknown rhs variable")
            right_type = type(variables[right][0])
            rhs_var = right
            rhs_value = None
        else:
            rhs_var = None
            rhs_value = rule["value"]
            if not _scalar(rhs_value):
                raise InvalidFormalConstraints("Invalid comparison value")
            right_type = type(rhs_value)
        left_type = type(variables[left][0])
        if (left_type is not right_type
                or (left_type is bool and rule["op"] not in ("==", "!="))):
            raise InvalidFormalConstraints("Incompatible comparison types")
        prepared.append((left, rule["op"], rhs_var, rhs_value))
    seen_groups = []
    for group in distinct:
        if (type(group) is not list or not 2 <= len(group) <= MAX_VARIABLES
                or any(type(name) is not str or name not in variables
                       for name in group)
                or len(set(group)) != len(group)
                or len({type(variables[name][0]) for name in group}) != 1):
            raise InvalidFormalConstraints("Invalid all_different group")
        seen_groups.append(tuple(group))
    count = 0
    for values in product(*(variables[name] for name in names)):
        assignment = dict(zip(names, values))
        if not all(
            _satisfies(assignment[left], op,
                       assignment[rhs_var] if rhs_var is not None else rhs_value)
            for left, op, rhs_var, rhs_value in prepared
        ):
            continue
        if any(len({assignment[name] for name in group}) != len(group)
               for group in seen_groups):
            continue
        count += 1
        if query == "satisfiable":
            return 1
    return count if query == "count" else 0
