"""Bounded, deterministic finite-domain constraint solver for explicit schemas.

Not a natural-language classifier. Only fully declared NERAL_CONSTRAINTS_JSON
requests enter this route. No code generation, eval, network or model call.
"""
from __future__ import annotations

from itertools import permutations, product
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


def _compile_boolean(expr, variables: dict, depth: int, node_budget: list[int]):
    """Validate a bounded, explicitly typed Boolean expression AST."""
    node_budget[0] += 1
    if depth > 4 or node_budget[0] > 64:
        raise InvalidFormalConstraints("Boolean AST exceeds bounds")
    if type(expr) is bool:
        return ("const", expr)
    if type(expr) is str:
        domain = variables.get(expr)
        if (type(domain) is not list or not domain
                or any(type(value) is not bool for value in domain)):
            raise InvalidFormalConstraints("Unknown or non-Boolean AST variable")
        return ("var", expr)
    if type(expr) is not dict or "op" not in expr:
        raise InvalidFormalConstraints("Malformed Boolean AST")
    op = expr["op"]
    if type(op) is not str or op not in (
            "and", "or", "xor", "not", "implies", "eq", "at_least"):
        raise InvalidFormalConstraints("Unknown Boolean operator")
    keys = {"op", "args", "min"} if op == "at_least" else {"op", "args"}
    if set(expr) != keys:
        raise InvalidFormalConstraints("Unexpected Boolean AST members")
    args = expr["args"]
    if type(args) is not list or not 1 <= len(args) <= MAX_VARIABLES:
        raise InvalidFormalConstraints("Invalid Boolean arguments")
    if op == "not" and len(args) != 1:
        raise InvalidFormalConstraints("NOT expects one argument")
    if op in ("implies", "eq") and len(args) != 2:
        raise InvalidFormalConstraints("Binary operator expects two arguments")
    if op in ("and", "or", "xor", "at_least") and len(args) < 2:
        raise InvalidFormalConstraints("Operator requires two or more arguments")
    if op == "at_least":
        minimum = expr["min"]
        if (type(minimum) is not int or not 1 <= minimum <= len(args)
                or any(type(arg) is not str for arg in args)
                or len(set(args)) != len(args)):
            raise InvalidFormalConstraints("Invalid Boolean cardinality")
    else:
        minimum = None
    compiled = tuple(
        _compile_boolean(arg, variables, depth + 1, node_budget)
        for arg in args
    )
    return ("op", op, compiled, minimum)


def _eval_boolean(node, assignment: dict) -> bool:
    """Pure, total Boolean evaluator. No Python evaluation of user text."""
    kind = node[0]
    if kind == "const":
        return node[1]
    if kind == "var":
        return assignment[node[1]]
    _, op, args, minimum = node
    if op == "not":
        return not _eval_boolean(args[0], assignment)
    if op == "and":
        return all(_eval_boolean(arg, assignment) for arg in args)
    if op == "or":
        return any(_eval_boolean(arg, assignment) for arg in args)
    if op == "xor":
        return sum(_eval_boolean(arg, assignment) for arg in args) % 2 == 1
    if op == "implies":
        return (not _eval_boolean(args[0], assignment)
                or _eval_boolean(args[1], assignment))
    if op == "eq":
        return (_eval_boolean(args[0], assignment)
                == _eval_boolean(args[1], assignment))
    if op == "at_least":
        return sum(_eval_boolean(arg, assignment) for arg in args) >= minimum
    raise InvalidFormalConstraints("Unrecognized compiled operator")


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
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise InvalidFormalConstraints("Duplicate JSON member")
            result[key] = value
        return result

    try:
        payload = json.loads(question[len(PREFIX):],
                             object_pairs_hook=unique_keys)
    except (TypeError, ValueError):
        raise InvalidFormalConstraints("Invalid JSON") from None
    if (type(payload) is not dict
            or not {"variables", "rules", "query"} <= set(payload)
            or set(payload) - {"variables", "rules", "query", "all_different", "bool_rules"}):
        raise InvalidFormalConstraints("Invalid top-level schema")
    variables = payload["variables"]
    rules = payload["rules"]
    distinct = payload.get("all_different", [])
    boolean_rules = payload.get("bool_rules", [])
    query = payload["query"]
    if (type(variables) is not dict or not 1 <= len(variables) <= MAX_VARIABLES
            or type(rules) is not list or len(rules) > MAX_RULES
            or type(distinct) is not list or len(distinct) > MAX_RULES
            or type(boolean_rules) is not list
            or len(rules) + len(distinct) + len(boolean_rules) > MAX_RULES
            or type(query) is not str
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
    prepared = []
    for rule in rules:
        if (type(rule) is not dict or "left" not in rule
                or "op" not in rule or type(rule["op"]) is not str
                or rule["op"] not in _OPERATORS
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
    node_budget = [0]
    prepared_boolean = tuple(
        _compile_boolean(expr, variables, 0, node_budget)
        for expr in boolean_rules
    )
    # A full all_different constraint collapses an N^N domain product to
    # N! complete assignments. Enable this general optimization only when
    # every variable has the SAME domain of exactly N distinct values.
    # No benchmark-text routing, probabilistic pruning or elevated cap.
    if max_states <= MAX_ASSIGNMENTS:
        candidate_values = product(*(variables[name] for name in names))
    else:
        full_distinct = any(set(group) == set(names) for group in seen_groups)
        common_domain = variables[names[0]]
        exhaustive_permutations = (
            full_distinct
            and len(common_domain) == len(names)
            and all(variables[name] == common_domain for name in names)
            and math.factorial(len(names)) <= MAX_ASSIGNMENTS
        )
        if not exhaustive_permutations:
            raise InvalidFormalConstraints("Search space exceeds hard cap")
        candidate_values = permutations(common_domain, len(names))
    count = 0
    for values in candidate_values:
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
        if not all(_eval_boolean(expr, assignment) for expr in prepared_boolean):
            continue
        count += 1
        if query == "satisfiable":
            return 1
    return count if query == "count" else 0
