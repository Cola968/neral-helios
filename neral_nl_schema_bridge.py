"""Fail-closed dual NL-to-v2-schema agreement probe (NOT NL verification).

Two models agreeing on a constraint set does NOT prove that either preserved
the question's meaning. This module is diagnostic/opt-in and never supplies
a PR #537 verification signature. No default routing or model changes.
"""
from __future__ import annotations

from itertools import product
import json
import math
import re
from typing import Callable

from neral_typed_constraints_v2 import (
    PREFIX, MAX_INPUT_BYTES, MAX_ASSIGNMENTS, InvalidTypedConstraints,
    _type_predicate, _predicate_value, solve_typed_constraints,
)

_V2_COMPACT_GRAMMAR = (
    'Gib ausschließlich EIN kompaktes JSON-Objekt aus. Exakt vier Schlüssel: '
    '"version":2, "variables", "constraints", "query":"count" '
    '(alternativ query="satisfiable"). version ist die integer-Zahl 2, '
    'kein String. variables ist ein Objekt: '
    '"X":{"type":"int","domain":[0,1,2]} oder '
    '"B0":{"type":"bool","domain":[false,true]}. '
    'Integer-Domains benutzen Zahlen, Farben IMMER als Integer kodieren '
    '(rot=0, grün=1, blau=2). '
    'constraints ist eine Liste typisierter AST-Prädikate. '
    'TERM: {"var":"X"}, {"const":2}, {"sum":[TERM,TERM]}, '
    '{"mod":{"term":TERM,"by":2}}. '
    'Prädikate: {"compare":{"left":TERM,"op":"!=","right":TERM}}, '
    '{"all_different":["X","Y"]}, '
    '{"exactly":{"vars":["B0","B1"],"value":true,"count":1}}, '
    '{"not":P}, {"and":[P,P]}, {"or":[P,P]}, '
    '{"iff":[P,P]}, {"implies":[P,P]}. '
    'Ersetze TERM und P durch echte Objekte. '
    'Verwende kein "logic", keine "variable"-Listen, keine Freitextregeln. '
    'Alle Bedingungen explizit abbilden, insbesondere nicht/kein, '
    'Ungleichheit, Nachbarschaft und Quantoren. '
    'Keine redundanten Doppel-Constraints, keine Wiederholungsschleifen. '
    'Keine Markdown-Formatierung, keine Erläuterung.'
)
EXTRACT_QLORA_PROMPT = (
    'Extrahiere alle Nebenbedingungen direkt und möglichst kurz. '
    + _V2_COMPACT_GRAMMAR
)
EXTRACT_BASE_PROMPT = (
    'Formuliere die Originalaufgabe unabhängig als vollständige endliche '
    'Constraint-Formel. Prüfe Vorzeichen, Reihenfolge und Quantoren. '
    + _V2_COMPACT_GRAMMAR
)
MAX_QUESTION_CHARS = 8192
# Conservative lexical warning signals; NOT a semantic completeness proof.
_SIGNALS = (
    (r"\b(?:sum|summe)\b", frozenset({"sum"})),
    (r"\b(?:even|gerade|modulo|divisible|teilbar)\b", frozenset({"mod"})),
    (r"\b(?:distinct|unique|verschieden|unterschiedlich)\b",
     frozenset({"all_different", "!="})),
    (r"\b(?:before|after|vor|nach)\b", frozenset({"<", ">", "<=", ">="})),
    (r"\b(?:opposite|gegenteilig)\b", frozenset({"!="})),
    (r"\b(?:equivalent|äquivalent)\b", frozenset({"iff"})),
    (r"\b(?:exactly|genau)\b", frozenset({"exactly", "==", "all_different"})),
    (r"\b(?:not|nicht|no|kein|keine)\b", frozenset({"not", "!="})),
)


def _reject(status: str) -> dict:
    # Never log either model output or question text.
    return {
        "status": status, "schema_equivalent": False,
        "candidate_count": None, "verified": False,
        "model_generated": True, "safe_to_promote": False,
        "proof_backend": None,
    }


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError("duplicate")
        obj[key] = value
    return obj


def _parse_candidate(raw: str) -> dict:
    if type(raw) is not str or len(raw.encode("utf-8")) > MAX_INPUT_BYTES:
        raise ValueError("invalid size or type")
    try:
        obj = json.loads(raw, object_pairs_hook=_no_duplicate_keys)
    except (ValueError, UnicodeError, TypeError) as exc:
        raise ValueError("invalid JSON") from exc
    if type(obj) is not dict:
        raise ValueError("not a schema object")
    if set(obj) != {"version", "variables", "constraints", "query"}:
        raise ValueError("unknown schema fields")
    # The strict v2 interpreter is the sole authority on grammar/types/bounds.
    # The count is discarded: candidates must not become answers before proof.
    solve_typed_constraints(PREFIX + json.dumps(obj, ensure_ascii=False),
                            max_seconds=1.5)
    return obj


def _signature(doc: dict) -> tuple:
    # Treat domains as sets, with strict bool/int distinction. The same
    # coordinate names and domains are mandatory; no guessed renaming.
    return (
        doc["query"],
        tuple(sorted((
            name, spec["type"],
            tuple(sorted((type(v).__name__, repr(v)) for v in spec["domain"]))
        ) for name, spec in doc["variables"].items())),
    )


def _features(doc: dict) -> set[str]:
    found: set[str] = set()

    def walk(node: object):
        if type(node) is dict:
            for key, value in node.items():
                if key == "compare" and type(value) is dict:
                    found.add(value.get("op", ""))
                elif key in {"sum", "mod", "all_different", "exactly", "not",
                             "and", "or", "iff", "implies"}:
                    found.add(key)
                walk(value)
        elif type(node) is list:
            for item in node:
                walk(item)

    walk(doc["constraints"])
    return found



def _source_core_witnesses(question: str, schema: dict) -> str | None:
    """Conservative checks for narrowly recognizable source clauses only.

    These necessary witnesses do not prove complete semantic translation.
    Alternative equivalent ASTs can be rejected; an unrecognized phrase does
    NOT receive a completeness certificate. No benchmark IDs or answer routing.
    """
    import ast

    nodes = []
    def walk(value):
        if type(value) is dict:
            nodes.append(value)
            for nested in value.values():
                walk(nested)
        elif type(value) is list:
            for nested in value:
                walk(nested)
    walk(schema["constraints"])

    def compare(node):
        return node.get("compare") if type(node) is dict else None

    def var_id(term):
        if type(term) is dict and type(term.get("var")) is str:
            matched = re.search(r"(\d+)$", term["var"])
            return int(matched.group(1)) if matched else None
        return None

    def scalar(term):
        return term.get("const") if type(term) is dict else None

    def has_pair(op, a, b):
        for node in nodes:
            cmp_ = compare(node)
            if type(cmp_) is not dict or cmp_.get("op") != op:
                continue
            left, right = var_id(cmp_.get("left")), var_id(cmp_.get("right"))
            if (left, right) in ((a, b), (b, a)):
                return True
        return False

    edge_match = re.search(
        r"\bedges\s*(\[[^\]]{1,1200}\])\s+must\s+differ", question, re.I)
    if edge_match:
        try:
            edges = ast.literal_eval(edge_match.group(1))
            if (type(edges) is not list or not 1 <= len(edges) <= 32 or
                any(type(pair) not in (list, tuple) or len(pair) != 2 or
                    any(type(i) is not int or not 0 <= i < 10 for i in pair)
                    for pair in edges)):
                return "graph_edge_source_unsupported"
        except (SyntaxError, ValueError, TypeError, MemoryError):
            return "graph_edge_source_unsupported"
        if any(a == b or not has_pair("!=", a, b) for a, b in edges):
            return "graph_edge_witness_missing"
        if re.search(r"\bvertex\s+0\s+is\s+red\b", question, re.I):
            def fixed_red(node):
                item = compare(node)
                if type(item) is not dict or item.get("op") != "==":
                    return False
                l, r = item.get("left"), item.get("right")
                return ((var_id(l) == 0 and type(scalar(r)) is int and scalar(r) == 0)
                        or (var_id(r) == 0 and type(scalar(l)) is int and scalar(l) == 0))
            if not any(fixed_red(x) for x in nodes):
                return "graph_fixed_color_missing"

    binary = re.search(
        r"\bbinary\s+strings?\s+of\s+length\s+(\d+)\s+with\s+exactly\s+(\d+)\s+ones?\b",
        question, re.I)
    if binary:
        n, k = map(int, binary.groups())
        variables = schema["variables"]
        if (not 2 <= n <= 10 or not 0 <= k <= n or len(variables) != n
                or any(spec["type"] != "bool" for spec in variables.values())):
            return "binary_domain_mismatch"
        if not any(type(x.get("exactly")) is dict
                   and x["exactly"].get("count") == k
                   and x["exactly"].get("value") is True
                   and set(x["exactly"].get("vars", ())) == set(variables)
                   for x in nodes):
            return "binary_cardinality_witness_missing"
        if (re.search(r"\bfirst\s+two\s+bits\s+opposite\b", question, re.I)
                and not has_pair("!=", 0, 1)):
            return "binary_first_pair_witness_missing"
        if re.search(r"\bno\s+two\s+ones\s+exactly\s+two\s+positions\s+apart\b",
                     question, re.I):
            forbidden = set()
            for node in nodes:
                neg = node.get("not")
                if type(neg) is not dict or type(neg.get("and")) is not list:
                    continue
                pair = set()
                for child in neg["and"]:
                    item = compare(child)
                    if type(item) is not dict or item.get("op") != "==":
                        continue
                    left, right = item.get("left"), item.get("right")
                    if scalar(left) is True and var_id(right) is not None:
                        pair.add(var_id(right))
                    if scalar(right) is True and var_id(left) is not None:
                        pair.add(var_id(left))
                if len(pair) == 2:
                    forbidden.add(frozenset(pair))
            if any(frozenset((i, i+2)) not in forbidden
                   for i in range(n-2)):
                return "binary_distance_witness_missing"

    # Narrow, explicitly numbered German PIN clauses. An equivalent schema
    # using a different variable naming convention may fail closed here.
    # Never classify this lexical check as NL semantic verification.
    if re.search(r"\b(?:fünfstelliger|5[\s-]*stelliger)\s+code\b", question, re.I):
        variables = schema["variables"]
        if len(variables) != 5:
            return "pin_domain_dimension_missing"
        if re.search(r"\bgenau\s+einmal\b", question, re.I):
            unique = any(type(node.get("all_different")) is list and
                         set(node["all_different"]) == set(variables)
                         for node in nodes)
            if not unique:
                return "pin_uniqueness_witness_missing"

        def sum_of(var_index, constant):
            # Either term order is legal inside the arithmetic sum.
            def check(term):
                if type(term) is not dict or type(term.get("sum")) is not list:
                    return False
                operands = term["sum"]
                if len(operands) != 2:
                    return False
                a, b = operands
                return ((var_id(a) == var_index and type(scalar(b)) is int
                         and scalar(b) == constant)
                        or (var_id(b) == var_index and type(scalar(a)) is int
                            and scalar(a) == constant))
            return check

        def shifted(left_id, operation, right_id, distance):
            for node in nodes:
                item = compare(node)
                if type(item) is not dict or item.get("op") != operation:
                    continue
                if (var_id(item.get("left")) == left_id and
                    sum_of(right_id, distance)(item.get("right"))):
                    return True
                if operation == "==" and (var_id(item.get("right")) == left_id
                    and sum_of(right_id, distance)(item.get("left"))):
                    return True
            return False

        before = re.search(
            r"\bdie\s+(\d+)\s+steht\s+vor\s+der\s+(\d+)\s*,\s*nicht\s+direkt\s+davor\b",
            question, re.I)
        if before:
            left, right = map(int, before.groups())
            if not shifted(right, ">=", left, 2):
                return "pin_nonadjacent_order_missing"

        between = re.search(
            r"\bzwischen\s+der\s+(\d+)\s+und\s+(\d+)\s+steht\s+genau\s+eine\s+ziffer\b",
            question, re.I)
        if between:
            a, b = map(int, between.groups())
            # Two directions are necessary: exactly one place in between.
            if not (shifted(a, "==", b, 2)
                    and shifted(b, "==", a, 2)
                    and any(type(node.get("or")) is list
                            for node in nodes)):
                return "pin_between_distance_missing"

        immediate = re.search(
            r"\bdie\s+(\d+)\s+steht\s+unmittelbar\s+rechts\s+von\s+der\s+(\d+)\b",
            question, re.I)
        if immediate:
            a, b = map(int, immediate.groups())
            if not shifted(a, "==", b, 1):
                return "pin_immediate_right_missing"

        edge = re.search(r"\bdie\s+(\d+)\s+steht\s+nicht\s+am\s+rand\b",
                         question, re.I)
        if edge:
            digit = int(edge.group(1))
            excluded = set()
            for node in nodes:
                item = compare(node)
                if type(item) is not dict or item.get("op") != "!=":
                    continue
                x, y = item.get("left"), item.get("right")
                if var_id(x) == digit and type(scalar(y)) is int:
                    excluded.add(scalar(y))
                if var_id(y) == digit and type(scalar(x)) is int:
                    excluded.add(scalar(x))
            if not {0, 4}.issubset(excluded):
                return "pin_edge_exclusion_missing"

    return None


def coverage_gate(question: str, schema: dict) -> dict:
    """Catch obvious missing lexical anchors, but never certify semantics.

    Integer/identifier mentions and known relation signals are necessary
    clues in some forms of English/German; they're not sufficient evidence.
    Fail closed for unsupported or ambiguous wording.
    """
    if (type(question) is not str or not question.strip()
            or len(question) > MAX_QUESTION_CHARS or type(schema) is not dict):
        return {"passed": False, "reason": "invalid_input"}
    try:
        names = schema["variables"]
        if type(names) is not dict:
            raise ValueError()
        observed = _features(schema)
        numbers: set[str] = set()
        for name, variable in names.items():
            if type(name) is not str or type(variable) is not dict:
                raise ValueError()
            numbers.update(re.findall(r"\d+", name))
            for value in variable["domain"]:
                if type(value) is int:
                    numbers.add(str(value))
        # Only schema scalar literals, cardinality thresholds, and modulo bases.
        def collect(node):
            if type(node) is dict:
                for key, value in node.items():
                    if key in {"const", "count", "by"} and type(value) is int:
                        numbers.add(str(value))
                    collect(value)
            elif type(node) is list:
                for item in node:
                    collect(item)
        collect(schema["constraints"])
        numbers.add(str(len(names)))  # "length N" is the count of named slots
        source_numbers = set(re.findall(r"(?<![A-Za-z_])\d+(?![A-Za-z_])",
                                        question))
        if not source_numbers.issubset(numbers):
            return {"passed": False, "reason": "numeric_anchor_missing"}
        # Source variable labels with simple unambiguous A/B/X format.
        for name in re.findall(r"(?<![A-Za-z_])[A-Z](?![A-Za-z_])", question):
            if name not in names:
                return {"passed": False, "reason": "entity_anchor_missing"}
        for pattern, alternatives in _SIGNALS:
            if re.search(pattern, question, flags=re.IGNORECASE):
                if not observed.intersection(alternatives):
                    return {"passed": False, "reason": "relation_anchor_missing"}
        for token in ("<=", ">=", "!=", "=="):
            if token in question and token not in observed:
                return {"passed": False, "reason": "relation_anchor_missing"}
        core_reason = _source_core_witnesses(question, schema)
        if core_reason is not None:
            return {"passed": False, "reason": core_reason}
        return {"passed": True, "reason": "lexical_and_core_witnesses_present"}
    except (KeyError, TypeError, ValueError):
        return {"passed": False, "reason": "invalid_schema"}


def _z3_proof(a: dict, b: dict) -> tuple[bool, str]:
    """Prove UNSAT of A XOR B on identical finite domains."""
    import z3

    zvars = {
        name: z3.Int(name) if spec["type"] == "int" else z3.Bool(name)
        for name, spec in a["variables"].items()
    }

    def scalar(value):
        return z3.BoolVal(value) if type(value) is bool else z3.IntVal(value)

    def term(node):
        if "var" in node:
            return zvars[node["var"]]
        if "const" in node:
            return scalar(node["const"])
        if "sum" in node:
            return z3.Sum(*(term(part) for part in node["sum"]))
        v = node["mod"]
        return term(v["term"]) % v["by"]

    def predicate(expr):
        if "compare" in expr:
            item = expr["compare"]
            x, y, op = term(item["left"]), term(item["right"]), item["op"]
            return {"==": lambda: x == y, "!=": lambda: x != y,
                    "<": lambda: x < y, "<=": lambda: x <= y,
                    ">": lambda: x > y, ">=": lambda: x >= y}[op]()
        if "all_different" in expr:
            return z3.Distinct(*(zvars[name] for name in expr["all_different"]))
        if "exactly" in expr:
            item = expr["exactly"]
            return z3.Sum(*(z3.If(zvars[name] == scalar(item["value"]), 1, 0)
                            for name in item["vars"])) == item["count"]
        if "not" in expr:
            return z3.Not(predicate(expr["not"]))
        if "and" in expr:
            return z3.And(*(predicate(x) for x in expr["and"]))
        if "or" in expr:
            return z3.Or(*(predicate(x) for x in expr["or"]))
        if "iff" in expr:
            x, y = expr["iff"]
            return predicate(x) == predicate(y)
        x, y = expr["implies"]
        return z3.Implies(predicate(x), predicate(y))

    domains = [z3.Or(*(zvars[name] == scalar(value) for value in spec["domain"]))
               for name, spec in a["variables"].items()]
    lhs = z3.And(*(predicate(c) for c in a["constraints"]))
    rhs = z3.And(*(predicate(c) for c in b["constraints"]))
    solver = z3.Solver()
    solver.set("timeout", 1500)
    solver.add(*domains)
    solver.add(z3.Xor(lhs, rhs))
    verdict = solver.check()
    if verdict == z3.unsat:
        return True, "z3_unsat"
    if verdict == z3.sat:
        return False, "z3_counterexample"
    return False, "z3_unknown"


def _exhaustive_proof(a: dict, b: dict) -> tuple[bool, str]:
    # Offline fallback remains exhaustive, never probabilistic/sampling.
    names = tuple(sorted(a["variables"]))
    domains = [a["variables"][name]["domain"] for name in names]
    if math.prod(len(d) for d in domains) > MAX_ASSIGNMENTS:
        return False, "enumeration_bounds_exceeded"
    types = {name: a["variables"][name]["type"] for name in names}
    budget_a, budget_b = [0], [0]
    clauses_a = [_type_predicate(c, types, budget_a, 0) for c in a["constraints"]]
    clauses_b = [_type_predicate(c, types, budget_b, 0) for c in b["constraints"]]
    for values in product(*domains):
        assignment = dict(zip(names, values))
        if (all(_predicate_value(c, assignment) for c in clauses_a)
                != all(_predicate_value(c, assignment) for c in clauses_b)):
            return False, "enumeration_counterexample"
    return True, "enumeration_exhaustive"


def _equivalence(a: dict, b: dict) -> tuple[bool, str]:
    try:
        return _z3_proof(a, b)
    except ImportError:
        return _exhaustive_proof(a, b)
    except Exception:
        # An internal Z3 failure is not permission to promote either schema.
        return False, "z3_error"


def assess_dual_extraction(
    question: str,
    qlora_extract: Callable[[list[dict]], str],
    base_extract: Callable[[list[dict]], str],
) -> dict:
    """Use independent model callables and compare only their candidate ASTs.

    The caller is responsible for choosing different model backends. Runtime
    cannot prove model independence. This function NEVER returns verified=True
    or a guard-trusted scope, even when the two formulas are equivalent.
    """
    if (type(question) is not str or not question.strip()
            or len(question) > MAX_QUESTION_CHARS
            or not callable(qlora_extract) or not callable(base_extract)):
        return _reject("invalid_input")
    if qlora_extract is base_extract:
        return _reject("same_extractor")
    try:
        # Different prompts, identical original question; neither extractor
        # receives or can copy the other's proposed AST via this interface.
        raw_a = qlora_extract([
            {"role": "system", "content": EXTRACT_QLORA_PROMPT},
            {"role": "user", "content": question},
        ])
        raw_b = base_extract([
            {"role": "system", "content": EXTRACT_BASE_PROMPT},
            {"role": "user", "content": question},
        ])
        a, b = _parse_candidate(raw_a), _parse_candidate(raw_b)
    except (InvalidTypedConstraints, ValueError, TypeError, UnicodeError):
        return _reject("invalid_schema")
    except Exception:
        return _reject("extractor_error")
    if _signature(a) != _signature(b):
        return _reject("signature_mismatch")
    cov_a, cov_b = coverage_gate(question, a), coverage_gate(question, b)
    if not cov_a["passed"] or not cov_b["passed"]:
        result = _reject("coverage_rejected")
        result["coverage_reason"] = (cov_a["reason"] if not cov_a["passed"]
                                     else cov_b["reason"])
        return result
    same, backend = _equivalence(a, b)
    if not same:
        status = ("proof_inconclusive" if backend in {"z3_unknown", "z3_error",
                     "enumeration_bounds_exceeded"} else "non_equivalent")
        result = _reject(status)
        result["proof_backend"] = backend
        return result
    try:
        # Exact value is only a *candidate* for offline scoring, never an
        # authoritative response in the guarded user-facing chat.
        answer = solve_typed_constraints(
            PREFIX + json.dumps(a, ensure_ascii=False), max_seconds=1.5)
    except (InvalidTypedConstraints, ValueError):
        return _reject("solver_rejected")
    result = _reject("consensus_unverified")
    result.update(schema_equivalent=True, candidate_count=answer,
                  proof_backend=backend)
    return result
