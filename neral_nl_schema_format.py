"""JSON Schema for experimental Ollama structured NL -> NERAL constraints v2.

Advisory *generation* constraint, not a proof about the input text. The exact
solver remains the final strict validator; no automatically verified model
response, schema repair, user source modification, or v1 format changes.
"""
from __future__ import annotations

import ast
import copy
import re

_IDENT = r"^[A-Za-z_][A-Za-z_0-9]{0,31}$"
_NAME = {"type": "string", "pattern": _IDENT}
_INT = {"type": "integer", "minimum": -1000000, "maximum": 1000000}
_VALUE = {"oneOf": [_INT, {"type": "boolean"}]}


def _obj(properties: dict, *, required=None) -> dict:
    return {
        "type": "object", "properties": properties,
        "required": list(properties) if required is None else list(required),
        "additionalProperties": False,
    }


def _array(items: dict, minimum: int, maximum: int, *, unique=False) -> dict:
    spec = {"type": "array", "items": items,
            "minItems": minimum, "maxItems": maximum}
    if unique:
        spec["uniqueItems"] = True
    return spec


def _build() -> dict:
    term_ref = {"$ref": "#/$defs/term"}
    predicate_ref = {"$ref": "#/$defs/predicate"}
    terms = _array(term_ref, 2, 12)
    variable_names = _array(_NAME, 2, 10, unique=True)
    bool_args = _array(predicate_ref, 2, 12)
    binary_args = _array(predicate_ref, 2, 2)
    term = {"oneOf": [
        _obj({"var": _NAME}),
        _obj({"const": _VALUE}),
        _obj({"sum": terms}),
        _obj({"mod": _obj({"term": term_ref,
                          "by": {"type": "integer", "minimum": 1,
                                 "maximum": 10000}})}),
    ]}
    predicate = {"oneOf": [
        _obj({"compare": _obj({
            "left": term_ref,
            "op": {"type": "string",
                   "enum": ["==", "!=", "<", "<=", ">", ">="]},
            "right": term_ref,
        })}),
        _obj({"all_different": variable_names}),
        _obj({"exactly": _obj({
            "vars": _array(_NAME, 1, 10, unique=True),
            "value": _VALUE,
            "count": {"type": "integer", "minimum": 0, "maximum": 10},
        })}),
        _obj({"not": predicate_ref}),
        _obj({"and": bool_args}),
        _obj({"or": bool_args}),
        _obj({"iff": binary_args}),
        _obj({"implies": binary_args}),
    ]}
    typed_variables = {"oneOf": [
        _obj({"type": {"const": "int"},
              "domain": _array(_INT, 1, 12, unique=True)}),
        _obj({"type": {"const": "bool"},
              "domain": _array({"type": "boolean"}, 1, 2, unique=True)}),
    ]}
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "version": {"const": 2},
            "variables": {
                "type": "object",
                "patternProperties": {_IDENT: typed_variables},
                "additionalProperties": False,
                "minProperties": 1,
                "maxProperties": 10,
            },
            "constraints": _array(predicate_ref, 1, 32),
            "query": {"type": "string", "enum": ["count", "satisfiable"]},
        },
        "required": ["version", "variables", "constraints", "query"],
        "additionalProperties": False,
        "$defs": {"term": term, "predicate": predicate},
    }



# Only literal source-pattern families with unambiguous variable identities.
# No historic case ID, gold answer, or frozen corpus content influences this.
_CODE_LENGTH_WORDS = {
    "zwei": 2, "drei": 3, "vier": 4, "fünf": 5, "fuenf": 5,
    "sechs": 6, "sieben": 7, "acht": 8, "neun": 9, "zehn": 10,
}


def expected_variable_declarations(question: str) -> dict | None:
    """Extract an *untrusted* declaration scaffold from explicit source shape.

    Only a few generic, unambiguous problem formats are supported. The
    scaffold is for constrained model generation, never an authoritative
    interpretation of natural language. Unsupported questions get None.
    """
    if type(question) is not str or not 10 < len(question) <= 8192:
        return None

    graph = re.search(
        r"\bColor vertices 0\.\.(\d+)\s+with\s+red,\s*green\s+or\s+blue\b",
        question, re.IGNORECASE)
    if graph:
        last = int(graph.group(1))
        if 1 <= last <= 9:
            return {f"C{i}": {"type": "int", "domain": [0, 1, 2]}
                    for i in range(last + 1)}
        return None

    binary = re.search(
        r"\bbinary\s+strings?\s+of\s+length\s+(\d+)\b",
        question, re.IGNORECASE)
    if binary:
        n = int(binary.group(1))
        if 2 <= n <= 10:
            return {f"B{i}": {"type": "bool", "domain": [False, True]}
                    for i in range(n)}
        return None

    pin = re.search(
        r"\b([a-zäöüß0-9]+)stelliger\s+Code\s+verwendet\s+(.{1,180}?)"
        r"\s+genau\s+einmal\b", question, re.IGNORECASE)
    if pin:
        stated = pin.group(1).lower()
        n = int(stated) if stated.isdecimal() else _CODE_LENGTH_WORDS.get(stated)
        if n is None or not 2 <= n <= 10:
            return None
        digits = [int(i) for i in re.findall(r"\d+", pin.group(2))]
        if len(digits) != n or sorted(digits) != list(range(1, n + 1)):
            return None
        return {f"p{i}": {"type": "int", "domain": list(range(1, n + 1))}
                for i in range(1, n + 1)}

    return None



def expected_minimum_source_constraints(question: str) -> int | None:
    """Advisory lower bound on top-level rules for narrow source families.

    This is only a shape hint for constrained generation, NOT a certificate
    of the task's meaning, nor an answer. The independent source grammar and
    Z3 proof in the guard remain mandatory for any promoted output.
    """
    decl = expected_variable_declarations(question)
    if decl is None:
        return None
    names = tuple(decl)
    if all(re.fullmatch(r"C\d+", x) for x in names):
        if not re.search(r"\bVertex\s+0\s+is\s+red\b", question, re.I):
            return None
        match = re.search(
            r"\bedges\s+(\[[^\n]{1,800}\])\s+must\s+differ\b",
            question, re.I)
        if not match:
            return None
        try:
            pairs = ast.literal_eval(match.group(1))
        except (SyntaxError, ValueError, TypeError, MemoryError):
            return None
        if (type(pairs) is not list or not 1 <= len(pairs) <= 31 or
                any(type(p) not in (tuple, list) or len(p) != 2 or
                    any(type(v) is not int or v < 0 or v >= len(names)
                        for v in p) or p[0] == p[1] for p in pairs)):
            return None
        return 1 + len(pairs)
    if all(re.fullmatch(r"B\d+", x) for x in names):
        if (re.search(r"\bexactly\s+\d+\s+ones?\b", question, re.I)
                and re.search(r"\bfirst\s+two\s+bits\s+opposite\b", question, re.I)
                and re.search(
                    r"\bno\s+two\s+ones\s+exactly\s+two\s+positions\s+apart\b",
                    question, re.I)):
            return len(names)  # exactly-k + opposite + n-2 forbidden pairs
        return None
    if all(re.fullmatch(r"p\d+", x) for x in names):
        forms = (
            r"\bDie\s+\d+\s+steht\s+vor\s+der\s+\d+,\s*nicht\s+direkt\s+davor\b",
            r"\bZwischen\s+der\s+\d+\s+und\s+\d+\s+steht\s+genau\s+eine\s+Ziffer\b",
            r"\bDie\s+\d+\s+steht\s+unmittelbar\s+rechts\s+von\s+der\s+\d+\b",
            r"\bDie\s+\d+\s+steht\s+nicht\s+am\s+Rand\b",
        )
        if len(names) >= 3 and all(re.search(p, question, re.I) for p in forms):
            return 6  # all-different, order, OR, right, lower edge, upper edge
    return None


def _source_bound_variable_properties(declarations: dict) -> dict:
    """Use explicit *required* names, not ignored minProperties, for known shapes."""
    entries = {}
    for name, spec in declarations.items():
        domain = spec["domain"]
        entries[name] = _obj({
            "type": {"const": spec["type"]},
            "domain": _array(
                {"enum": domain}, len(domain), len(domain), unique=True),
        })
    return {
        "type": "object",
        "properties": entries,
        "required": list(entries),
        "additionalProperties": False,
    }

_FORMAT = _build()


def get_v2_ollama_format(question: str | None = None) -> dict:
    """Return per-request isolated JSON Schema.

    For explicit finite families, fixed required variable keys avoid the
    Ollama/llama.cpp free-dictionary minProperties gap. For unknown language,
    the generic map remains and the downstream solver rejects an empty map.
    """
    fmt = copy.deepcopy(_FORMAT)
    declarations = expected_variable_declarations(question)
    if declarations is not None:
        fmt["properties"]["variables"] = _source_bound_variable_properties(
            declarations)
    minimum = expected_minimum_source_constraints(question)
    if minimum is not None:
        # The sampler may ignore an array bound; downstream coverage MUST
        # still independently reject omitted source clauses.
        fmt["properties"]["constraints"]["minItems"] = minimum
    return fmt
