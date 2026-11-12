"""JSON Schema for experimental Ollama structured NL -> NERAL constraints v2.

Advisory *generation* constraint, not a proof about the input text. The exact
solver remains the final strict validator; no automatically verified model
response, schema repair, user source modification, or v1 format changes.
"""
from __future__ import annotations

import copy

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


_FORMAT = _build()


def get_v2_ollama_format() -> dict:
    """Return isolated JSON schema; no shared mutation between Ollama calls."""
    return copy.deepcopy(_FORMAT)
