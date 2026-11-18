"""Strict model-JSON to bounded deterministic Z3 Python, never host execution.

The generated code is sandbox-only evidence, NOT source-proven reasoning.
"""
from __future__ import annotations

import ast
import json
import re


class SchemaError(ValueError):
    """Invalid or unsupported model-proposed Z3 constraint schema."""


_IDENTIFIER = re.compile(r"[a-z][a-z0-9_]{0,31}\Z", re.ASCII)
_RULE_ID = re.compile(r"R[1-9][0-9]{0,2}\Z", re.ASCII)
_FUNCTIONS = {"And": (2, 16), "Or": (2, 16), "Not": (1, 1), "Implies": (2, 2)}
_OPS = {ast.Add, ast.Sub, ast.Mult}
_CMP = {ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE}
_RESERVED = {"z3", "solver", "count", "m", "all_vars", "v", "range"}


def _unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise SchemaError("JSON enthält doppelte Schlüssel: " + key[:40])
        value[key] = item
    return value


def _bad_constant(value):
    raise SchemaError("JSON enthält verbotenen numerischen Wert")


class _CanonicalLogicalSyntax(ast.NodeTransformer):
    """Translate only semantics-preserving Python logic sugar into Z3 calls.

    Whitelist validation still runs AFTER this rewrite. No code is executed.
    """
    def visit_BoolOp(self, node):
        if type(node.op) not in (ast.And, ast.Or):
            return node
        name = "And" if isinstance(node.op, ast.And) else "Or"
        return ast.Call(func=ast.Name(id=name, ctx=ast.Load()),
                        args=[self.visit(arg) for arg in node.values],
                        keywords=[])

    def visit_UnaryOp(self, node):
        if isinstance(node.op, ast.Not):
            return ast.Call(func=ast.Name(id="Not", ctx=ast.Load()),
                            args=[self.visit(node.operand)], keywords=[])
        return self.generic_visit(node)

    def visit_Compare(self, node):
        if len(node.ops) > 1:
            items = [node.left, *node.comparators]
            comparisons = [
                ast.Compare(left=items[i], ops=[op], comparators=[items[i+1]])
                for i, op in enumerate(node.ops)
            ]
            return ast.Call(
                func=ast.Name(id="And", ctx=ast.Load()),
                args=[self.visit(n) for n in comparisons], keywords=[])
        return self.generic_visit(node)


def _checked_expression(value: object, names: set[str]) -> str:
    if type(value) is not str or not 3 <= len(value) <= 230:
        raise SchemaError("Regelausdruck fehlt oder ist zu lang")
    try:
        expression = ast.parse(value, mode="eval")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise SchemaError("Ungültiger Regelausdruck") from exc
    if sum(1 for _ in ast.walk(expression)) > 85:
        raise SchemaError("Regelausdruck zu komplex")
    try:
        expression = ast.fix_missing_locations(
            _CanonicalLogicalSyntax().visit(expression))
    except RecursionError as exc:
        raise SchemaError("Regelausdruck zu tief verschachtelt") from exc
    if sum(1 for _ in ast.walk(expression)) > 100:
        raise SchemaError("Normalisierter Regelausdruck zu komplex")

    def type_of(n):
        if isinstance(n, ast.Name):
            if n.id not in names:
                raise SchemaError("Nicht definierte Variable: " + n.id[:50])
            return "int"
        if isinstance(n, ast.Constant) and type(n.value) is int:
            if abs(n.value) > 1_000_000:
                raise SchemaError("Ganzzahl außerhalb des Wertebereichs")
            return "int"
        if isinstance(n, ast.UnaryOp) and type(n.op) in (ast.UAdd, ast.USub):
            if type_of(n.operand) != "int":
                raise SchemaError("Vorzeichen nur für Ganzzahlen erlaubt")
            return "int"
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            if type_of(n.left) != "int" or type_of(n.right) != "int":
                raise SchemaError("Arithmetik nur mit Ganzzahlen erlaubt")
            return "int"
        if isinstance(n, ast.Compare):
            if (len(n.ops) != 1 or len(n.comparators) != 1 or
                    type(n.ops[0]) not in _CMP or
                    type_of(n.left) != "int" or
                    type_of(n.comparators[0]) != "int"):
                raise SchemaError("Vergleich benötigt zwei Ganzzahlausdrücke")
            return "bool"
        if isinstance(n, ast.Call):
            if (not isinstance(n.func, ast.Name) or
                    n.func.id not in _FUNCTIONS or n.keywords):
                raise SchemaError("Nur And, Or, Not und Implies sind erlaubt")
            low, high = _FUNCTIONS[n.func.id]
            if not low <= len(n.args) <= high:
                raise SchemaError("Falsche Anzahl logischer Argumente")
            if any(type_of(arg) != "bool" for arg in n.args):
                raise SchemaError("Logikfunktionen benötigen Vergleiche")
            return "bool"
        raise SchemaError("Nicht erlaubte AST-Operation: " + type(n).__name__)

    if type_of(expression.body) != "bool":
        raise SchemaError("Jede Regel muss einen booleschen Vergleich liefern")

    class PrefixZ3(ast.NodeTransformer):
        def visit_Call(self, node):
            node = self.generic_visit(node)
            node.func = ast.Attribute(
                value=ast.Name(id="z3", ctx=ast.Load()),
                attr=node.func.id, ctx=ast.Load())
            return node

    tree = ast.fix_missing_locations(PrefixZ3().visit(expression))
    return ast.unparse(tree.body)


def compile_z3_json(raw: object) -> str:
    """Fail closed on unknown fields, duplicate keys, AST injection and oversize.

    The model supplies constraints, not Python. This compiler constructs the
    only executable source and is never invoked with user-provided Python.
    """
    if type(raw) is not str or not 2 <= len(raw.encode("utf-8")) <= 7000:
        raise SchemaError("JSON-Antwort fehlt oder überschreitet 7000 Byte")
    # Only deterministic presentation wrappers; parsed content stays strict.
    raw = re.sub(
        r"\A\s*(?:(?:hier\s+ist\s+(?:das\s+)?)?json(?:-objekt)?|antwort)\s*:\s*",
        "", raw, count=1, flags=re.I)
    # A single optional JSON fence is presentation, not executable code.
    # No surrounding prose, multiple fences, or other languages are accepted.
    fenced = re.fullmatch(
        r"\s*\x60{3}(?:json)?[ \t]*\r?\n([\s\S]*?)\r?\n\x60{3}\s*",
        raw, re.I)
    if fenced is not None:
        raw = fenced.group(1)
    try:
        payload = json.loads(raw, object_pairs_hook=_unique_pairs,
                             parse_constant=_bad_constant)
    except SchemaError:
        raise
    except json.JSONDecodeError as exc:
        raise SchemaError(
            f"JSON syntaktisch ungültig (Zeile {exc.lineno}, Spalte {exc.colno}): "
            + str(exc.msg)[:75]) from exc
    except (ValueError, TypeError, RecursionError) as exc:
        raise SchemaError("Erwartet wird genau ein gültiges JSON-Objekt") from exc
    if type(payload) is not dict:
        raise SchemaError("JSON muss ein Objekt sein; erhalten: " + type(payload).__name__)
    expected = {"variables", "constraints", "objective"}
    if set(payload) != expected:
        missing = ",".join(sorted(expected - set(payload))) or "-"
        unexpected = ",".join(sorted(
            key[:32] for key in set(payload) - expected
            if type(key) is str and re.fullmatch(r"[A-Za-z0-9_]{1,32}", key))) or "andere"
        raise SchemaError("Falsche JSON-Schlüssel: fehlend=" + missing +
                          "; unerwartet=" + unexpected)
    if payload["objective"] not in ("count_all", "find_one"):
        raise SchemaError("objective muss count_all oder find_one sein")
    variables = payload["variables"]
    rules = payload["constraints"]
    if (type(variables) is not list or not 1 <= len(variables) <= 12 or
            type(rules) is not list or not 1 <= len(rules) <= 16):
        raise SchemaError("Erwartet: 1..12 Variablen und 1..16 Regeln")

    names = set()
    for var in variables:
        if type(var) is not dict or set(var) != {"name", "min", "max"}:
            raise SchemaError("Variable braucht ausschließlich name, min, max")
        name = var["name"]
        if (type(name) is not str or _IDENTIFIER.fullmatch(name) is None
                or name in _RESERVED or name.startswith("rule_")
                or name in names):
            raise SchemaError("Ungültiger oder doppelter Variablenname")
        lo, hi = var["min"], var["max"]
        if (type(lo) is not int or type(hi) is not int or
                not -100000 <= lo <= hi <= 100000 or hi - lo > 100000):
            raise SchemaError("Ganzzahl-Domain min/max ist ungültig")
        names.add(name)

    seen_ids = set()
    seen_names = set()
    compiled = []
    for rule in rules:
        if type(rule) is not dict or set(rule) != {
                "rule_id", "description", "expr"}:
            raise SchemaError("Regel braucht rule_id, description, expr")
        rid, desc = rule["rule_id"], rule["description"]
        if (type(rid) is not str or not _RULE_ID.fullmatch(rid)
                or rid in seen_ids):
            raise SchemaError("rule_id muss eindeutig wie R1, R2 ... sein")
        if (type(desc) is not str or not 4 <= len(desc.strip()) <= 150
                or any(ord(ch) < 32 for ch in desc)):
            raise SchemaError("Kurze Regelbeschreibung aus der Aufgabe fehlt")
        expr = _checked_expression(rule["expr"], names)
        for n in ast.walk(ast.parse(rule["expr"], mode="eval")):
            if isinstance(n, ast.Name) and n.id in names:
                seen_names.add(n.id)
        seen_ids.add(rid)
        compiled.append((rid, desc.strip().replace("\n", " "), expr))
    if seen_names != names:
        raise SchemaError("Jede Variable muss in mindestens einer Regel vorkommen")

    # Compact deterministic construction keeps 12 variables / 16 rules
    # inside the unchanged 4096-byte / 384-AST-node Docker limits.
    named = [var["name"] for var in variables]
    lines = [
        "import z3",
        "all_vars=z3.Ints(" + repr(" ".join(named)) + ")",
        ",".join(named) + "=all_vars" if len(named)>1
        else named[0] + ",=all_vars",
        "solver=z3.Solver()",
        "bounds_lo=[" + ",".join(str(v["min"]) for v in variables) + "]",
        "bounds_hi=[" + ",".join(str(v["max"]) for v in variables) + "]",
        "for v,lo,hi in zip(all_vars,bounds_lo,bounds_hi):solver.add(v>=lo,v<=hi)",
        "rule_ids=" + repr(" ".join(rid for rid,_,_ in compiled)) + ".split()",
        "rules=[" + ",".join(expr for _,_,expr in compiled) + "]",
        "for rid,rule in zip(rule_ids,rules):solver.add(rule)",
    ]
    # The ID-to-expression pairing stays positional. Every rule goes into
    # solver.add once and is independently asserted for every found model.
    checks = [
        "    for rid,rule in zip(rule_ids,rules):",
        "        assert z3.is_true(m.eval(rule,model_completion=True)),rid",
    ]
    if payload["objective"] == "count_all":
        lines += [
            "count=0",
            "while solver.check() == z3.sat:",
            "    m=solver.model()",
            *checks,
            "    count+=1",
            "    if count>100000: raise RuntimeError('Enumeration limit exceeded')",
            "    solver.add(z3.Or([v != m[v] for v in all_vars]))",
            "print(count)",
        ]
    else:
        lines += [
            "if solver.check() == z3.sat:",
            "    m=solver.model()",
            *checks,
            "    print('; '.join(str(v)+'='+str(m.eval(v,model_completion=True)) for v in all_vars))",
            "else:",
            "    print('UNSAT')",
        ]
    code = "\n".join(lines) + "\n"
    if (len(code.encode("utf-8")) > 4096 or
            sum(1 for _ in ast.walk(ast.parse(code))) > 384):
        raise SchemaError("Z3-Programm übersteigt Docker-Sandboxgrenzen")
    return code
