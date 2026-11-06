"""Preview-only, deterministic repairs for the local NERAL agent.

No generated code is executed and no file is written by this module.
The calling AgentEngine is responsible for allowlisted paths, SHA-256 CAS,
explicit approval and reversible atomic writes.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path


class QuickFixError(ValueError):
    """A requested transformation is unavailable or not provably safe."""


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise QuickFixError("Doppelte JSON-Schlüssel dürfen nicht stillschweigend verschwinden.")
        result[key] = value
    return result


def _reject_constant(value):
    raise QuickFixError("Nichtstandardisierte JSON-Zahlen werden nicht übernommen.")


def _safe_ast(text):
    try:
        return ast.dump(ast.parse(text), include_attributes=False)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise QuickFixError("Python-Code ist nicht für diese sichere Änderung geeignet.") from exc


def _python_whitespace(content: str) -> str:
    baseline = _safe_ast(content)
    parts = []
    for line in content.splitlines(keepends=True):
        if line.endswith("\r\n"):
            body, newline = line[:-2], "\r\n"
        elif line.endswith(("\n", "\r")):
            body, newline = line[:-1], line[-1]
        else:
            body, newline = line, ""
        parts.append(body.rstrip(" \t") + newline)
    updated = "".join(parts)
    if updated and not updated.endswith(("\n", "\r")):
        updated += "\r\n" if "\r\n" in content else "\n"
    if _safe_ast(updated) != baseline:
        raise QuickFixError("Semantik könnte sich ändern (z. B. Leerzeichen in mehrzeiligen Zeichenketten).")
    return updated


def _python_colon(content: str) -> str:
    try:
        ast.parse(content)
    except SyntaxError as exc:
        line_number = exc.lineno
    except (ValueError, RecursionError) as exc:
        raise QuickFixError("Python-Quelltext ist nicht sicher analysierbar.") from exc
    else:
        raise QuickFixError("Keine Syntaxkorrektur erforderlich.")
    if type(line_number) is not int or line_number < 1:
        raise QuickFixError("Fehlerstelle nicht eindeutig.")
    lines = content.splitlines(keepends=True)
    if line_number > len(lines):
        raise QuickFixError("Fehlerstelle nicht eindeutig.")
    original = lines[line_number - 1]
    if original.endswith("\r\n"):
        body, newline = original[:-2], "\r\n"
    elif original.endswith(("\r", "\n")):
        body, newline = original[:-1], original[-1]
    else:
        body, newline = original, ""
    stripped = body.rstrip(" \t")
    if (not re.match(r"^[ \t]*(?:async[ \t]+)?(?:def|class|if|elif|else|for|while|try|except|finally|with)\b", stripped)
            or stripped.endswith((":","\\")) or "#" in stripped or not stripped.strip()):
        raise QuickFixError("Kein eindeutig fehlender Doppelpunkt.")
    lines[line_number - 1] = stripped + ":" + newline
    updated = "".join(lines)
    try:
        ast.parse(updated)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise QuickFixError("Ein einzelner Doppelpunkt behebt den Code nicht vollständig.") from exc
    return updated


def build_quick_fix(filename: str, content: str, kind: str) -> str:
    """Return a proposed UTF-8 replacement, not a mutation or a promise."""
    if not isinstance(filename, str) or not isinstance(content, str) or not isinstance(kind, str):
        raise QuickFixError("Ungültige Quick-Fix-Eingabe.")
    if len(content.encode("utf-8")) > 32768:
        raise QuickFixError("Datei zu groß für eine kontrollierte Reparatur.")
    ext = Path(filename).suffix.lower()
    if kind == "json_pretty" and ext == ".json":
        try:
            data = json.loads(content, object_pairs_hook=_unique_pairs, parse_constant=_reject_constant)
            result = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        except (ValueError, OverflowError, RecursionError, UnicodeError) as exc:
            raise QuickFixError("JSON kann nicht verlustfrei und sicher formatiert werden.") from exc
    elif kind == "python_whitespace" and ext == ".py":
        result = _python_whitespace(content)
    elif kind == "python_colon" and ext == ".py":
        result = _python_colon(content)
    else:
        raise QuickFixError("Dieser Quick-Fix ist für diesen Dateityp nicht verfügbar.")
    if result == content:
        raise QuickFixError("Keine Änderung notwendig.")
    if len(result.encode("utf-8")) > 32768:
        raise QuickFixError("Repariertes Ergebnis überschreitet Bearbeitungsgrenze.")
    return result
