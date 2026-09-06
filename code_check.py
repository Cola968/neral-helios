"""Static Python checks only. Submitted code is never executed or imported."""
from __future__ import annotations

import ast
import re

MAX_CODE_BYTES = 4096


def _name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return ""


def _literal_string(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _finding(node, code, message, severity="medium"):
    return {
        "line": getattr(node, "lineno", None),
        "column": getattr(node, "col_offset", None),
        "code": code,
        "severity": severity,
        "message": message,
    }


def check_code(source):
    """Parse Python and flag high-signal security/reliability patterns.

    This is intentionally static: no submitted code is executed, imported, evaluated, or
    passed to a shell. Findings are heuristics and are not a substitute for a full SAST tool.
    """
    if not isinstance(source, str):
        raise ValueError("Code muss Text sein.")
    if len(source.encode("utf-8")) > MAX_CODE_BYTES:
        raise ValueError("Maximal 4096 UTF-8-Bytes Code erlaubt.")
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return {
            "syntax_ok": False,
            "line": e.lineno,
            "column": e.offset,
            "message": e.msg,
            "findings": [],
            "executed": False,
        }
    except (RecursionError, ValueError) as e:
        raise ValueError("Code kann nicht sicher geparst werden.") from e

    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            findings.append(_finding(node, "bare-except", "Nacktes except: möglichst konkrete Ausnahmetypen behandeln.", "low"))

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defaults = list(node.args.defaults) + [x for x in node.args.kw_defaults if x is not None]
            if any(isinstance(x, (ast.List, ast.Dict, ast.Set)) for x in defaults):
                findings.append(_finding(node, "mutable-default", "Veränderbarer Standardwert wird zwischen Aufrufen geteilt. Nutze None und initialisiere im Funktionskörper.", "low"))

        if isinstance(node, ast.Call):
            name = _name(node.func)
            if name in {"eval", "exec", "builtins.eval", "builtins.exec"}:
                findings.append(_finding(node, "dynamic-code-execution", "Dynamische Codeausführung: keine nicht vertrauenswürdigen Eingaben ausführen.", "high"))
            if name in {"os.system", "os.popen"}:
                findings.append(_finding(node, "shell-execution", "Shell-Ausführung erkannt. Bevorzuge subprocess mit Argumentliste und ohne shell=True.", "high"))
            if name in {"subprocess.run", "subprocess.call", "subprocess.Popen", "subprocess.check_output", "subprocess.check_call"}:
                for kw in node.keywords:
                    if kw.arg == "shell" and isinstance(kw.value, ast.Constant) and kw.value.value is True:
                        findings.append(_finding(node, "subprocess-shell-true", "shell=True erhöht Command-Injection-Risiken. Nutze eine Argumentliste und shell=False.", "high"))
            if name in {"pickle.loads", "pickle.load", "marshal.loads", "marshal.load"}:
                findings.append(_finding(node, "unsafe-deserialization", "Unsichere Deserialisierung kann bei nicht vertrauenswürdigen Daten Code ausführen. Nutze ein sicheres Datenformat.", "high"))
            if name in {"yaml.load"}:
                safe_loader = any(kw.arg == "Loader" and "SafeLoader" in _name(kw.value) for kw in node.keywords)
                if not safe_loader:
                    findings.append(_finding(node, "yaml-unsafe-load", "yaml.load ohne SafeLoader kann unsichere Objekte konstruieren. Nutze yaml.safe_load.", "high"))
            if name.startswith("requests."):
                for kw in node.keywords:
                    if kw.arg == "verify" and isinstance(kw.value, ast.Constant) and kw.value.value is False:
                        findings.append(_finding(node, "tls-verification-disabled", "TLS-Zertifikatsprüfung ist deaktiviert (verify=False).", "high"))
            if name in {"hashlib.md5", "hashlib.sha1"}:
                findings.append(_finding(node, "weak-crypto-hash", "MD5/SHA-1 sind für kryptografische Integrität/Passwörter ungeeignet.", "medium"))
            if name.endswith(".execute") and node.args:
                query = node.args[0]
                if isinstance(query, ast.JoinedStr) or isinstance(query, ast.BinOp) and isinstance(query.op, (ast.Add, ast.Mod)):
                    findings.append(_finding(node, "sql-string-interpolation", "Dynamisch zusammengesetzte SQL-Abfrage erkannt. Nutze parametrisierte Queries.", "high"))

        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            value = node.value
            secret_value = _literal_string(value)
            if secret_value and len(secret_value) >= 8:
                target_names = " ".join(_name(t).lower() for t in targets)
                if re.search(r"(password|passwd|secret|api_?key|token|credential)", target_names):
                    findings.append(_finding(node, "hardcoded-secret", "Möglicherweise fest eingebettetes Geheimnis. Nutze Umgebungsvariablen oder einen Secret Store.", "high"))

    # Stable output makes regression tests and UI rendering deterministic.
    findings.sort(key=lambda f: ((f["line"] or 0), f["code"], f["message"]))
    severities = {"high": 0, "medium": 0, "low": 0}
    for finding in findings:
        severities[finding["severity"]] += 1
    return {
        "syntax_ok": True,
        "findings": findings,
        "finding_count": len(findings),
        "severity_counts": severities,
        "executed": False,
        "note": "Syntax und ausgewählte defensive Muster geprüft. Keine Aussage über funktionale Korrektheit oder vollständige Sicherheit.",
    }
