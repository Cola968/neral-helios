"""32 project-internal development probes for NERAL/Qwen comparison.

No training, external test execution or private prompts. These are explicitly
published development prompts, not hidden or independent external holdouts.
Coding uses strict AST-shape equivalence, NOT semantic unit-test correctness.
"""
from __future__ import annotations
import ast
from collections import Counter
from fractions import Fraction
import json
import re


def _case(id, family, prompt, expected):
    return {"id":id,"family":family,"prompt":prompt,"expected":expected}


CASES = [
    _case("math-01","math","Gib nur das numerische Ergebnis: Ein Viertel minus eins.","-3/4"),
    _case("math-02","math","Ein Messprotokoll enthält die Zahlen 4, 10 und 13. Gib ausschließlich ihren Mittelwert an.","9"),
    _case("math-03","math","Ein Zug fährt 360 km in 4,5 h bei gleichbleibendem Tempo. Wie viele km/h sind das? Antworte nur mit der Zahl.","80"),
    _case("math-04","math","Bestimme y aus der Gleichung 7y - 5 = 30. Antworte nur mit dem Wert von y.","5"),
    _case("math-05","math","Eine Gebühr beträgt 20 Prozent von 145 Euro. Wie hoch ist die Gebühr? Nur die Zahl nennen.","29"),
    _case("math-06","math","In einer Urne liegen drei rote und fünf blaue gleichwahrscheinliche Kugeln. Gib die Wahrscheinlichkeit für Rot als Bruch.","3/8"),
    _case("math-07","math","Ein Artikel kostet 120 Euro. Wie viel bleibt nach einem Rabatt von genau einem Viertel zu bezahlen? Nur Zahl.","90"),
    _case("math-08","math","Welcher ganzzahlige Wert entsteht durch (-4)² + 2³? Antworte nur mit der Zahl.","24"),

    _case("sql-01","sql","Ein Bericht soll ALLE Künstler zeigen, auch solche ohne Album. Welche SQL-JOIN-Art passt? Antworte nur mit zwei SQL-Schlüsselwörtern.","LEFT JOIN"),
    _case("sql-02","sql","Es sollen nur Flugzeuge erscheinen, für die ein passender Flughafen-Datensatz existiert. Welche JOIN-Art? Nur Schlüsselwörter.","INNER JOIN"),
    _case("sql-03","sql","Eine aggregierte Gruppe muss nach COUNT(*) mindestens zehn Datensätze enthalten. Welche SQL-Klausel filtert Gruppen? Nur ein Schlüsselwort.","HAVING"),
    _case("sql-04","sql","Eine SQL-Abfrage soll Zeilen filtern, bevor gruppiert wird. Welche Klausel wird verwendet? Nur ein Schlüsselwort.","WHERE"),
    _case("sql-05","sql","Welche SQL-Klausel fasst Datensätze nach ihrer GenreID zusammen? Gib nur die Klausel an.","GROUP BY"),
    _case("sql-06","sql","Welche SQL-Klausel sortiert Ergebnisse nach einem berechneten Wert? Gib nur die Klausel an.","ORDER BY"),
    _case("sql-07","sql","Welche SQL-Bedingung prüft, ob AlbumID keinen Wert enthält? Gib nur die zwei SQL-Schlüsselwörter an.","IS NULL"),
    _case("sql-08","sql","Welche SQL-Aggregatfunktion zählt Zeilen unabhängig von NULL-Spaltenwerten? Nur die Funktion mit Klammern.","COUNT(*)"),

    _case("code-01","coding_ast","Schreibe nur Python-Code: Funktion multiply(a,b) soll die beiden Argumente multiplizieren und zurückgeben.",
          "def multiply(a,b):\n    return a*b\n"),
    _case("code-02","coding_ast","Nur gültiger Python-Code: Funktion even(n) soll für gerade Ganzzahlen True zurückgeben, sonst False.",
          "def even(n):\n    return n%2==0\n"),
    _case("code-03","coding_ast","Schreibe eine Python-Funktion safe_first(items), die das erste Listenelement zurückgibt, sonst None.",
          "def safe_first(items):\n    return items[0] if items else None\n"),
    _case("code-04","coding_ast","Nur Python-Quelltext: Funktion clamp(x,lo,hi) soll einen Wert mit max und min auf das Intervall begrenzen.",
          "def clamp(x,lo,hi):\n    return max(lo,min(x,hi))\n"),
    _case("code-05","coding_ast","Python: Funktion average(values) gibt sum(values) geteilt durch len(values) zurück. Gib nur eine Funktionsdefinition.",
          "def average(values):\n    return sum(values)/len(values)\n"),
    _case("code-06","coding_ast","Python: Funktion is_palindrome(s) soll exakt s mit der umgekehrten Zeichenfolge vergleichen. Nur Code.",
          "def is_palindrome(s):\n    return s == s[::-1]\n"),
    _case("code-07","coding_ast","Implementiere in Python unique(seq) unter Erhaltung der Reihenfolge über list(dict.fromkeys(seq)). Nur Code.",
          "def unique(seq):\n    return list(dict.fromkeys(seq))\n"),
    _case("code-08","coding_ast","Nur Python: Funktion area_rectangle(width,height) gibt width mal height zurück.",
          "def area_rectangle(width,height):\n    return width*height\n"),

    _case("agent-01","agent",'Du bist ein Werkzeugagent. Ein Entwickler möchte eine erlaubte, konkrete Python-Datei ansehen. Welches vorhandene Werkzeug? Antworte nur als JSON {"action":"..."} .',"read"),
    _case("agent-02","agent",'Du brauchst alle Textstellen mit dem Wort "timeout" im bereits erlaubten Projektverzeichnis. Welches Agentenwerkzeug? Nur JSON mit action.',"search"),
    _case("agent-03","agent",'Ein Python-Modul soll auf reine Syntaxfehler überprüft werden, ohne es auszuführen. Wähle das Werkzeug als JSON mit action.',"python"),
    _case("agent-04","agent",'Die syntaktische Gültigkeit einer lokalen TOML-Datei soll geprüft werden. Wähle das passende Agentenwerkzeug als JSON.',"formats"),
    _case("agent-05","agent",'Suche nur statisch nach auffälligen Aufrufen in Quelltexten; führe nichts aus. Welches Werkzeug? Nur JSON mit action.',"security"),
    _case("agent-06","agent",'Du möchtest 7*x+3=31 exakt mit dem vorhandenen lokalen Rechenwerkzeug lösen. Wähle das Werkzeug als JSON.',"solve"),
    _case("agent-07","agent",'Verschaffe dir zunächst einen Überblick über erlaubte Dateien im Arbeitsverzeichnis, ohne sie zu öffnen. Nur JSON mit action.',"inventory"),
    _case("agent-08","agent",'Alle geplanten Prüfschritte wurden durchgeführt, die Ergebnisse liegen vor. Welche Aktion schließt den Auftrag ab? Nur JSON mit action.',"finish"),
]


def case_ids() -> list[str]:
    return [x["id"] for x in CASES]


def _numeric(text: str) -> Fraction | None:
    if not isinstance(text,str) or len(text)>80:
        return None
    value=text.strip()
    value=re.sub(r"^x\s*=\s*","",value,flags=re.I)
    if not re.fullmatch(r"[+-]?(?:\d+|\d+\.\d+|\d+/\d+)",value):
        return None
    try:
        return Fraction(value)
    except (ValueError,ZeroDivisionError):
        return None


def _sql(text: str) -> str | None:
    if not isinstance(text,str) or len(text)>80:
        return None
    normalized=" ".join(text.upper().split())
    if not re.fullmatch(r"[A-Z ]+(?:\(\*\))?",normalized):
        return None
    return normalized


def _code_ast(text: str) -> str | None:
    if not isinstance(text,str) or len(text)>2000:
        return None
    try:
        tree=ast.parse(text)
    except (SyntaxError,ValueError,RecursionError):
        return None
    if len(tree.body)!=1 or not isinstance(tree.body[0],ast.FunctionDef):
        return None
    fn=tree.body[0]
    if (fn.decorator_list or len(fn.body)!=1 or not isinstance(fn.body[0],ast.Return)
        or any(isinstance(x,(ast.Import,ast.ImportFrom,ast.Lambda,ast.Global,ast.Nonlocal,
                             ast.Await,ast.Yield,ast.YieldFrom)) for x in ast.walk(tree))):
        return None
    return ast.dump(tree,include_attributes=False)


def _agent(text: str) -> str | None:
    if not isinstance(text,str) or len(text)>200:
        return None
    try:
        result=json.loads(text)
    except (json.JSONDecodeError,TypeError):
        return None
    if not isinstance(result,dict) or set(result)!={"action"} or not isinstance(result["action"],str):
        return None
    if result["action"] not in {"read","search","python","formats","security","solve","inventory","finish"}:
        return None
    return result["action"]


def grade(case: dict, raw: str) -> dict:
    family=case["family"]
    expected=case["expected"]
    if family=="math":
        parsed=_numeric(raw)
        correct=parsed is not None and parsed==_numeric(expected)
    elif family=="sql":
        parsed=_sql(raw)
        correct=parsed is not None and parsed==expected
    elif family=="coding_ast":
        parsed=_code_ast(raw)
        correct=parsed is not None and parsed==_code_ast(expected)
    elif family=="agent":
        parsed=_agent(raw)
        correct=parsed is not None and parsed==expected
    else:
        raise ValueError("Unknown evaluation family")
    return {"id":case["id"],"family":family,"correct":bool(correct)}


def score(responses: dict[str,str]) -> dict:
    expected=set(case_ids())
    if not isinstance(responses,dict) or set(responses)!=expected:
        raise ValueError("Missing, duplicated or extra evaluation cases")
    records=[grade(case,responses[case["id"]]) for case in CASES]
    counters={family:{"correct":sum(row["correct"] for row in records if row["family"]==family),
                      "total":sum(row["family"]==family for row in records)}
              for family in ("math","sql","coding_ast","agent")}
    return {"schema":1,"suite":"NERAL_32_PROJECT_INTERNAL_DEV_GENERALIZATION",
            "total":len(CASES),"correct":sum(x["correct"] for x in records),
            "by_family":counters,"independent_external_holdout":False,
            "code_executed":False,"model_quality_improvement_proven":False,
            "training_performed":False,"raw_prompts_uploaded":False}


if __name__=="__main__":
    print("NERAL 32-case project-internal probe available; no inference or training executed.")
