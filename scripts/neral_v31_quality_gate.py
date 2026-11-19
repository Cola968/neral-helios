"""Prospective V3.1 100-task quality corpus and conservative, auditable scoring.

The 80 objective tasks are scored against deterministic integer/string oracles.
20 live-web tasks are scored ONLY for literal citation provenance, two-sided
coverage and absence of unquoted numbers; that is not factual truth proof.
Never count a failed API call, incomplete response or absent task as a pass.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import statistics
import unicodedata
from collections import Counter
from pathlib import Path

from scripts.neral_blind_100_cases import (
    mathematics, logic, programming, build as prior_blind100,
)
from scripts.neral_generalization_100_cases import build as prior_generalization100

SEED = 202611192107
CATEGORIES = ("math", "logic", "programming", "web", "knowledge")
TOTAL = 100
TARGET = 90

# Pairs are immutable and require actual HTTPS text reads for BOTH subjects.
# Subject names must be unique and word-boundary match the cited page text.
WEB_PAIRS = (
    ("Firefox", "Chrome", "Browser-Engine und Schutz vor Tracking"),
    ("Git", "Subversion", "verteilte und zentrale Versionsverwaltung"),
    ("Python", "Ruby", "Sprachmerkmale und Laufzeit"),
    ("TypeScript", "JavaScript", "statische Typprüfung und Ausführung"),
    ("PostgreSQL", "SQLite", "Datenbankbetrieb und Transaktionen"),
    ("Linux", "Windows", "Betriebssystem und Lizenzierung"),
    ("TCP", "UDP", "Datenübertragung und Zuverlässigkeit"),
    ("HTTP", "HTTPS", "Verschlüsselung und Transport"),
    ("React", "Vue", "Komponenten und Zustandsverwaltung"),
    ("Angular", "Svelte", "Framework und Build-Prozess"),
    ("Docker", "Podman", "Containerbetrieb und Daemon"),
    ("Kubernetes", "Nomad", "Container-Orchestrierung"),
    ("Nginx", "Apache", "Webserver und Konfiguration"),
    ("CSS", "Sass", "Stylesheets und Vorverarbeitung"),
    ("NumPy", "pandas", "Datenstrukturen und wissenschaftliches Rechnen"),
    ("PyTorch", "TensorFlow", "Machine-Learning-Frameworks"),
    ("FastAPI", "Flask", "Web-APIs und Python-Frameworks"),
    ("Rust", "Go", "Speicherverwaltung und Nebenläufigkeit"),
    ("Redis", "Memcached", "Cache und Datenstrukturen"),
    ("GitHub", "GitLab", "Codehosting und CI-Funktionen"),
)

# Fixed, historical, non-contested facts. Require exact short answer.
KNOWLEDGE = (
    ("Wie heißt die Hauptstadt von Kanada? Antworte nur mit dem Stadtnamen.", "Ottawa"),
    ("Wie heißt die Hauptstadt von Australien? Antworte nur mit dem Stadtnamen.", "Canberra"),
    ("Wie heißt die Hauptstadt von Japan? Antworte nur mit dem Stadtnamen.", "Tokio"),
    ("Wie heißt die Hauptstadt von Griechenland? Antworte nur mit dem Stadtnamen.", "Athen"),
    ("Wie heißt die Hauptstadt von Island? Antworte nur mit dem Stadtnamen.", "Reykjavík"),
    ("Wie heißt die Hauptstadt von Finnland? Antworte nur mit dem Stadtnamen.", "Helsinki"),
    ("Wie heißt die Hauptstadt von Portugal? Antworte nur mit dem Stadtnamen.", "Lissabon"),
    ("Wie heißt die Hauptstadt von Norwegen? Antworte nur mit dem Stadtnamen.", "Oslo"),
    ("Welches chemische Elementsymbol gehört zu Gold? Nur das Symbol.", "Au"),
    ("Welches chemische Elementsymbol gehört zu Sauerstoff? Nur das Symbol.", "O"),
    ("Welches chemische Elementsymbol gehört zu Natrium? Nur das Symbol.", "Na"),
    ("Welches chemische Elementsymbol gehört zu Kalium? Nur das Symbol.", "K"),
    ("Wie lautet die chemische Formel für reines Wasser? Nur die Formel.", "H2O"),
    ("Wie heißt die SI-Einheit der Frequenz? Nur die Einheit.", "Hertz"),
    ("Wie heißt die SI-Einheit der elektrischen Stromstärke? Nur die Einheit.", "Ampere"),
    ("Wie heißt die SI-Einheit der Temperatur? Nur die Einheit.", "Kelvin"),
    ("Wie heißt der größte Planet des Sonnensystems? Nur der Name.", "Jupiter"),
    ("Wie heißt der rote Planet des Sonnensystems? Nur der Name.", "Mars"),
    ("In welchem Jahr fiel die Berliner Mauer? Nur die Jahreszahl.", "1989"),
    ("Welcher Fluss durchfließt Paris? Nur der Flussname.", "Seine"),
)

def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _generate():
    rng = random.Random(SEED)
    prior = {digest(case["question"]) for case in
             (prior_blind100() + prior_generalization100())}
    used = set(prior)
    cases = []
    for category, generator in (
        ("math", mathematics), ("logic", logic), ("programming", programming)
    ):
        for i in range(20):
            for attempt in range(300):
                # Rotate the five families instead of favoring one template.
                question, gold, family = generator(i, rng)
                # Distinguish this prospective set from the two earlier 100s.
                if digest(question) not in used:
                    used.add(digest(question))
                    break
            else:
                raise RuntimeError("Could not generate unseen objective case")
            cases.append({
                "id": f"v31-{category}-{i+1:02d}", "category": category,
                "family": family, "question": question +
                    "\nReturn exactly one final integer. No explanation.",
                "gold": gold,
            })
    for i, (left, right, facet) in enumerate(WEB_PAIRS, 1):
        question = (f"Suche aktuelle Webseiten und vergleiche {left} mit {right}: "
                    f"Welche belegten Unterschiede gibt es bei {facet}? "
                    "Nenne wörtlich belegte Quellen für beide.")
        cases.append({
            "id": f"v31-web-{i:02d}", "category": "web",
            "family": "live_comparison", "question": question,
            "web_query": f"{left} {right} {facet}"[:160],
            "entities": [left, right], "gold": None,
        })
    for i, (question, gold) in enumerate(KNOWLEDGE, 1):
        cases.append({
            "id": f"v31-knowledge-{i:02d}", "category": "knowledge",
            "family": "static_fact", "question": question, "gold": gold,
        })
    for case in cases:
        case["question_sha256"] = digest(case["question"])
    return cases

def build_cases():
    return _generate()

def validate_case_manifest(cases):
    if not isinstance(cases, list) or len(cases) != TOTAL:
        raise ValueError("Exactly 100 sealed cases required")
    if cases != _generate():
        raise ValueError("Case corpus differs from frozen prospective generator")
    if len({c["id"] for c in cases}) != TOTAL or \
       len({c["question_sha256"] for c in cases}) != TOTAL:
        raise ValueError("Duplicate prospective case")
    if any(sum(c["category"] == cat for c in cases) != 20 for cat in CATEGORIES):
        raise ValueError("Missing category")
    return TOTAL

def manifest_sha256(cases):
    validate_case_manifest(cases)
    canonical = json.dumps(cases, ensure_ascii=False, sort_keys=True,
                           separators=(",", ":")) + "\n"
    return digest(canonical)

_INTEGER = re.compile(r"-?(?:0|[1-9]\d*)")
_TERMINAL = re.compile(r"(?i)FINAL_ANSWER:\s*(-?(?:0|[1-9]\d*))")
_QUOTE = re.compile(r"\[web(\d+)\]\s*„([^“]{10,650})“")
_NUMERIC = re.compile(r"(?<![\w])\d+(?:[,.]\d+)?(?!\w)")

def _strict_integer(text):
    if not isinstance(text, str):
        return None
    content = text.strip()
    token = _INTEGER.fullmatch(content) or _TERMINAL.fullmatch(content)
    if token is None:
        return None
    return int(token.group(1) if token.groups() else token.group())

def _static_answer(text, gold):
    if not isinstance(text, str) or not text.strip():
        return False
    def normalized(value):
        return unicodedata.normalize("NFKC", value).casefold().strip().rstrip(".")
    return normalized(text) == normalized(gold)

def _web_integrity(case, result):
    answer = result.get("text")
    sources = result.get("sources", ())
    if (result.get("agent_checked") is not True or
            result.get("tool_used") != "direct_reasoning" or
            not isinstance(answer, str) or
            not isinstance(sources, list) or len(sources) < 2):
        return False
    quoted = _QUOTE.findall(answer)
    if not quoted:
        return False
    matched = []
    for number, quote in quoted:
        index = int(number)-1
        if not 0 <= index < len(sources):
            return False
        page = sources[index]
        if not isinstance(page, dict) or page.get("page_read") is not True \
                or not str(page.get("url", "")).startswith("https://"):
            return False
        matched.append((index, quote))
    # At least one literal cited passage for each side. Both sides must
    # appear in answer prose and in source-backed quote strings.
    body = answer.split("\n\n[web", 1)[0]
    for name in case["entities"]:
        pattern = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I)
        if not pattern.search(body):
            return False
        if not any(pattern.search(quote) for _, quote in matched):
            return False
    if len({idx for idx, _ in matched}) < 2:
        return False
    # A comparison must not smuggle new quantified claims unsupported by its
    # literal quotes or user-provided quantities.
    approved = set(_NUMERIC.findall(case["question"] + "\n" +
                                     " ".join(q for _, q in matched)))
    if any(number not in approved for number in _NUMERIC.findall(body)):
        return False
    return True

def score_case(case, response, status):
    if not isinstance(case, dict) or case.get("category") not in CATEGORIES:
        raise ValueError("Unrecognized case category")
    if not isinstance(response, dict):
        response = {}
    seconds = response.get("seconds")
    measured = type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= 7200
    api_ok = status == 200 and measured and isinstance(response.get("text"), str)
    valid = False
    critical = False
    if api_ok and case["category"] == "web":
        valid = _web_integrity(case, response)
        critical = response.get("agent_checked") is True and not valid
    elif api_ok and case["category"] == "knowledge":
        valid = _static_answer(response["text"], case["gold"])
    elif api_ok:
        valid = _strict_integer(response["text"]) == case["gold"]
    # Only scalars, not model completions, questions, source URLs or accounts.
    return {
        "id": case["id"], "category": case["category"],
        "question_sha256": case["question_sha256"],
        "passed": bool(valid),
        "critical_unverified": bool(critical),
        "seconds": round(seconds, 4) if measured else None,
        "http_status": status if type(status) is int else None,
        "requested_tier": response.get("_requested_tier"),
        "executed_tier": response.get("tier"),
        "tool": response.get("tool_used"),
        "agent_checked": response.get("agent_checked") is True,
    }

def _latency(values):
    data = sorted(x for x in values if type(x) in (float, int) and math.isfinite(x))
    if not data:
        return {"n": 0, "median": None, "p95": None}
    return {"n": len(data),
            "median": round(statistics.median(data), 4),
            "p95": round(data[math.ceil(0.95*len(data))-1], 4)}

def summarize(cases, rows):
    validate_case_manifest(cases)
    if not isinstance(rows, list) or len(rows) != TOTAL:
        raise ValueError("Cannot score incomplete evaluation")
    lookup = {row.get("id"): row for row in rows if isinstance(row, dict)}
    if len(lookup) != TOTAL or set(lookup) != {c["id"] for c in cases}:
        raise ValueError("Missing or duplicate case IDs")
    cats = {
        name: {"n": 20, "passed": sum(r["passed"] is True for r in rows
                                      if r["category"] == name)}
        for name in CATEGORIES
    }
    score = sum(r["passed"] is True for r in rows)
    critical = sum(r["critical_unverified"] is True for r in rows)
    executed = sorted({r.get("executed_tier") for r in rows
                       if isinstance(r.get("executed_tier"), str)})
    return {
        "schema_version": 1,
        "status": "COMPLETE_100_CASES",
        "total": TOTAL,
        "manifest_sha256": manifest_sha256(cases),
        "categories": cats,
        "objective_oracle": {"passed": sum(cats[name]["passed"] for name
                                           in ("math", "logic", "programming", "knowledge")),
                             "n": 80},
        "web_citation_integrity": {"passed": cats["web"]["passed"], "n": 20,
                                   "not_equivalent_to_factual_correctness": True},
        "combined_passed": score,
        "critical_unverified": critical,
        "threshold_90_of_100_met": score >= TARGET and critical == 0,
        "latency_seconds": _latency([r["seconds"] for r in rows]),
        "latency_by_executed_tier": {
            name: _latency([r["seconds"] for r in rows if r.get("executed_tier") == name])
            for name in executed
        },
        "latency_by_category": {
            name: _latency([r["seconds"] for r in rows if r["category"] == name])
            for name in CATEGORIES
        },
        "v22_paired_comparison": "NOT_RUN",
        "production_promotion_allowed": False,
        "web_factual_correctness_independently_proven": False,
    }
