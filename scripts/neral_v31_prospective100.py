"""Prospective, sealed, synthetic NERAL V3.1 100-case quality/latency gate.

Never modifies the installed service, personal files or model weights.
The random seed and individual outcomes remain only in the runner temp folder.
Only aggregate scalar diagnostics are published. This is NOT a proof of
universal accuracy, nor a paired comparison with V2.2.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import http.cookiejar
import json
import math
import os
from pathlib import Path
import random
import re
import secrets
import statistics
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener, HTTPCookieProcessor, ProxyHandler

CATEGORIES = ("mathematics", "logic", "programming", "knowledge", "web")
TIERS = ("Luna", "Terra", "Helios")
BASE = "http://127.0.0.1:9690"
MAX_RESPONSE_BYTES = 262_144

# Stable reference answers, frozen before prospective inference.
KNOWLEDGE = (
    ("Wie heißt das chemische Element mit der Ordnungszahl 8? Nur den deutschen Namen.", "Sauerstoff"),
    ("Wie heißt das chemische Element mit der Ordnungszahl 79? Nur den deutschen Namen.", "Gold"),
    ("Wie heißt die SI-Einheit der elektrischen Stromstärke? Nur den Namen.", "Ampere"),
    ("Wie heißt die SI-Einheit der elektrischen Spannung? Nur den Namen.", "Volt"),
    ("Wie heißt die SI-Einheit der Frequenz? Nur den Namen.", "Hertz"),
    ("Wie heißt die SI-Einheit der elektrischen Leistung? Nur den Namen.", "Watt"),
    ("Wie heißt die SI-Einheit der elektrischen Kapazität? Nur den Namen.", "Farad"),
    ("Wie heißt die SI-Einheit der elektrischen Ladung? Nur den Namen.", "Coulomb"),
    ("Wie heißt die SI-Einheit des elektrischen Widerstands? Nur den Namen.", "Ohm"),
    ("Wie heißt die SI-Einheit des Drucks? Nur den Namen.", "Pascal"),
    ("Wie heißt die SI-Einheit der Kraft? Nur den Namen.", "Newton"),
    ("Wie heißt der größte Planet unseres Sonnensystems? Nur den Namen.", "Jupiter"),
    ("Welcher Planet unseres Sonnensystems ist der Sonne am nächsten? Nur den Namen.", "Merkur"),
    ("Welcher Planet ist für seine besonders auffälligen Ringe bekannt? Nur den Namen.", "Saturn"),
    ("Welcher Planet hat den größten bekannten Vulkan des Sonnensystems, Olympus Mons? Nur den Namen.", "Mars"),
    ("Wie heißt der natürliche Satellit der Erde? Nur den Namen.", "Mond"),
    ("In welcher Stadt befindet sich der Sitz des Europäischen Parlaments für die Plenartagungen? Nur den Stadtnamen.", "Straßburg"),
    ("Wie heißt die Hauptstadt von Portugal? Nur den Namen.", "Lissabon"),
    ("Wie heißt die Hauptstadt von Finnland? Nur den Namen.", "Helsinki"),
    ("Wie heißt die Hauptstadt von Estland? Nur den Namen.", "Tallinn"),
)
# Four wordings for five entity pairs = 20 real opt-in web tasks.
# These are factual expectations, not proof that a live page supports each claim.
WEB_PAIRS = (
    ("Firefox", "Chrome", "Gecko", "Blink", "Browser-Engine"),
    ("Firefox", "Safari", "Gecko", "WebKit", "Browser-Engine"),
    ("Chrome", "Edge", "Blink", "Blink", "Browser-Engine"),
    ("Python", "Ruby", "Guido van Rossum", "Yukihiro Matsumoto", "Entwickler und Schöpfer"),
    ("Linux", "Windows", "Linux-Kernel", "Windows NT", "Betriebssystem-Kernel"),
)
WEB_WORDINGS = (
    "Suche Webseiten und vergleiche {a} mit {b}: {topic}. Belege beide mit gelesenen Quellen.",
    "Vergleich {a} und {b}: Was unterscheidet sie bei {topic}? Zitiere die gelesenen Quellen.",
    "Unterschied zwischen {a} und {b}: {topic}. Verwende belegte Webseiten.",
    "{a} vs {b}: Vergleiche {topic} anhand tatsächlich gelesener Webseiten.",
)


def sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def _mk(category: str, number: int, question: str, **fields) -> dict:
    if not isinstance(question, str) or len(question) > 1000:
        raise ValueError("Unbounded synthetic case")
    return {
        "id": f"v31-{category}-{number:02d}",
        "category": category, "question": question,
        "question_sha256": sha256(question),
        **fields,
    }


def build_suite(seed: str) -> list[dict]:
    """Create a prospective, deterministic 20-per-category suite for a secret seed."""
    if not isinstance(seed, str) or not re.fullmatch(r"[0-9a-fA-F]{32,64}", seed):
        raise ValueError("Private 128-bit or 256-bit seed required")
    rng = random.Random(int(seed, 16))
    cases = []
    for i in range(20):
        f = i % 4
        if f == 0:
            a, b, c = rng.randint(11, 89), rng.randint(3, 12), rng.randint(1, 9)
            q = f"Ein Lager besitzt {a} Kisten mit jeweils {b} Bauteilen. {c} Bauteile fehlen. Wie viele Bauteile bleiben? Nur das ganzzahlige Ergebnis."
            gold = a*b-c
        elif f == 1:
            a, b = rng.randint(12, 42), rng.randint(13, 55)
            q = f"Berechne die Summe aller ganzen Zahlen von {a} bis einschließlich {b+a}. Antworte mit der ganzen Zahl."
            gold = sum(range(a, b+a+1))
        elif f == 2:
            a, b = rng.randint(4, 16), rng.randint(5, 15)
            q = f"Ein Rechteckgitter ist {a} Felder breit und {b} Felder hoch. Wie viele achsenparallele Rechtecke beliebiger Größe enthält das Gitter? Nur das ganzzahlige Ergebnis."
            gold = a*(a+1)*b*(b+1)//4
        else:
            a, b, c = rng.randint(18, 90), rng.randint(3, 12), rng.randint(4, 9)
            q = f"Eine Sonde sendet {a*b*c} Bits mit {b*c} Bits pro Sekunde. Wie viele Sekunden dauert die Übertragung? Nur das ganzzahlige Ergebnis."
            gold = a
        cases.append(_mk("mathematics", i+1, q, gold=gold, kind="integer", tool="python_sandbox"))

    from itertools import permutations
    for i in range(20):
        names = ("Ada", "Ben", "Cem", "Dora")
        fixed = rng.randrange(4)
        excluded = rng.randrange(4)
        other = rng.randrange(4)
        # One-to-one assignment to four ordered shifts; conditions are fully
        # enumerated independently of NERAL's Z3 tool.
        assignments = tuple(p for p in permutations(range(1, 5))
                            if p[0] == fixed+1 and p[1] != excluded+1
                            and p[2] != other+1)
        q = ("Vier Mitarbeitende Ada, Ben, Cem und Dora erhalten jeweils genau "
             "eine der Schichten 1, 2, 3, 4. Jede Schicht ist genau einmal "
             f"besetzt. Ada arbeitet Schicht {fixed+1}, Ben darf nicht Schicht "
             f"{excluded+1} und Cem darf nicht Schicht {other+1} arbeiten. "
             "Wie viele zulässige vollständige Schichtpläne gibt es? "
             "Zähle mit Z3 alle Lösungen und gib die Anzahl aus.")
        cases.append(_mk("logic", i+1, q, gold=len(assignments),
                         kind="integer", tool="z3_solver"))

    for i in range(20):
        f = i % 4
        a, b, c = rng.randint(7, 35), rng.randint(2, 11), rng.randint(3, 15)
        if f == 0:
            program = f"x={a}\ny={b}\nz={c}\nprint(x*y-z)"
            gold = a*b-c
        elif f == 1:
            program = f"nums=[{a},{b},{c}]\nprint(sum(nums)*2)"
            gold = 2*(a+b+c)
        elif f == 2:
            program = f"x={a}\ny={b}\nfor _ in range(3):\n x=x+y\nprint(x)"
            gold = a+3*b
        else:
            program = f"n={a}\nm={b}\nk={c}\nprint((n+m)*k)"
            gold = (a+b)*c
        q = ("Welchen ganzzahligen Wert gibt dieses Python-Programm aus? "
             "Berechne das Ergebnis mit einem ausgeführten Werkzeug; "
             "erfinde keine Ausführung.\n" + "'''python\n" + program + "\n'''")
        cases.append(_mk("programming", i+1, q, gold=gold,
                         kind="integer", tool="python_sandbox"))
    for i, (q, answer) in enumerate(KNOWLEDGE, 1):
        cases.append(_mk("knowledge", i, q, gold=answer, kind="word"))
    for i, (a, b, fa, fb, topic) in enumerate(WEB_PAIRS):
        for j, wording in enumerate(WEB_WORDINGS):
            q = wording.format(a=a,b=b,topic=topic)
            cases.append(_mk("web", i*4+j+1, q,
                             entities=[a,b], facts=[fa,fb], kind="web",
                             web_query=f"{a} {b} {topic}"))
    if (len(cases) != 100 or
            len({case["question_sha256"] for case in cases}) != 100 or
            Counter(c["category"] for c in cases) != {c: 20 for c in CATEGORIES}):
        raise ValueError("Prospective suite invalid or duplicates found")
    rng.shuffle(cases)
    return cases


def seal_suite(path: Path, cases: list[dict]) -> str:
    """Write gold/questions once, before any local inference."""
    raw = (json.dumps({"version": 1, "cases": cases},
                      ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf8")
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(raw).hexdigest()


def canonical_integer(text: object) -> int | None:
    """Extract only an unambiguous tool result, never arbitrary numbers in prose."""
    if not isinstance(text, str) or not text.strip() or len(text) > 12000:
        return None
    for pattern in (r"(?im)^Z3:\s*(-?\d+)\s*$",
                    r"(?im)^(?:Ergebnis|Resultat|Antwort|Result|Summe|Anzahl|Wert)\s*:\s*(-?\d+)(?:\s+\w+)?\s*$",
                    r"(?m)^[^\n:]{2,65}:\s*(-?\d+)(?:\s+\w+)?\s*(?:\n|$)",
                    r"^\s*(-?\d+)\s*$"):
        matches = re.findall(pattern, text)
        if len(matches) == 1:
            return int(matches[0])
    return None


def answer_token(text: object) -> str:
    if not isinstance(text, str) or len(text) > 256:
        return ""
    compact = re.sub(r"^\s*(?:Antwort|Ergebnis)\s*:\s*", "", text.strip(), flags=re.I)
    compact = compact.strip(" \n\t*._!")
    if any(c in compact for c in ("\n", ";", ",", "?", ":")):
        return ""
    return compact.casefold()


def evaluate_case(case: dict, response: dict, status: int) -> dict:
    """No partial credit for missing tool execution, fake sources or bad HTTP."""
    text = response.get("text", "") if isinstance(response, dict) else ""
    if status != 200:
        return {"correct": False, "error": "http_error"}
    if type(text) is not str or not text.strip():
        return {"correct": False, "error": "empty_answer"}
    if case.get("kind") == "word":
        return {"correct": answer_token(text) == case.get("gold", "").casefold(),
                "error": None if answer_token(text) == case.get("gold", "").casefold() else "wrong_knowledge"}
    if response.get("agent_checked") is not True:
        return {"correct": False, "error": "not_agent_checked"}
    if case.get("kind") == "integer":
        if response.get("tool_used") != case.get("tool"):
            return {"correct": False, "error": "wrong_tool"}
        if case.get("tool") == "z3_solver" and "Z3:" not in text:
            return {"correct": False, "error": "missing_z3"}
        match = canonical_integer(text)
        return {"correct": match == case.get("gold"),
                "error": None if match == case.get("gold") else "wrong_integer"}
    if case.get("kind") == "web":
        if response.get("tool_used") != "direct_reasoning":
            return {"correct": False, "error": "wrong_web_tool"}
        sources = response.get("sources")
        if not isinstance(sources, list) or sum(s.get("page_read") is True
                for s in sources if isinstance(s, dict)) < len(case["entities"]):
            return {"correct": False, "error": "insufficient_read_pages"}
        quotes = set(re.findall(r"\[web(\d+)\]\s*„[^“]{8,650}“", text))
        if len(quotes) < len(case["entities"]) or any(
                int(id_) not in range(1, len(sources)+1) or
                sources[int(id_)-1].get("page_read") is not True for id_ in quotes):
            return {"correct": False, "error": "missing_literal_citations"}
        body = text.split("\n\n[web", 1)[0]
        for entity, fact in zip(case["entities"], case["facts"]):
            pattern = r"(?i)(?<!\w)" + re.escape(entity) + r"(?!\w)"
            matches = list(re.finditer(pattern, body))
            if not matches:
                return {"correct": False, "error": "missing_entity"}
            if not any(re.search(re.escape(fact), body[m.end():m.end()+180], re.I)
                       for m in matches):
                return {"correct": False, "error": "missing_expected_fact"}
        return {"correct": True, "error": None}
    return {"correct": False, "error": "unknown_kind"}


def summarize(cases: list[dict], rows: list[dict]) -> dict:
    if len(cases) != 100 or len(rows) != 100:
        raise ValueError("Exactly 100 full observations are mandatory")
    ids = [x["id"] for x in cases]
    result_ids = [x.get("id") for x in rows]
    if len(set(ids)) != 100 or len(set(result_ids)) != 100 or set(ids) != set(result_ids):
        raise ValueError("Missing or duplicate evaluated case")
    outcomes = {x["id"]:x for x in rows}
    grouped = {}
    for category in CATEGORIES:
        subset = [outcomes[c["id"]] for c in cases if c["category"] == category]
        if len(subset) != 20:
            raise ValueError("Category size mismatch")
        grouped[category] = {
            "count": 20,
            "correct": sum(x.get("correct") is True for x in subset),
            "errors": dict(Counter(x.get("error") or "wrong" for x in subset
                                   if x.get("correct") is not True)),
        }
    durations = [x.get("latency_seconds") for x in rows]
    if not all(type(x) in (int,float) and math.isfinite(x) and 0 <= x <= 3600
               for x in durations):
        raise ValueError("Missing or invalid latency")
    good = sum(x.get("correct") is True for x in rows)
    by_tier = {}
    for tier in TIERS:
        ts = [x for x in rows if x.get("requested_tier") == tier]
        if ts:
            by_tier[tier] = {
                "count": len(ts),
                "correct": sum(x.get("correct") is True for x in ts),
                "median_seconds": round(statistics.median(x["latency_seconds"] for x in ts), 3),
            }
    return {
        "status": "COMPLETE",
        "cases": 100,
        "correct": good,
        "categories": grouped,
        "latency_median_seconds": round(statistics.median(durations), 3),
        "latency_p95_seconds": round(sorted(durations)[94], 3),
        "tiers": by_tier,
        "gate_90_of_100": good >= 90,
        "unverified_web_successes": 0,  # checker required in evaluate_case
        "baseline_v22": "NOT_RUN_NO_SIDE_BY_SIDE_SERVICE",
        "model_promotion_authorized": False,
    }


def _open_client():
    cookie = http.cookiejar.CookieJar()
    opener = build_opener(ProxyHandler({}), HTTPCookieProcessor(cookie))
    def fetch(path: str, payload: dict | None = None, csrf: str | None = None):
        headers = {"Accept": "application/json"}
        body = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            headers["X-Neral-CSRF"] = csrf or ""
            body = json.dumps(payload, ensure_ascii=False).encode("utf8")
        req = Request(BASE + path, data=body, headers=headers,
                      method="POST" if body is not None else "GET")
        with opener.open(req, timeout=360 if body else 12) as f:
            data = f.read(MAX_RESPONSE_BYTES+1)
            if len(data) > MAX_RESPONSE_BYTES:
                raise ValueError("Oversized local API response")
            return f.status, json.loads(data)
    status, _ = fetch("/")
    if status != 200:
        raise RuntimeError("Local app unavailable")
    code, session = fetch("/api/neral/preview/session")
    if (code != 200 or session.get("three_tier_enabled") is not True or
            session.get("evidence_agent_enabled") is not True or
            not isinstance(session.get("csrf"), str)):
        raise RuntimeError("Active service is not NERAL V3 with evidence agent")
    return fetch, session["csrf"]


def run_live(args) -> dict:
    if args.authorize != "NERAL_V31_OWNER_APPROVED_PROSPECTIVE_100":
        raise PermissionError("Explicit prospective 100-run approval required")
    if not Path(args.output).is_absolute() or not Path(args.sealed).is_absolute():
        raise ValueError("Use absolute runner-temp output paths")
    seed = secrets.token_hex(16)
    cases = build_suite(seed)
    seal_digest = seal_suite(Path(args.sealed), cases)
    # Freeze the complete source and gold before creating any model response.
    fetch, csrf = _open_client()
    rows = []
    progress = Path(args.progress)
    progress.parent.mkdir(parents=True, exist_ok=True)
    with progress.open("x", encoding="utf8") as saved:
        for index, case in enumerate(cases):
            # Rotate requested levels; effective model routing is recorded
            # separately, so this does not claim independent model arms.
            tier = TIERS[index % len(TIERS)]
            started = time.monotonic()
            try:
                payload = {"tier":tier, "prompt":case["question"], "history":[],
                           "force_sandbox": False}
                if case["category"] == "web":
                    payload["web_query"] = case["web_query"]
                status, response = fetch("/api/neral/chat/three-tier", payload, csrf)
                score = evaluate_case(case, response, status)
                effective = response.get("tier")
                tool = response.get("tool_used")
            except HTTPError as exc:
                score = {"correct": False, "error": f"http_{exc.code}"}
                effective,tool = None,None
            except (URLError, TimeoutError, ValueError, OSError) as exc:
                score = {"correct": False, "error": type(exc).__name__}
                effective,tool = None,None
            elapsed = round(time.monotonic()-started, 3)
            row = {"id":case["id"], "category":case["category"],
                   "requested_tier":tier, "effective_tier":effective,
                   "tool":tool if isinstance(tool,str) else None,
                   "correct":score["correct"], "error":score["error"],
                   "latency_seconds":elapsed}
            rows.append(row)
            saved.write(json.dumps(row, ensure_ascii=False, separators=(",",":"))+"\n")
            saved.flush()
            os.fsync(saved.fileno())
            # Current WebSearch permits 6 outbound queries per minute.
            # Comparisons call the search provider per entity, possibly again
            # on evidence-retry. A conservative gap avoids self-induced 429s.
            if case["category"] == "web":
                time.sleep(35)
    report = summarize(cases, rows)
    report["sealed_manifest_sha256"] = seal_digest
    report["source_sha"] = os.environ.get("GITHUB_SHA", "")
    report["case_questions_or_answers_in_report"] = False
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf8") as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,sort_keys=True)
    print("NERAL_V31_PROSPECTIVE100_SCALAR " +
          json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize", required=True)
    parser.add_argument("--sealed", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_live(args)
    # Not meeting 90/100 is a measurement result, not a CI infrastructure
    # failure. The report records the actual score for follow-up work.
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
