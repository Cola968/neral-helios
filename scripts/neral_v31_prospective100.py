"""Prospective, sealed, synthetic NERAL V3.3 100-case quality/latency gate.

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
    ("Wie heißt das chemische Element mit der Ordnungszahl 6? Nur den deutschen Namen.", "Kohlenstoff"),
    ("Wie heißt das chemische Element mit der Ordnungszahl 14? Nur den deutschen Namen.", "Silizium"),
    ("Wie heißt das chemische Element mit der Ordnungszahl 26? Nur den deutschen Namen.", "Eisen"),
    ("Wie heißt das chemische Element mit der Ordnungszahl 33? Nur den deutschen Namen.", "Arsen"),
    ("Wie heißt das chemische Element mit der Ordnungszahl 34? Nur den deutschen Namen.", "Selen"),
    ("Wie heißt das chemische Element mit der Ordnungszahl 35? Nur den deutschen Namen.", "Brom"),
    ("Wie heißt das chemische Element mit der Ordnungszahl 36? Nur den deutschen Namen.", "Krypton"),
    ("Wie heißt das chemische Element mit der Ordnungszahl 47? Nur den deutschen Namen.", "Silber"),
    ("Wie heißt das chemische Element mit der Ordnungszahl 50? Nur den deutschen Namen.", "Zinn"),
    ("Wie heißt das chemische Element mit der Ordnungszahl 53? Nur den deutschen Namen.", "Jod"),
    ("Wie heißt die Hauptstadt von Griechenland? Nur den deutschen Namen.", "Athen"),
    ("Wie heißt die Hauptstadt von Österreich? Nur den deutschen Namen.", "Wien"),
    ("Wie heißt die Hauptstadt von Belgien? Nur den deutschen Namen.", "Brüssel"),
    ("Wie heißt die Hauptstadt von Italien? Nur den deutschen Namen.", "Rom"),
    ("Wie heißt die Hauptstadt von Kanada? Nur den Namen.", "Ottawa"),
    ("Wie heißt die Hauptstadt von Finnland? Nur den Namen.", "Helsinki"),
    ("Wie heißt die Hauptstadt von Norwegen? Nur den Namen.", "Oslo"),
    ("Wie heißt die Hauptstadt von Kroatien? Nur den Namen.", "Zagreb"),
    ("Wie heißt die Hauptstadt von Slowenien? Nur den Namen.", "Ljubljana"),
    ("Wie heißt die Hauptstadt von Irland? Nur den Namen.", "Dublin"),
)
# Four wordings for five entity pairs = 20 real opt-in web tasks.
# These are factual expectations, not proof that a live page supports each claim.
WEB_PAIRS = (
    ("Firefox", "Edge", "Gecko", "Blink", "Browser-Engine"),
    ("Chrome", "Safari", "Blink", "WebKit", "Browser-Engine"),
    ("Python", "Rust", "Guido van Rossum", "Graydon Hoare", "Entwickler und Schöpfer"),
    ("Ruby", "JavaScript", "Yukihiro Matsumoto", "Brendan Eich", "Entwickler und Schöpfer"),
    ("FreeBSD", "NetBSD", "FreeBSD-Kernel", "NetBSD-Kernel", "Betriebssystem-Kernel"),
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
    seen_math=set()
    for i in range(20):
        f=i % 4
        for attempt in range(256):
            if f == 0:
                a,b,c=rng.randint(11,89),rng.randint(3,12),rng.randint(1,9)
                q=f"Ein Lager besitzt {a} Kisten mit jeweils {b} Bauteilen. {c} Bauteile fehlen. Wie viele Bauteile bleiben? Nur das ganzzahlige Ergebnis."
                gold=a*b-c
            elif f == 1:
                a,b=rng.randint(12,42),rng.randint(13,55)
                q=f"Berechne die Summe aller ganzen Zahlen von {a} bis einschließlich {b+a}. Antworte mit der ganzen Zahl."
                gold=sum(range(a,b+a+1))
            elif f == 2:
                a,b=rng.randint(4,16),rng.randint(5,15)
                q=f"Ein Rechteckgitter ist {a} Felder breit und {b} Felder hoch. Wie viele achsenparallele Rechtecke beliebiger Größe enthält das Gitter? Nur das ganzzahlige Ergebnis."
                gold=a*(a+1)*b*(b+1)//4
            else:
                a,b,c=rng.randint(18,90),rng.randint(3,12),rng.randint(4,9)
                q=f"Eine Sonde sendet {a*b*c} Bits mit {b*c} Bits pro Sekunde. Wie viele Sekunden dauert die Übertragung? Nur das ganzzahlige Ergebnis."
                gold=a
            if q not in seen_math:
                break
        else:
            raise ValueError("Unable to generate unique arithmetic question")
        seen_math.add(q)
        cases.append(_mk("mathematics",i+1,q,gold=gold,kind="integer",tool="python_sandbox"))

    from itertools import permutations
    unique_constraints = set()
    for i in range(20):
        names = ("Ada", "Ben", "Cem", "Dora")
        for _ in range(128):
            fixed = rng.randrange(4)
            excluded = rng.randrange(4)
            other = rng.randrange(4)
            triple = (fixed, excluded, other)
            if triple not in unique_constraints:
                unique_constraints.add(triple)
                break
        else:
            raise ValueError("Insufficient unique schedule cases")
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

    seen_programs=set()
    for i in range(20):
        f=i % 4
        for attempt in range(256):
            a,b,c=rng.randint(7,35),rng.randint(2,11),rng.randint(3,15)
            if f == 0:
                program=f"x={a}\ny={b}\nz={c}\nprint(x*y-z)"
                gold=a*b-c
            elif f == 1:
                program=f"nums=[{a},{b},{c}]\nprint(sum(nums)*2)"
                gold=2*(a+b+c)
            elif f == 2:
                program=f"x={a}\ny={b}\nfor _ in range(3):\n x=x+y\nprint(x)"
                gold=a+3*b
            else:
                program=f"n={a}\nm={b}\nk={c}\nprint((n+m)*k)"
                gold=(a+b)*c
            q=("Welchen ganzzahligen Wert gibt dieses Python-Programm aus? "
               "Berechne das Ergebnis mit einem ausgeführten Werkzeug; "
               "erfinde keine Ausführung.\n" + "'''python\n" + program + "\n'''")
            if q not in seen_programs:
                break
        else:
            raise ValueError("Unable to generate unique programming question")
        seen_programs.add(q)
        cases.append(_mk("programming",i+1,q,gold=gold,
                         kind="integer",tool="python_sandbox"))

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


def _number_correct(text: str, gold: object) -> bool:
    from decimal import Decimal, InvalidOperation
    parsed = canonical_integer(text)
    if type(gold) is int:
        return parsed == gold
    if type(gold) not in (float, Decimal):
        return False
    if parsed is None:
        match = re.fullmatch(r"(?is)\s*(?:Antwort|Ergebnis|Resultat)\s*:\s*(-?\d+(?:[.,]\d+)?)\s*",text)
        if match is None:
            return False
        try:
            result = Decimal(match[1].replace(",","."))
        except InvalidOperation:
            return False
    else:
        result = Decimal(parsed)
    target = Decimal(str(gold))
    # No tolerance on discrete integers. Limited relative/absolute tolerance
    # applies only to explicitly non-integer gold values.
    return abs(result-target) <= max(Decimal("0.000001"),abs(target)*Decimal("0.000000001"))


def _web_facts_in_answer(case: dict, text: str) -> bool:
    """Prove that each entity's portion contains the expected fact.

    Search the ENTIRE answer section, not a brittle 180-character window.
    Avoid credit from the other entity's section or citations alone.
    """
    body = re.split(r"\n\s*\[web\d+\]\s*„",text,maxsplit=1)[0]
    pairs = list(zip(case.get("entities",[]),case.get("facts",[])))
    if len(pairs)<2:
        return False
    for index,(name,fact) in enumerate(pairs):
        own=re.search(r"(?im)^\s*"+re.escape(name)+r"\s*:",body)
        if own is None:
            return False
        other_sections=[
            m.start() for other,_ in pairs if other!=name
            for m in [re.search(r"(?im)^\s*"+re.escape(other)+r"\s*:",body)]
            if m is not None and m.start()>own.start()
        ]
        fragment=body[own.end():min(other_sections) if other_sections else len(body)]
        normalized=re.sub(r"[^\w]+"," ",fragment.casefold()).strip()
        expected=re.sub(r"[^\w]+"," ",fact.casefold()).strip()
        if expected not in normalized:
            return False
    return True


def _word_correct(text: str, gold: str) -> bool:
    synonyms={
        "straßburg":{"strasbourg","straßburg"},
        "lissabon":{"lisboa","lissabon"},
        "helsinki":{"helsinki","helsingfors"},
        "tallinn":{"tallinn"},
        "mond":{"mond"},
    }
    response=answer_token(text)
    expected=gold.casefold()
    return response in synonyms.get(expected,{expected})


def evaluate_case(case: dict, response: dict, status: int) -> dict:
    """Orthogonal factual correctness and independently evidenced execution.

    Failure classification never hides a numerically correct, unverified
    answer. No score is promoted from model-written claims of tool execution.
    """
    response=response if type(response) is dict else {}
    value=response.get("text","")
    text=value if type(value) is str else ""
    kind=case.get("kind")
    if status!=200:
        return {"answer_correct":False,"tool_verified":False,
                "strict_e2e_pass":False,"correct":False,"error":"http_error"}
    if kind=="word":
        correct=_word_correct(text,case.get("gold",""))
        return {"answer_correct":correct,"tool_verified":False,
                "strict_e2e_pass":correct,"correct":correct,
                "error":None if correct else "wrong_knowledge"}
    correct=False
    if kind=="integer":
        correct=_number_correct(text,case.get("gold"))
    elif kind=="web":
        correct=_web_facts_in_answer(case,text)
    tool=response.get("tool_used")
    permitted_error={"no_tool_called","sandbox_error","z3_schema_error","z3_unsat",
                     "checker_veto","format_mismatch","source_evidence_missing"}
    explicit=response.get("error_code")
    if explicit not in permitted_error:
        explicit=None
    verified=False
    if kind=="integer":
        expected=case.get("tool")
        # The explicit V3.2 verifier returns tool_verified. Existing V3
        # checks are recognized only when the tool identity and checker status
        # are both present; this is backward-compatible, not an independent
        # correctness certificate.
        verified=tool==expected and (
            response.get("tool_verified") is True or
            response.get("agent_checked") is True)
    elif kind=="web":
        sources=response.get("sources",[])
        valid=isinstance(sources,list) and all(isinstance(s,dict) for s in sources)
        read=[(i+1,s) for i,s in enumerate(sources) if s.get("page_read") is True] if valid else []
        citations=set(re.findall(r"\[web(\d+)\]\s*„[^“]{8,650}“",text))
        verified=(tool=="direct_reasoning" and
                  (response.get("tool_verified") is True or
                   response.get("agent_checked") is True) and
                  len(read)>=len(case.get("entities",[])) and
                  len(citations)>=len(case.get("entities",[])) and
                  all(1<=int(n)<=len(sources) and
                      sources[int(n)-1].get("page_read") is True
                      for n in citations))
    strict=bool(correct and verified)
    error=None
    if not strict:
        if explicit is not None:
            error=explicit
        elif not text.strip():
            error="format_mismatch"
        elif not tool or (kind=="integer" and tool!=case.get("tool")):
            error="no_tool_called"
        elif not verified:
            error="checker_veto"
        elif not correct and kind=="integer" and canonical_integer(text) is None:
            error="format_mismatch"
        else:
            error="answer_incorrect"
    return {"answer_correct":bool(correct),"tool_verified":bool(verified),
            "strict_e2e_pass":strict,"correct":strict,"error":error}


def summarize(cases: list[dict], rows: list[dict]) -> dict:
    if len(cases)!=100 or len(rows)!=100:
        raise ValueError("Exactly 100 full observations are mandatory")
    ids=[x["id"] for x in cases]
    result_ids=[x.get("id") for x in rows]
    if len(set(ids))!=100 or len(set(result_ids))!=100 or set(ids)!=set(result_ids):
        raise ValueError("Missing or duplicate evaluated case")
    durations=[x.get("latency_seconds") for x in rows]
    if not all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=3600
               for v in durations):
        raise ValueError("Missing or invalid latency")
    byid={x["id"]:x for x in rows}
    def answer(r):
        return r.get("answer_correct",r.get("correct")) is True
    def strict(r):
        return r.get("strict_e2e_pass",r.get("correct")) is True
    grouped={}
    for category in CATEGORIES:
        subset=[byid[c["id"]] for c in cases if c["category"]==category]
        if len(subset)!=20:
            raise ValueError("Category size mismatch")
        uses=sum(bool(r.get("tool")) for r in subset)
        grouped[category]={
            "count":20,
            "answer_correct":sum(answer(x) for x in subset),
            "tool_verified":sum(x.get("tool_verified") is True for x in subset),
            "tool_use_count":uses,
            "tool_use_percent":round(100*uses/20,1),
            "strict_e2e_pass":sum(strict(x) for x in subset),
            "correct":sum(strict(x) for x in subset),
            "average_latency_seconds":round(statistics.mean(x["latency_seconds"] for x in subset),3),
            "errors":dict(Counter(x.get("error") or "unclassified" for x in subset
                                  if not strict(x))),
        }
    good=sum(strict(x) for x in rows)
    by_tier={}
    for tier in TIERS:
        ts=[x for x in rows if x.get("requested_tier")==tier]
        if ts:
            by_tier[tier]={
                "count":len(ts),"correct":sum(strict(x) for x in ts),
                "median_seconds":round(statistics.median(x["latency_seconds"] for x in ts),3)}
    return {
        "status":"COMPLETE","cases":100,"correct":good,
        "answer_correct":sum(answer(x) for x in rows),
        "tool_verified":sum(x.get("tool_verified") is True for x in rows),
        "strict_e2e_pass":good,
        "categories":grouped,
        "latency_average_seconds":round(statistics.mean(durations),3),
        "latency_median_seconds":round(statistics.median(durations),3),
        "latency_p95_seconds":round(sorted(durations)[94],3),
        "tiers":by_tier,"gate_90_of_100":good>=90,
        "baseline_v22":"NOT_RUN_NO_SIDE_BY_SIDE_SERVICE",
        "model_promotion_authorized":False,
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
    # The homepage is deliberately HTML. Visit it to establish the signed
    # Flask session cookie, then parse JSON only from the session endpoint.
    with opener.open(Request(BASE + "/", method="GET"), timeout=12) as home:
        if home.status != 200:
            raise RuntimeError("Local app unavailable")
        home.read(65536)
    code, session = fetch("/api/neral/preview/session")
    if (code != 200 or session.get("three_tier_enabled") is not True or
            session.get("evidence_agent_enabled") is not True or
            not isinstance(session.get("csrf"), str)):
        raise RuntimeError("Active service is not NERAL V3 with evidence agent")
    return fetch, session["csrf"]


def run_live(args) -> dict:
    if args.authorize != "NERAL_V32_OWNER_APPROVED_NEW_100":
        raise PermissionError("Explicit V3.2 prospective 100-run approval required")
    for path in (args.output,args.sealed,args.progress,args.details):
        if not Path(path).is_absolute():
            raise ValueError("Use absolute runner-temp output paths")
    seed=secrets.token_hex(16)
    cases=build_suite(seed)
    seal_digest=seal_suite(Path(args.sealed),cases)
    fetch,csrf=_open_client()
    rows=[]
    progress=Path(args.progress)
    details=Path(args.details)
    progress.parent.mkdir(parents=True,exist_ok=True)
    details.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create mode prevents accidental overwriting of earlier evidence.
    with progress.open("x",encoding="utf8") as saved, details.open("x",encoding="utf8") as audit:
        for index,case in enumerate(cases):
            tier=TIERS[index%len(TIERS)]
            started=time.monotonic()
            response={}
            status=0
            try:
                payload={"tier":tier,"prompt":case["question"],"history":[],
                         "force_sandbox":False}
                if case["category"]=="web":
                    payload["web_query"]=case["web_query"]
                status,response=fetch("/api/neral/chat/three-tier",payload,csrf)
                score=evaluate_case(case,response,status)
                effective=response.get("tier")
                tool=response.get("tool_used")
            except HTTPError as exc:
                status=exc.code
                score=evaluate_case(case,{},status)
                effective,tool=None,None
            except (URLError,TimeoutError,ValueError,OSError) as exc:
                score={"answer_correct":False,"tool_verified":False,
                       "strict_e2e_pass":False,"correct":False,
                       "error":type(exc).__name__}
                effective,tool=None,None
            elapsed=round(time.monotonic()-started,3)
            row={
                "id":case["id"],"category":case["category"],
                "requested_tier":tier,"effective_tier":effective,
                "tool":tool if isinstance(tool,str) else None,
                "answer_correct":score["answer_correct"],
                "tool_verified":score["tool_verified"],
                "strict_e2e_pass":score["strict_e2e_pass"],
                "correct":score["strict_e2e_pass"],
                "error":score["error"],"latency_seconds":elapsed,
            }
            rows.append(row)
            saved.write(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n")
            saved.flush()
            os.fsync(saved.fileno())
            # Every prompt is synthetic and generated by this script. Never
            # publish browser session cookies, CSRF, private chats or source
            # page_text. This record retains enough raw model/tool output to
            # distinguish routing, validation, formatting and correctness.
            response=response if isinstance(response,dict) else {}
            allowed=("text","tool_used","tool_verified","agent_checked",
                     "agent_steps","agent_attempts","error_code","verified",
                     "verification_status","proof_backend","sources",
                     "sandbox_executed","sandbox_attempts",
                     "sandbox_failure_code","tool_calls_succeeded","tier","model",
                     "web_model_diagnostic")
            observed={k:response[k] for k in allowed if k in response}
            if isinstance(observed.get("text"),str):
                observed["text"]=observed["text"][:12000]
            # The user explicitly requested raw model-output diagnostics for
            # synthetic test cases, with bounded size and no source page text.
            diagnostic=observed.get("web_model_diagnostic")
            if isinstance(diagnostic,dict):
                observed["web_model_diagnostic"]={
                    "raw_output":str(diagnostic.get("raw_output",""))[:800],
                    "exception":str(diagnostic.get("exception",""))[:500]}
            else:
                observed.pop("web_model_diagnostic",None)
            if isinstance(observed.get("sources"),list):
                observed["sources"]=[
                    {k:v for k,v in entry.items() if k in
                     ("url","page_read","entity_focus","title","retrieved_url")}
                    for entry in observed["sources"] if isinstance(entry,dict)
                ][:12]
            detail={"case":case,"request_tier":tier,"http_status":status,
                    "observed":observed,"grader":score,
                    "elapsed_seconds":elapsed}
            audit.write(json.dumps(detail,ensure_ascii=False,sort_keys=True)+"\n")
            audit.flush()
            os.fsync(audit.fileno())
            # External web search enforces a provider rate limit. The pause is
            # outside recorded inference latency and cannot inflate scores.
            if case["category"]=="web":
                time.sleep(35)
    report=summarize(cases,rows)
    report["sealed_manifest_sha256"]=seal_digest
    report["details_sha256"]=hashlib.sha256(details.read_bytes()).hexdigest()
    report["detail_count"]=len(rows)
    report["source_sha"]=os.environ.get("GITHUB_SHA","")
    report["synthetic_individual_rows_retained"]=True
    report["baseline_v22"]="NOT_RUN_NO_SIDE_BY_SIDE_SERVICE"
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf8") as handle:
        json.dump(report,handle,ensure_ascii=False,indent=2,sort_keys=True)
    print("NERAL_V32_PROSPECTIVE100_SCALAR "+
          json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authorize", required=True)
    parser.add_argument("--sealed", required=True, type=Path)
    parser.add_argument("--progress", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--details", required=True, type=Path)
    args = parser.parse_args(argv)
    result = run_live(args)
    # Not meeting 90/100 is a measurement result, not a CI infrastructure
    # failure. The report records the actual score for follow-up work.
    return 0 if result["status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
