"""Generate a private, new NERAL blind set on the Windows owner, never in git.

Gold labels come from local arithmetic/program specifications or independent,
time-stamped RSS/GitHub API snapshots; no model produces the expected answers.
"""
from __future__ import annotations
from collections import Counter
from datetime import datetime, timezone, timedelta
from email.utils import parsedate_to_datetime
from hashlib import sha256
import json
import os
from pathlib import Path
import random
import re
import secrets
from urllib.request import Request, urlopen
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from scripts.neral_v41_blind200 import CATEGORIES, validate_suite

FEEDS = [
    ("Tagesschau", "https://www.tagesschau.de/xml/rss2/"),
    ("NASA", "https://www.nasa.gov/feed/"),
    ("ESA", "https://www.esa.int/rssfeed/Our_Activities/Space_Science"),
    ("heise", "https://www.heise.de/rss/heise-atom.xml"),
    ("Phys.org", "https://phys.org/rss-feed/"),
    ("BBC Technology", "https://feeds.bbci.co.uk/news/technology/rss.xml"),
    ("Guardian World", "https://www.theguardian.com/world/rss"),
    ("Space.com", "https://www.space.com/feeds/all"),
    ("arXiv astronomy", "https://export.arxiv.org/rss/astro-ph"),
    ("Euronews", "https://www.euronews.com/rss?level=theme&name=travel"),
]
REPOS = [
    "python/cpython","microsoft/vscode","pytorch/pytorch",
    "godotengine/godot","rust-lang/rust","pandas-dev/pandas",
    "numpy/numpy","blender/blender","tensorflow/tensorflow",
    "scikit-learn/scikit-learn",
]
TOPICS = [
    "Stadtwärmeinsel", "elektrisches Feld", "Binnenschifffahrt", "Wetterfront",
    "Wasserkreislauf", "Satellitenorbit", "Wärmepumpe", "Digitalprivatsphäre",
    "Regelungstechnik", "Klimadiagramm", "Open-Source-Lizenzen",
    "Phasendiagramm", "Auftrieb am Flügel", "Solarzelle", "Kontinentaldrift",
    "Datenbankschlüssel", "Luftdruck", "Reibung", "Schlafrhythmus",
    "Entropie", "Lichtbrechung", "Zugverspätungen", "Wissenschaftsethik",
    "Mikroplastik", "Sternentstehung", "Akkudegradation", "Fehlerfortpflanzung",
    "Risikomanagement", "Zellatmung", "Rotverschiebung", "Hochwasser",
    "Flächenversiegelung", "Mikrocontroller", "Dopplereffekt", "Netzneutralität",
]
AUDIENCES = ["einen interessierten Jugendlichen", "eine Berufsschulklasse",
             "eine sachliche Radiosendung", "einen neugierigen Laien"]
WORDS = ["Komet", "Kran", "Segel", "Düne", "Orbit", "Pegel", "Wolke",
         "Fjord", "Falte", "Signal", "Mosaik", "Molekül"]


def norm(text: str) -> str:
    return " ".join(re.findall(r"\w+", text.casefold(), flags=re.UNICODE))


def _prior_rows(paths: list[Path]) -> tuple[set[str], set[str]]:
    seen, provenance = set(), set()
    if not paths:
        raise ValueError("No historical local V3.9/V3.10 archive paths supplied")
    files = []
    for root in paths:
        if not root.exists() or root.is_symlink():
            continue
        files.extend(sorted(root.rglob("*.jsonl")) if root.is_dir() else [root])
    for file in files[:500]:
        if file.is_symlink() or not file.is_file() or file.stat().st_size > 64*1024*1024:
            continue
        try:
            with file.open(encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row=json.loads(line)
                    if type(row) is not dict:
                        continue
                    meta=row.get("meta") if type(row.get("meta")) is dict else {}
                    source=meta.get("source")
                    if source in ("v39_benchmark", "live_verified"):
                        provenance.add(source)
                    for key in ("instruction","question","prompt"):
                        value=row.get(key)
                        if type(value) is str:
                            seen.add(norm(value))
                    nested=row.get("case")
                    if type(nested) is dict:
                        for key in ("question","prompt"):
                            if type(nested.get(key)) is str:
                                seen.add(norm(nested[key]))
        except (OSError, UnicodeError, ValueError):
            raise ValueError(f"Cannot audit archive {file.name}")
    if not {"v39_benchmark", "live_verified"} <= provenance:
        raise ValueError("V3.9/V3.10 provenance missing; refuse blind seal")
    return seen, provenance


def _fetch_bytes(url: str, *, timeout: int = 16) -> bytes:
    req=Request(url, headers={"User-Agent":"NERAL-Benchmark/1.0 independent evidence",
                              "Accept":"application/json,application/rss+xml,application/atom+xml,application/xml"})
    with urlopen(req, timeout=timeout) as reply:
        return reply.read(1_200_001)[:1_200_000]


def _parse_rss(raw: bytes, name: str) -> list[dict]:
    root=ET.fromstring(raw)
    now=datetime.now(timezone.utc)
    entries=root.findall(".//item")
    if not entries:
        entries=root.findall(".//{http://www.w3.org/2005/Atom}entry")
    found=[]
    for e in entries:
        def field(short):
            n=e.find(short)
            if n is None:
                n=e.find("{http://www.w3.org/2005/Atom}"+short)
            return "".join(n.itertext()).strip() if n is not None else ""
        title=" ".join(field("title").split())
        link=field("link")
        if not link:
            el=e.find("{http://www.w3.org/2005/Atom}link")
            if el is not None:
                link=el.attrib.get("href","")
        date=field("pubDate") or field("published") or field("updated")
        if not (title and link.startswith("https://") and date):
            continue
        try:
            published=parsedate_to_datetime(date) if "," in date else datetime.fromisoformat(date.replace("Z","+00:00"))
            if published.tzinfo is None:
                published=published.replace(tzinfo=timezone.utc)
            if not now-timedelta(days=14) <= published <= now+timedelta(days=1):
                continue
        except (ValueError, TypeError, OverflowError):
            continue
        found.append({"title":title[:350],"url":link,"source":name,
                      "published":published.date().isoformat(),"category":"news"})
    return found


def live_web_facts() -> list[dict]:
    """Get independent gold before asking any NERAL mode; fail closed."""
    items, urls = [], set()
    providers=set()
    for name,url in FEEDS:
        try:
            rows=_parse_rss(_fetch_bytes(url),name)
        except (OSError, ValueError, ET.ParseError):
            continue
        taken=0
        for row in rows:
            if row["url"] in urls:
                continue
            urls.add(row["url"]); items.append(row); taken+=1
            if taken >= 5:
                break
        if taken:
            providers.add(name)
        if len(items)>=20 and len(providers)>=4:
            break
    if len(items)<20 or len(providers)<4:
        raise RuntimeError("Fewer than 20 recent headlines from four independent RSS providers")
    facts=items[:20]
    numeric=[]
    for repo in REPOS:
        try:
            data=json.loads(_fetch_bytes("https://api.github.com/repos/"+repo))
            if not (type(data) is dict and
                    type(data.get("stargazers_count")) is int and
                    type(data.get("open_issues_count")) is int):
                continue
            for metric,label in (("stargazers_count","Sterne"),
                                 ("open_issues_count","offene Issues")):
                numeric.append({"title":repo+" "+label,
                    "url":"https://github.com/"+repo,"source":"GitHub",
                    "published":datetime.now(timezone.utc).isoformat(),
                    "category":"number","value":str(data[metric]),
                    "metric":metric,"repo":repo})
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if len(numeric)>=15:
            break
    if len(numeric)<15 or len({x["repo"] for x in numeric[:15]})<7:
        raise RuntimeError("Fewer than 15 live repository numbers from seven projects")
    return facts+numeric[:15]


def _case(category, index, prompt, *, history=None, grader="blind_pairwise",
          reference=None, web_query=None):
    row={"id":f"{category}-{index:03d}","category":category,"prompt":prompt,
         "history":history or [],"grader":grader}
    if reference is not None:
        row["sealed_reference"]=reference
    if web_query is not None:
        row["web_query"]=web_query
    return row


def _code_family(index:int, rng) -> tuple[str, list[dict], str]:
    suffix=secrets.token_hex(3)
    family=index%5
    if family==0:
        name="sum_even_"+suffix
        desc="Summe aller geraden ganzen Zahlen in einer Liste; leere Liste ergibt 0"
        samples=[{"args":[[2,3,8,-4]],"expected":6},
                 {"args":[[]],"expected":0},{"args":[[1,5,7]],"expected":0}]
    elif family==1:
        name="count_vowels_"+suffix
        desc="Anzahl der Vokale a,e,i,o,u in einem Text ohne Groß-/Kleinschreibung"
        samples=[{"args":["Auto"],"expected":3},{"args":["XYZ"],"expected":0},
                 {"args":["Universum"],"expected":4}]
    elif family==2:
        name="normalize_ws_"+suffix
        desc="mehrfache Leerzeichen und Zeilenumbrüche durch einzelne Leerzeichen ersetzen und außen trimmen"
        samples=[{"args":["  ein  Test\n mit  Lücken "],"expected":"ein Test mit Lücken"},
                 {"args":["  "],"expected":""},
                 {"args":["abc"],"expected":"abc"}]
    elif family==3:
        name="max_or_none_"+suffix
        desc="das größte Element einer Ganzzahlliste zurückgeben, bei leerer Liste None"
        samples=[{"args":[[3,9,-2]],"expected":9},
                 {"args":[[]],"expected":None},
                 {"args":[[-7,-3]],"expected":-3}]
    else:
        name="gcd_positive_"+suffix
        desc="ggT zweier nichtnegativer ganzer Zahlen (ggT(0,0)=0)"
        samples=[{"args":[48,18],"expected":6},{"args":[0,0],"expected":0},
                 {"args":[17,19],"expected":1}]
    return name,samples,desc


def build_suite(out: Path, *, archives: list[Path], web_facts: list[dict]) -> dict:
    """Write-once seal, refuse historical overlap and unscorable web snapshots."""
    if out.exists():
        raise FileExistsError("Private benchmark already sealed; never overwrite")
    prior,provenance=_prior_rows(archives)
    if len(web_facts)!=35 or sum(x.get("category")=="news" for x in web_facts)!=20 or sum(x.get("category")=="number" for x in web_facts)!=15:
        raise ValueError("Need 20 live headlines and 15 live numbers, no placeholders")
    rng=random.SystemRandom()
    cases=[]
    for i,topic in enumerate(TOPICS):
        cases.append(_case("everyday",i,
            f"Erkläre {topic} für {rng.choice(AUDIENCES)} verständlich. "
            f"Verwende ein konkretes neues Beispiel rund um {rng.choice(WORDS)} "
            f"und benenne eine typische Fehlvorstellung."))
    for i in range(40):
        if i<20:
            left=rng.randrange(111,800); right=rng.randrange(111,800)
            old=rng.randrange(20,250)
            hist=[{"role":"user","content":f"Bei Molliair gab es {old} Flüge. Gib eine vorsichtige Einordnung."},
                  {"role":"assistant","content":"Das ist eine ältere Gesprächsnotiz, keine geprüfte Zahl."}]
            prompt=f"Neues Thema (Fall {secrets.token_hex(4).upper()}): Wie viel ist {left} plus {right}? Antworte ausschließlich mit der Ergebniszahl."
            ref={"kind":"integer","value":str(left+right),"topic_switch":True,
                 "forbidden_terms":["Molliair"]}
        elif i<30:
            code=secrets.token_hex(4).upper()
            label=secrets.token_hex(4).upper()
            hist=[{"role":"user","content":f"Für Auftrag B-{label} lautet meine selbst erfundene Kennung K-{code}. Bitte merke sie nur für diesen Dialog."},
                  {"role":"assistant","content":"Verstanden; die Kennung gilt nur in diesem Gespräch."}]
            prompt=f"Und wie lautet die Kennung für Auftrag B-{label}? Antworte nur mit der Kennung."
            ref={"kind":"string","value":"K-"+code,"context_required":True}
        else:
            a=rng.randrange(7,100);b=rng.randrange(7,100);c=rng.randrange(1,70)
            hist=[{"role":"user","content":f"Ein Zug lädt {a} Pakete und später weitere {b}. Was ist die Gesamtsumme?"},
                  {"role":"assistant","content":f"Das wären {a+b} Pakete."}]
            prompt=f"Und wenn danach noch {c} Pakete beim Fall {secrets.token_hex(4).upper()} hinzukommen, wie viele sind es insgesamt? Gib nur die Zahl an."
            ref={"kind":"integer","value":str(a+b+c),"context_required":True}
        cases.append(_case("multiturn",i,prompt,history=hist,grader="independent_exact",reference=ref))
    for i,web in enumerate(web_facts):
        url=web["url"]; source=web["source"]
        if web["category"]=="news":
            prompt=(f"Recherchiere den genauen Titel des Nachrichtenartikels von {source} "
                    f"vom {web['published']} unter dieser URL: {url}. "
                    "Nenne die Überschrift exakt und die belegte Quelle.")
            ref={"kind":"headline","value":web["title"],"url":url,"source":source}
        else:
            what="GitHub-Sterne" if web.get("metric")=="stargazers_count" else "offene GitHub-Issues"
            prompt=(f"Wie viele {what} weist das öffentliche Repository "
                    f"{web['repo']} zum UTC-Datensnapshot {web['published']} aus? "
                    "Recherchiere die offizielle GitHub-Seite und gib die Zahl mit Quelle an.")
            ref={"kind":"web_integer","value":web["value"],"url":url,"source":source}
        cases.append(_case("live_web",i,prompt,grader="source_freshness",reference=ref,
                           web_query=url[:160]))
    seen_math=set()
    for i in range(30):
        for attempt in range(150):
            a=rng.randrange(11,650); b=rng.randrange(7,250);c=rng.randrange(3,77)
            if i<10:
                prompt=f"Berechne exakt ({a} * {b}) + {c}. Gib ausschließlich die Zahl an."
                value=a*b+c
            elif i<20:
                mod=rng.choice([7,11,13,17,19,23])
                prompt=f"Welchen Rest hat {a*b+c} bei Division durch {mod}? Antworte nur mit dem Rest."
                value=(a*b+c)%mod
            else:
                x=rng.randrange(11,70)
                m=rng.choice([7,11,13])
                n=rng.choice([5,9,17])
                prompt=(f"Finde die kleinste positive ganze Zahl unter {m*n} mit "
                        f"Rest {x%m} modulo {m} und Rest {x%n} modulo {n}. "
                        "Antworte nur mit der Zahl.")
                value=next(z for z in range(1,m*n+1) if z%m==x%m and z%n==x%n)
            if norm(prompt) not in seen_math:
                seen_math.add(norm(prompt))
                break
        else:
            raise RuntimeError("Could not generate unique independent math case")
        cases.append(_case("math_logic",i,prompt,grader="independent_exact",
                           reference={"kind":"integer","value":str(value)}))
    for i in range(25):
        name,samples,desc=_code_family(i,rng)
        prompt=(f"Schreibe eine Python-3-Funktion namens {name}(...) mit folgender "
                f"Aufgabe: {desc}. Antworte ausschließlich mit ausführbarem Python-Code, "
                "ohne externe Pakete.")
        cases.append(_case("programming",i,prompt,grader="test_execution",
                           reference={"kind":"python_tests","function":name,
                                      "tests":samples}))
    for i in range(20):
        target=rng.choice(["einen Geschäftspartner","eine Schulgruppe","ein Projektteam",
                           "einen Kundenservice","eine gute Freundin"])
        setting=rng.choice(["Terminverschiebung","Missverständnis","Projektverzug",
                            "Rückfrage zu einer Rechnung","Bitte um eine Erklärung"])
        prompt=(f"Verfasse auf Deutsch eine sachliche, respektvolle Nachricht an {target} "
                f"zum Thema {setting} mit der fiktiven Aktennummer {secrets.token_hex(4).upper()}. Nutze etwa {rng.randrange(60,121)} Wörter, "
                "keine erfundenen persönlichen Angaben und keine Floskeln.")
        cases.append(_case("writing",i,prompt))
    for i in range(15):
        if i<10:
            a=rng.randrange(21,130);b=rng.randrange(5,40)
            wrong=a*b+rng.choice([-4,-3,-2,2,3,4])
            prompt=(f"Behauptung: {a} multipliziert mit {b} ergibt {wrong}. "
                    "Ist das korrekt? Antworte ausschließlich mit Ja oder Nein.")
            cases.append(_case("uncertainty",i,prompt,grader="independent_exact",
                               reference={"kind":"yesno","value":"Nein"}))
        else:
            prompt=(f"Wie viele unregistrierte Objekte der Sorte {rng.choice(WORDS)} "
                    f"mit dem fiktiven Katalogcode {secrets.token_hex(4).upper()} existieren genau in diesem Augenblick? Erkläre die Messgrenze "
                    "ohne vorgetäuschte Echtzeitpräzision.")
            cases.append(_case("uncertainty",i,prompt))
    if Counter(c["category"] for c in cases)!=Counter(CATEGORIES):
        raise AssertionError("Incorrect distributions")
    # Include source and test code as an extra exclusion barrier.
    source=Path(__file__).resolve().parents[1]/"neral_verified_notes.py"
    source_text=norm(source.read_text(encoding="utf-8")) if source.is_file() else ""
    existing=set()
    for case in cases:
        n=norm(case["prompt"])
        if n in prior or n in existing or (n and n in source_text):
            raise ValueError(f"Duplicate with archive or within suite: {case['category']} / {case['id']}")
        existing.add(n)
    out.parent.mkdir(parents=True,exist_ok=True)
    body="".join(json.dumps(c,ensure_ascii=False,sort_keys=True)+"\n" for c in cases)
    # Exclusive creation prevents accidental replacement of already seen tasks.
    with out.open("x",encoding="utf-8") as f:
        f.write(body);f.flush();os.fsync(f.fileno())
    os.chmod(out,0o600)
    manifest=validate_suite(out)
    manifest["generated_utc"]=datetime.now(timezone.utc).isoformat()
    manifest["historical_sources_checked"]=sorted(provenance)
    manifest["web_sources"]=dict(Counter(x["source"] for x in web_facts))
    return manifest


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument("--sealed",type=Path,required=True)
    p.add_argument("--archive",type=Path,action="append",required=True)
    p.add_argument("--manifest-out",type=Path,required=True)
    a=p.parse_args()
    if a.sealed.exists() or a.manifest_out.exists():
        raise FileExistsError("Sealed input/manifest already exists")
    facts=live_web_facts()
    manifest=build_suite(a.sealed,archives=a.archive,web_facts=facts)
    a.manifest_out.parent.mkdir(parents=True,exist_ok=True)
    a.manifest_out.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",
                              encoding="utf-8")
    print("NERAL_BLIND200_SEALED sha256="+manifest["sha256"]+
          " categories="+json.dumps(manifest["categories"],sort_keys=True))


if __name__=="__main__":
    main()
