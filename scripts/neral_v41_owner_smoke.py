"""Owner-only HTTP/CSRF V4.1 compact smoke: 7 daily categories + aliases.

This does not award an intelligence score and never loads private test data.
Run only after the pinned local 9690 instance has been staged safely.
"""
from __future__ import annotations
import http.cookiejar
import json
from urllib.request import HTTPCookieProcessor, ProxyHandler, Request, build_opener

BASE = "http://127.0.0.1:9690"

CASES = [
    ("everyday", "Amber",
     "Erkläre in drei klaren Sätzen, wie eine städtische Wärmeinsel entsteht.", []),
    ("multiturn", "Amber",
     "Wie viel Treibstoff braucht er ungefähr auf dieser Strecke?", [
         {"role":"user","content":"Wie lange dauert ein Flug Berlin nach Abu Dhabi?"},
         {"role":"assistant","content":"Die Flugzeit hängt von Wind und Flugzeug ab."},
         {"role":"user","content":"Und mit einem A321XLR?"},
         {"role":"assistant","content":"Das ist ein Schmalrumpfflugzeug mit großer Reichweite."},
     ]),
    ("live_web", "Quartz",
     "Vergleiche die Desktop-Browser-Engines von Firefox und Chrome anhand gelesener Quellen.", []),
    ("math_logic", "Quartz",
     "How many positive divisors does the integer 180 have?", []),
    ("programming", "Obsidian",
     "Schreibe eine kurze Python-Funktion is_prime(n), mit Typprüfung und zwei Tests.", []),
    ("writing", "Amber",
     "Schreibe eine höfliche kurze Absage für einen Termin morgen.", []),
    ("uncertainty", "Obsidian",
     "Wie viele Menschen wohnen genau in dieser Sekunde in Berlin? Erkläre die Messgrenze.", []),
]
ALIASES = ["luna", "terra", "helios"]
# A greeting that stays on each historical model stop; elaborated requests
# legitimately route through the evidence agent and may escalate tiers.
ALIAS_PROMPT = "Hallo!"


def main() -> None:
    opener = build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()), ProxyHandler({}))
    def get(path):
        with opener.open(BASE + path, timeout=10) as res:
            if res.status != 200:
                raise RuntimeError(f"GET {path} returned {res.status}")
            return json.load(res) if path != "/" else {"status":res.status}
    get("/")
    session=get("/api/neral/preview/session")
    csrf=session.get("csrf")
    status=get("/api/neral/chat/three-tier/status")
    if (type(csrf) is not str or len(csrf)<12 or
        status.get("public_tiers")!=["Amber","Quartz","Obsidian"] or
        not all(x in status.get("tiers",[]) for x in ("Luna","Terra","Helios"))):
        raise RuntimeError("NERAL V4.1 public/legacy tier contract missing")

    def post(tier,prompt,history,web_query=None):
        payload={"tier":tier,"prompt":prompt,"history":history}
        if web_query is not None:
            payload["web_query"]=web_query
        req=Request(BASE+"/api/neral/chat/three-tier",
                    data=json.dumps(payload,ensure_ascii=False).encode("utf-8"),
                    headers={"Content-Type":"application/json",
                             "X-Neral-CSRF":csrf,"Origin":BASE},method="POST")
        with opener.open(req,timeout=120) as response:
            if response.status!=200:
                raise RuntimeError(f"HTTP answer failed for {tier}")
            result=json.load(response)
        if (type(result.get("text")) is not str or not result["text"].strip() or
            result.get("runtime_tier_verified") is not True or
            result.get("verifier_ok") is not False and
                result.get("verifier_ok") is not True):
            raise RuntimeError(f"Malformed answer contract for {tier}")
        return result

    accepted=0
    for category,tier,prompt,history in CASES:
        web="Firefox Chrome desktop rendering engine" if category=="live_web" else None
        result=post(tier,prompt,history,web)
        if category=="live_web" and result.get("web_v4_fast_path") is True:
            if (result.get("model_generated") is not False or
                result.get("adapter_loaded") is not False):
                raise RuntimeError("V4 no-model source response mislabelled")
        if category=="multiturn" and result.get("verified") is True:
            raise RuntimeError("Elliptical follow-up incorrectly promoted to proof")
        if result.get("tier") not in ("Amber","Quartz","Obsidian"):
            raise RuntimeError("Public tier not canonical")
        accepted+=1
        print(f"V41_SMOKE_CATEGORY {category} {tier} HTTP200")
    # A new independent topic in the same conversation must not inherit
    # the earlier Berlin/Abu-Dhabi question or its unverified model replies.
    topic_switch = post("Amber", "Wie viele positive Teiler hat die Zahl 120?", CASES[1][3])
    if topic_switch.get("context_resolved") is True or (topic_switch.get("verification_scope") == "context_assisted_not_independent"):
        raise RuntimeError("Independent topic incorrectly carried previous context")
    print("V41_SMOKE_MULTITURN_TOPIC_SWITCH Amber HTTP200")
    for alias in ALIASES:
        result=post(alias,ALIAS_PROMPT,[])
        if result.get("tier") != alias.capitalize():
            raise RuntimeError(f"Legacy alias {alias} changed contract")
        print(f"V41_SMOKE_ALIAS {alias} HTTP200")
    if accepted!=7:
        raise RuntimeError("Not all 7 categories passed")
    get("/")
    print("NERAL_V41_OWNER_7_CATEGORIES_3_ALIASES_HTTP200")


if __name__=="__main__":
    main()
