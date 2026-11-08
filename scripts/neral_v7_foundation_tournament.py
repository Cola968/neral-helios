"""NERAL v7: verifiable, OFFLINE Ollama foundation tournament.

This is a *development* baseline selector, not a neural-weight breakthrough
or an independent release benchmark. It never downloads models or runs source
produced by them. User prompts and all raw model answers stay on the local PC.
"""
from __future__ import annotations

import argparse
from collections import Counter
from hashlib import sha256
import itertools
import json
import math
from pathlib import Path
import random
import re
import urllib.error
import urllib.request

SEED = 20261108
STATUS = "NERAL_V7_FOUNDATION_DEV_ONLY"
URL = "http://127.0.0.1:11434"
SYSTEM = (
    "Solve the user's math, logic or Python-output question accurately. "
    "Your final line MUST be exactly FINAL_ANSWER: <integer> "
    "with the correct signed integer. If thinking is available, "
    "put it in the model's separate thinking channel. "
    "Do not claim to have used external tools."
)
DOMAINS = ("math", "reasoning", "code_trace")
MODEL_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}\Z")
MODEL_DIGEST = re.compile(r"sha256:[a-f0-9]{64}\Z")
_RAW_MODEL_DIGEST = re.compile(r"[a-f0-9]{64}\Z")


def canonical_model_digest(value: str) -> str:
    """Normalize documented Ollama /api/tags bare 64-hex digests.

    Ollama uses bare hashes in /api/tags but sha256-prefixed digests in
    blob endpoints. Internally use a single strict canonical form.
    """
    if not isinstance(value, str):
        raise Refused("Invalid local model digest")
    if _RAW_MODEL_DIGEST.fullmatch(value):
        return "sha256:" + value
    if MODEL_DIGEST.fullmatch(value):
        return value
    raise Refused("Invalid local model digest")

FINAL_LINE = re.compile(r"FINAL_ANSWER: (-?(?:0|[1-9][0-9]{0,12}))\Z")
FAMILIES = {
    "math": ("binomial", "modular_inverse", "digit_weight", "exclusive_multiples"),
    "reasoning": ("implication_count", "dag_path_count", "bus_arrival", "ordering_constraints"),
    "code_trace": ("dict_accumulate", "weighted_slice", "nested_break", "string_filter"),
}


class Refused(ValueError):
    """Unsafe requests, untrusted provenance, or unsuccessful local-only run."""


def _json(value) -> str:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(",",":"),
                      allow_nan=False)


def _task(domain, family, i, prompt, answer):
    if (type(answer) is not int or not -2_000_000 < answer < 2_000_000 or
            not isinstance(prompt,str) or len(prompt)>1800):
        raise Refused("Invalid deterministic oracle")
    locale=("de" if i%2==0 else "en")
    suffix=(" Antworte mit einer ganzen Zahl." if locale=="de"
            else " Return one integer.")
    return {"id":f"{domain}-{family}-{i:02d}", "domain":domain,
            "family":family, "language":locale, "kind":"integer",
            "prompt":prompt+suffix, "answer":answer}


def generate_cases(seed=SEED):
    """12 different task methods × 6 variants; no user-provided source executes."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise Refused("Invalid benchmark seed")
    rng = random.Random(seed)
    cases=[]
    for i in range(6):
        language=i%2==0
        choose=lambda de,en: de if language else en

        # Exact integer mathematics (no floating point answer oracles).
        n=17+2*i; k=3+i%4
        cases.append(_task("math","binomial",i,choose(
            f"Wie viele Möglichkeiten gibt es, {k} Objekte aus {n} verschiedenen "
            "Objekten ohne Reihenfolge auszuwählen?",
            f"How many ways can you choose {k} distinct objects from {n} "
            "without ordering?"),math.comb(n,k)))

        p=(11,13,17,19,23,29)[i]
        a=rng.randint(2,p-1)
        cases.append(_task("math","modular_inverse",i,choose(
            f"Finde die kleinste positive ganze Zahl x mit ({a} * x) modulo {p} = 1.",
            f"Find the smallest positive integer x for which ({a} * x) mod {p} = 1."),
            pow(a,-1,p)))

        number=rng.randint(700,180000); base=rng.randint(3,8)
        digits=[]
        x=number
        while x:
            digits.append(x%base)
            x//=base
        weighted=sum((j+1)*v for j,v in enumerate(digits))
        cases.append(_task("math","digit_weight",i,choose(
            f"Schreibe {number} im Stellenwertsystem zur Basis {base}. "
            "Multipliziere jede Ziffer, von rechts mit Position 1 beginnend, "
            "mit ihrer Position und addiere alles.",
            f"Write {number} in base {base}. Multiply each digit by its position "
            "starting with 1 at the rightmost digit, then sum the products."),weighted))

        start=rng.randint(1,40); end=start+rng.randint(90,210)
        d1=rng.randint(3,8);d2=rng.randint(9,15)
        amount=sum((k%d1==0) != (k%d2==0) for k in range(start,end+1))
        cases.append(_task("math","exclusive_multiples",i,choose(
            f"Wie viele ganze Zahlen von {start} bis {end} einschließlich sind "
            f"durch genau eine der Zahlen {d1} oder {d2} teilbar?",
            f"How many integers from {start} to {end} inclusive are divisible "
            f"by exactly one of {d1} or {d2}?"),amount))

        # Exhaustive Boolean and graph problems with independently computable oracles.
        parity=i%2; threshold=1+i%3
        count=sum((not a or b) and ((c^d)==parity)
                  and (a+b+c+d<=threshold)
                  for a,b,c,d in itertools.product((0,1),repeat=4))
        cases.append(_task("reasoning","implication_count",i,choose(
            "Vier Variablen A,B,C,D sind jeweils 0 oder 1. Wie viele Belegungen "
            f"erfüllen gleichzeitig (A impliziert B), (C XOR D)={parity} "
            f"und A+B+C+D höchstens {threshold}?",
            "Four variables A,B,C,D are each 0 or 1. How many assignments "
            f"satisfy (A implies B), (C XOR D)={parity}, and "
            f"A+B+C+D at most {threshold}?"),count))

        edges=[(0,1),(0,2),(1,3),(2,3),(2,4),(3,5),(4,5)]
        if i%2==0:edges.append((1,4))
        if i%3==0:edges.append((0,3))
        if i%3==1:edges.append((3,4))
        # DAG paths from node 0 to node 5 (every edge strictly forward).
        dp={j:0 for j in range(6)}
        dp[0]=1
        for u in range(6):
            for x,y in edges:
                if x==u:dp[y]+=dp[u]
        cases.append(_task("reasoning","dag_path_count",i,choose(
            f"Ein gerichteter azyklischer Graph hat nur die Kanten {edges}. "
            "Wie viele verschiedene gerichtete Wege führen von Knoten 0 zu Knoten 5?",
            f"A directed acyclic graph has only these edges {edges}. "
            "How many distinct directed paths lead from node 0 to node 5?"),
            dp[5]))

        now=rng.randint(23,64)
        services=[(rng.randint(1,9),rng.randint(7,15),rng.randint(3,14))
                  for _ in range(3)]
        arrival=[]
        for offset,period,travel in services:
            first=offset if now<=offset else offset+((now-offset+period-1)//period)*period
            arrival.append(first+travel)
        cases.append(_task("reasoning","bus_arrival",i,choose(
            f"Es ist Minute {now}. Drei Buslinien haben (erste Abfahrt, Takt in "
            f"Minuten, Fahrtdauer) = {services}. Die erste Abfahrt und alle "
            "darauffolgenden Taktabfahrten sind gültig. Welcher früheste "
            "Ankunftszeitpunkt in Minuten ist ab jetzt erreichbar?",
            f"Current time is minute {now}. Three bus services have "
            f"(first departure, interval, ride duration) = {services}. "
            "First departure and all subsequent interval departures are valid. "
            "What is the earliest reachable absolute arrival minute?"),min(arrival)))

        prohibited=i%4
        left=(i//2)%3
        choices=sum(p[0]!=prohibited and p.index(left)<p.index(3)
                    for p in itertools.permutations(range(4)))
        cases.append(_task("reasoning","ordering_constraints",i,choose(
            "Vier unterschiedliche Aufgaben mit Nummern 0,1,2,3 werden "
            f"in eine Reihenfolge gebracht. Aufgabe {prohibited} darf "
            f"nicht zuerst stehen und Aufgabe {left} muss vor Aufgabe 3 "
            "stehen. Wie viele mögliche Reihenfolgen erfüllen beides?",
            "Four different tasks numbered 0,1,2,3 are ordered. "
            f"Task {prohibited} must not be first and task {left} "
            "must precede task 3. How many orders satisfy both?"),choices))

        # Python-output questions: supplied snippets are DATA. We NEVER
        # execute these snippets. Oracle follows separate trusted operations.
        rows=[(rng.choice(("a","b","c")),rng.randint(1,15)) for _ in range(7)]
        target=rng.choice(("a","b","c"))
        groups={}
        for key,value in rows:
            groups[key]=groups.get(key,0)+value
        code=f"rows = {rows!r}\ntotals = {{}}\nfor key, value in rows:\n    totals[key] = totals.get(key, 0) + value\nprint(totals.get({target!r}, 0))"
        cases.append(_task("code_trace","dict_accumulate",i,choose(
            "Welche ganze Zahl druckt dieses Python-3-Programm?\n",
            "What integer does this Python 3 program print?\n"
        )+"\x60\x60\x60python\n"+code+"\n\x60\x60\x60",
            groups.get(target,0)))

        values=[rng.randint(2,25) for _ in range(8)]
        step=rng.randint(2,3); first=rng.randint(0,1)
        expected=sum((j+1)*v for j,v in enumerate(values[first::step]))
        code=(f"values = {values!r}\npart = values[{first}::{step}]\n"
              "print(sum((i + 1) * value for i, value in enumerate(part)))")
        cases.append(_task("code_trace","weighted_slice",i,choose(
            "Bestimme die ganzzahlige Ausgabe dieses Python-Codes:\n",
            "Determine the integer output of this Python program:\n"
        )+"\x60\x60\x60python\n"+code+"\n\x60\x60\x60",expected))

        outer=rng.randint(5,10); inner=rng.randint(4,9)
        modulus=rng.randint(3,7)
        result=0
        for a in range(1,outer):
            for b in range(1,inner):
                if (a+b)%modulus==0:
                    break
                result+=a
        code=(f"total = 0\nfor a in range(1, {outer}):\n"
              f"    for b in range(1, {inner}):\n"
              f"        if (a + b) % {modulus} == 0:\n"
              "            break\n        total += a\nprint(total)")
        cases.append(_task("code_trace","nested_break",i,choose(
            "Was ist die Ausgabe von Python 3?\n",
            "What does this Python 3 code print?\n"
        )+"\x60\x60\x60python\n"+code+"\n\x60\x60\x60",result))

        word="".join(rng.choice("aeeiobbbccc") for _ in range(rng.randint(12,23)))
        offset=rng.randint(0,2); stride=rng.randint(2,4)
        score=sum(ch in "ae" for ch in word[offset::stride])
        code=(f"text = {word!r}\n"
              f"sample = text[{offset}::{stride}]\n"
              "print(sum(1 for ch in sample if ch in 'ae'))")
        cases.append(_task("code_trace","string_filter",i,choose(
            "Welche Integer-Zahl gibt das folgende Python-Programm aus?\n",
            "What integer does the following Python program print?\n"
        )+"\x60\x60\x60python\n"+code+"\n\x60\x60\x60",score))
    expected={d:24 for d in DOMAINS}
    if (len(cases)!=72 or len({c["id"] for c in cases})!=72 or
            len({c["prompt"] for c in cases})!=72 or
            {d:sum(c["domain"]==d for c in cases) for d in DOMAINS}!=expected or
            {d:{c["family"] for c in cases if c["domain"]==d}
             for d in DOMAINS}!={d:set(FAMILIES[d]) for d in DOMAINS}):
        raise Refused("Generator produced duplicate or missing tasks")
    return cases


def task_digest(cases):
    return sha256(("\n".join(_json(c) for c in cases)+"\n").encode("utf8")).hexdigest()


def evaluate_response(case, raw, *, complete=True):
    """Only a unique terminal FINAL_ANSWER integer is machine gradeable."""
    if not complete:
        return {"status":"incomplete","value":None,"format_valid":False}
    if not isinstance(raw,str) or len(raw)>16384 or not raw.strip():
        return {"status":"ungraded","value":None,"format_valid":False}
    lines=[line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines or len(lines)>256 or sum("FINAL_ANSWER:" in line for line in lines)!=1:
        return {"status":"ungraded","value":None,"format_valid":False}
    match=FINAL_LINE.fullmatch(lines[-1])
    if not match:
        return {"status":"ungraded","value":None,"format_valid":False}
    value=int(match.group(1))
    return {"status":("correct" if value==case["answer"] else "wrong_final"),
            "value":value,"format_valid":True}


class _RejectRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        raise Refused("Ollama localhost redirect rejected")


def _local_opener():
    return urllib.request.build_opener(urllib.request.ProxyHandler({}),_RejectRedirect())


class OllamaLocal:
    """Fixed 127.0.0.1 endpoint; never requests a download or remote API."""
    def __init__(self, timeout=120):
        if type(timeout) not in (int,float) or not 10<=timeout<=240:
            raise Refused("Invalid local per-task timeout")
        self.timeout=timeout

    def _request(self, path, payload=None):
        if path not in ("/api/tags","/api/chat"):
            raise Refused("Unsupported Ollama endpoint")
        data=(None if payload is None else _json(payload).encode("utf8"))
        request=urllib.request.Request(URL+path,data=data,
            headers={"Content-Type":"application/json"},method="GET" if data is None else "POST")
        try:
            with _local_opener().open(request,timeout=self.timeout) as response:
                raw=response.read(262145)
            if len(raw)>262144:
                raise Refused("Excessive local response body")
            reply=json.loads(raw)
        except (OSError, ValueError, urllib.error.URLError, UnicodeError) as exc:
            raise Refused("Local Ollama unavailable or response invalid") from exc
        if not isinstance(reply,dict):
            raise Refused("Unexpected Ollama response")
        return reply

    def inventory(self):
        payload=self._request("/api/tags")
        models=payload.get("models")
        if not isinstance(models,list) or len(models)>150:
            raise Refused("Untrusted local model inventory")
        installed={}
        for row in models:
            if not isinstance(row,dict):
                raise Refused("Invalid local model item")
            name,digest=row.get("name"),row.get("digest")
            if not isinstance(name,str) or not MODEL_TAG.fullmatch(name):
                raise Refused("Invalid model name in local inventory")
            digest=canonical_model_digest(digest)
            if name in installed:
                raise Refused("Duplicate local model name")
            installed[name]=digest
        return installed

    def chat(self, model, prompt, *, think, max_tokens):
        payload={
            "model":model,
            "messages":[{"role":"system","content":SYSTEM},
                        {"role":"user","content":prompt}],
            "think":think,
            "stream":False,
            "keep_alive":"2m",
            "options":{"temperature":0,"num_predict":max_tokens,
                       "num_ctx":4096,"seed":SEED},
        }
        result=self._request("/api/chat",payload)
        message=result.get("message")
        content=message.get("content") if isinstance(message,dict) else None
        reason=result.get("done_reason")
        eval_count=result.get("eval_count",0)
        duration_ns=result.get("total_duration",0)
        if (not isinstance(content,str) or len(content)>16384 or
                reason not in ("stop","length") or
                type(eval_count) is not int or not 0<=eval_count<100000 or
                type(duration_ns) is not int or not 0<=duration_ns<10**15):
            raise Refused("Malformed local response / no final completion")
        return {"content":content,"reason":reason,
                "eval_count":eval_count,"duration_ns":duration_ns}


def _validate_models(names, installed):
    if (not isinstance(names,list) or not 2<=len(names)<=3 or
            len(set(names))!=len(names) or not isinstance(installed,dict)):
        raise Refused("Choose 2 or 3 distinct installed Qwen foundation tags")
    result={}
    for name in names:
        if (not isinstance(name,str) or not MODEL_TAG.fullmatch(name) or
                not name.startswith(("qwen2.5-coder:","qwen3:","qwen3.5:")) or
                name not in installed or not isinstance(installed[name],str) or
                not MODEL_DIGEST.fullmatch(installed[name])):
            raise Refused("Foundation tag not installed or not approved")
        result[name]=installed[name]
    if len(set(result.values()))!=len(result):
        raise Refused("Same model digest masquerades as different foundations")
    return result


def _create_private_folder(output):
    folder=Path(output)
    if (folder.is_symlink() or folder.exists() or not folder.parent.is_dir() or
            folder.parent.is_symlink() or
            folder.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise Refused("Create-only local private folder outside repository required")
    folder.mkdir(exist_ok=False)
    return folder


def compare_local_models(client, models, output, *, think=False, limit_cases=None):
    """Same source, prompts, decoder and fixed inference budget for each model."""
    if type(think) is not bool:
        raise Refused("Separate explicit thinking and fast conditions")
    if limit_cases is not None and (type(limit_cases) is not int or
                                   not 1<=limit_cases<=72):
        raise Refused("Invalid smoke-test limit")
    installed=_validate_models(models,client.inventory())
    cases=generate_cases()
    if limit_cases is not None:
        cases=cases[:limit_cases]
    folder=_create_private_folder(output)
    meta={"schema":1,"status":STATUS,"models":installed,
          "suite_sha256":task_digest(cases),
          "task_count":len(cases),"seed":SEED,"think":think,
          "num_predict":2048 if think else 512,
          "num_ctx":4096,"temperature":0,
          "model_trained":False,"release_validated":False}
    (folder/"private-manifest.json").write_text(_json(meta)+"\n",encoding="utf8")
    tallies={}
    with (folder/"private-responses.jsonl").open("x",encoding="utf8") as out:
        for name in models:
            statuses=Counter()
            domains={domain:Counter() for domain in DOMAINS}
            families={family:Counter() for groups in FAMILIES.values() for family in groups}
            total_generated=0
            total_duration=0
            for case in cases:
                raw=client.chat(name,case["prompt"],think=think,
                                max_tokens=meta["num_predict"])
                if (not isinstance(raw,dict) or not isinstance(raw.get("content"),str)
                        or raw.get("reason") not in ("stop","length") or
                        type(raw.get("eval_count")) is not int or
                        type(raw.get("duration_ns")) is not int):
                    raise Refused("Missing source response telemetry")
                grade=evaluate_response(case,raw["content"],
                                        complete=raw["reason"]=="stop")
                statuses[grade["status"]]+=1
                domains[case["domain"]][grade["status"]]+=1
                families[case["family"]][grade["status"]]+=1
                total_generated+=raw["eval_count"]
                total_duration+=raw["duration_ns"]
                out.write(_json({"id":case["id"],"domain":case["domain"],
                    "family":case["family"],"prompt_private":case["prompt"],
                    "answer_private":case["answer"],"model":name,
                    "raw_private":raw["content"],"grade":grade,
                    "done_reason":raw["reason"]})+"\n")
                out.flush()
            tallies[name]={k:statuses[k] for k in
                           ("correct","wrong_final","ungraded","incomplete")}
            tallies[name].update({
                "by_domain":{d:dict(domains[d]) for d in DOMAINS},
                "by_family":{f:dict(families[f]) for groups in FAMILIES.values()
                             for f in groups},
                "total_tokens_generated":total_generated,
                "total_duration_seconds":round(total_duration/1e9,3),
            })
    # Verify that models cannot silently change between inference and scoring.
    recheck=client.inventory()
    for name,digest in installed.items():
        if recheck.get(name)!=digest:
            raise Refused("Local model digest drift during evaluation")
    result={
        "schema":1,"status":STATUS,"case_sample":(
            "pilot_not_full_suite" if limit_cases is not None
            else "frozen_public_development_suite"),
        "suite_sha256":meta["suite_sha256"],"seed":SEED,
        "count_per_model":len(cases),
        "models":tallies,
        "foundation_digests":installed,
        "think":think,"num_predict":meta["num_predict"],
        "num_ctx":4096,"temperature":0,
        "model_trained":False,"adapter_promoted":False,
        "release_validated":False,"model_superiority_proven":False,
        "independent_benchmark_proven":False,
        "raw_responses_uploaded":False,"no_model_downloads":True,
        "limitations":("A small public, researcher-authored development suite. "
                       "Score is strict terminal integer accuracy only. "
                       "Python output interpretation is not code generation. "
                       "Different foundation weights/quantizations and Ollama "
                       "model runtime can differ; memory/speed vary. "
                       "No model or adapter intelligence breakthrough proven."),
    }
    (folder/"scalar-summary.json").write_text(_json(result)+"\n",encoding="utf8")
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--mode",required=True,choices=("inventory","compare"))
    p.add_argument("--models",default="",help="Comma-separated names of EXACT preinstalled local models")
    p.add_argument("--out",type=Path)
    p.add_argument("--thinking",choices=("on","off"),default="off")
    p.add_argument("--limit-cases",type=int,default=None,
                   help="Diagnostic smoke test only, not a full benchmark")
    args=p.parse_args(argv)
    client=OllamaLocal()
    if args.mode=="inventory":
        if args.out or args.models or args.limit_cases is not None:
            p.error("Inventory cannot write output or select models")
        installed=client.inventory()
        print(_json({"status":"LOCAL_INVENTORY_READONLY","models":installed,
                     "models_downloaded":False,"model_trained":False}))
        return
    if args.out is None or not args.models:
        p.error("Explicit installed tags and private new output folder required")
    models=args.models.split(",")
    report=compare_local_models(client,models,args.out,
                                think=args.thinking=="on",
                                limit_cases=args.limit_cases)
    print(_json({"status":report["status"],"case_sample":report["case_sample"],
                 "count_per_model":report["count_per_model"],
                 "scores":{n:{k:v[k] for k in
                             ("correct","wrong_final","ungraded","incomplete")}
                           for n,v in report["models"].items()},
                 "model_superiority_proven":False,"release_validated":False}))


if __name__=="__main__":
    main()
