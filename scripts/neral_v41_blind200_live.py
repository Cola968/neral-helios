"""Owner-only serialized HTTP blind benchmark: 200 sealed tasks x three tiers.

No raw prompts, answer keys or model replies are written to GitHub artifacts.
Code grading only happens inside an offline, unprivileged Docker container.
"""
from __future__ import annotations
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import http.cookiejar
import ast
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, HTTPCookieProcessor, ProxyHandler, build_opener

from scripts.neral_v41_blind200 import TIERS, CATEGORIES, validate_suite

BASE="http://127.0.0.1:9690"
IMAGE="python:3.11-alpine"


def _norm(text):
    return " ".join(re.findall(r"\w+",str(text).casefold()))


def _exact_text(text,kind):
    from neral_v42_quality import final_marked_value
    val=str(text).strip()
    if "ENDERGEBNIS" in val.upper():
        isolated = final_marked_value(val)
        if isolated is None:
            return None
        val = isolated
    else:
        # Only unwrap an unmarked entire code fence. For a marked answer,
        # stripping the final backtick would corrupt ENDERGEBNIS: `345`.
        val = val.strip("`")
    if kind in ("integer","web_integer"):
        # Do not award correctness just because the right number is mentioned
        # somewhere among several wrong alternatives or worked-out steps.
        m=re.fullmatch(r"(?:Ergebnis|Antwort|Resultat)?\s*:?\s*([-+]?\d[\d., ]*)\s*(?:[.!])?",val,re.I)
        if m:
            v=re.sub(r"[. ,]","",m.group(1))
            return v
        # For web numeric answers only, permit a single well-delimited
        # number next to a star/issue unit, not multiple competing numbers.
        if kind=="web_integer":
            nums=re.findall(r"(?<!\w)\d[\d.,]*(?!\w)",val)
            if len(nums)==1:
                return re.sub(r"[.,]","",nums[0])
        return None
    if kind=="yesno":
        m=re.fullmatch(r"\s*(Ja|Nein)\s*[.!]?\s*",val,re.I)
        return m.group(1).casefold() if m else None
    return val


def _source_matches(ref,result):
    wanted=urlsplit(ref["url"])
    for s in result.get("sources",[]) if type(result.get("sources")) is list else []:
        if type(s) is not dict or s.get("page_read") is not True:
            continue
        candidate=s.get("url")
        if type(candidate) is not str:
            continue
        found=urlsplit(candidate)
        if found.scheme!="https" or not found.hostname or not wanted.hostname:
            continue
        # The page actually read must be the independent gold article/repo,
        # not just any page from the same publisher.
        if found.hostname.casefold()==wanted.hostname.casefold() and (
                found.path.rstrip("/")==wanted.path.rstrip("/")):
            return True
        if (wanted.hostname=="github.com" and found.hostname=="api.github.com"
                and found.path.rstrip("/")==
                    "/repos"+wanted.path.rstrip("/")):
            return True
    return False


def _code_only(text):
    code=str(text).strip()
    m=re.search(r"`{3}(?:python|py)?\s*\n(.*?)\n`{3}",code,re.I|re.S)
    code=m.group(1).strip() if m else code
    return code if len(code.encode("utf-8"))<=12000 else ""


# Runs only inside Docker; the remote model's Python is never executed on Windows.
_CONTAINER_GRADER = r'''
import json,sys,traceback,signal
signal.alarm(8)
data=json.load(sys.stdin)
scope={}
try:
    source=data["code"]
    if len(source.encode("utf-8"))>12000: raise ValueError("oversized")
    exec(compile(source,"candidate.py","exec"),scope,scope)
    name=data["function"]
    fn=scope.get(name)
    if not callable(fn): raise AssertionError("required function missing")
    for test in data["tests"]:
        actual=fn(*test["args"])
        if actual != test["expected"]:
            raise AssertionError("wrong result")
    print("NERAL_PRIVATE_TESTS_PASS")
except BaseException:
    print("NERAL_PRIVATE_TESTS_FAIL")
    sys.exit(1)
'''


def grade_python(case,response_text,*,docker_image=IMAGE):
    ref=case["sealed_reference"]
    code=_code_only(response_text)
    if not code or not re.search(r"\bdef\s+"+re.escape(ref["function"])+r"\s*\(",code):
        return "fail","no_required_function"
    # Reject invalid Python without launching Docker. No untrusted code is run.
    try:
        ast.parse(code)
    except SyntaxError:
        return "fail", "python_syntax_invalid"
    payload=json.dumps({"code":code,"function":ref["function"],"tests":ref["tests"]})
    cmd=["docker","run","--rm","-i","--pull","never",
         "--network","none","--read-only","--memory","128m",
         "--memory-swap","128m","--cpus","1","--pids-limit","64",
         "--cap-drop","ALL","--security-opt","no-new-privileges",
         "--user","65534:65534","--tmpfs","/tmp:rw,noexec,nosuid,size=16m",
         docker_image,"python","-B","-c",_CONTAINER_GRADER]
    try:
        run=subprocess.run(cmd,input=payload,text=True,capture_output=True,
                           timeout=20,check=False)
    except (OSError,subprocess.TimeoutExpired):
        return "ungraded","docker_unavailable_or_timeout"
    if run.returncode in (125,126,127):
        return "ungraded","docker_not_available"
    if run.returncode==0 and "NERAL_PRIVATE_TESTS_PASS" in run.stdout:
        return "pass","private_docker_tests_pass"
    return "fail","private_docker_tests_fail"


def grade_response(case: dict, response: dict, *, docker_image=IMAGE):
    """Return (pass/fail/ungraded, reason), never infer quality from HTTP200."""
    if type(response) is not dict or not str(response.get("text","")).strip():
        return "fail","empty_response"
    if response.get("error_code") is not None:
        return "fail","runtime_error"
    rubric=case["grader"]
    if rubric=="blind_pairwise":
        return "ungraded","needs_independent_human_review"
    ref=case["sealed_reference"]
    if rubric=="test_execution":
        return grade_python(case,response["text"],docker_image=docker_image)
    kind=ref["kind"]
    if rubric=="source_freshness":
        if not _source_matches(ref,response):
            return "fail","no_exact_fresh_independent_source"
        if kind=="headline":
            answer=_norm(response["text"])
            wanted=_norm(ref["value"])
            return ("pass","independent_rss_headline") if wanted in answer else (
                "fail","headline_mismatch")
        if kind=="web_integer":
            value=_exact_text(response["text"],kind)
            return ("pass","independent_api_snapshot") if value==ref["value"] else (
                "fail","web_number_mismatch")
        return "ungraded","unsupported_web_reference"
    if rubric=="independent_exact":
        if ref.get("topic_switch") is True:
            if response.get("context_resolved") is True or (
                    response.get("verification_scope")=="context_assisted_not_independent"):
                return "fail","topic_contamination"
            if any(t.casefold() in response["text"].casefold() for t in ref.get("forbidden_terms",[])):
                return "fail","prior_topic_leak"
        if kind=="string":
            val=_exact_text(response["text"],kind)
            if val is None:
                return "fail","exact_output_unparseable"
            val=val.strip().rstrip(".! ")
            return ("pass","exact_string") if val.casefold()==ref["value"].casefold() else (
                "fail","string_mismatch")
        result=_exact_text(response["text"],kind)
        if result is None and kind in ("integer","web_integer","yesno"):
            # Format failure is not proof of wrong arithmetic. Keep score FAIL,
            # but distinguish a parser contract mismatch from wrong content.
            return "fail","exact_output_unparseable"
        expected=ref["value"].casefold() if kind=="yesno" else ref["value"]
        return ("pass","independent_exact") if result==expected else (
            "fail","exact_mismatch")
    return "ungraded","unsupported_grader"


class OwnerClient:
    def __init__(self):
        # Fresh cookies and CSRF per task/chain; no global browser history.
        self.opener=build_opener(HTTPCookieProcessor(http.cookiejar.CookieJar()),
                                 ProxyHandler({}))
        self.csrf=None

    def get(self,path):
        with self.opener.open(BASE+path,timeout=12) as r:
            if r.status!=200:
                raise RuntimeError("NERAL health endpoint returned non-200")
            return json.load(r) if path!="/" else {"status":r.status}

    def connect(self):
        self.get("/")
        session=self.get("/api/neral/preview/session")
        self.csrf=session["csrf"]
        if type(self.csrf) is not str or len(self.csrf)<12:
            raise RuntimeError("Invalid CSRF session")

    def post(self,tier,prompt,history,*,web_query=None,diagnostic_telemetry=False):
        payload={"tier":tier,"prompt":prompt,"history":history,
                 "benchmark_isolated":True}
        if diagnostic_telemetry:
            payload["diagnostic_telemetry"]=True
        if web_query is not None:
            payload["web_query"]=web_query
        request=Request(BASE+"/api/neral/chat/three-tier",
                data=json.dumps(payload,ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type":"application/json","X-Neral-CSRF":self.csrf,
                         "Origin":BASE},method="POST")
        start=time.monotonic()
        try:
            with self.opener.open(request,timeout=180) as response:
                code=response.status
                result=json.load(response)
        except HTTPError as e:
            code=e.code
            try: result=json.loads(e.read(4096))
            except ValueError: result={"error":"http_error"}
        except (OSError,ValueError,URLError) as e:
            code=0
            result={"error_code":"connection_or_timeout","error":"transport_failure"}
        elapsed=time.monotonic()-start
        if code==200 and result.get("benchmark_isolated") is not True:
            raise RuntimeError("API did not confirm isolated, no-training benchmark mode")
        return code,result,elapsed


def _safe_history_piece(text):
    text=" ".join(str(text).split())
    while len(text.encode("utf-8"))>1800:
        text=text[:-1]
    return text if text else "[Keine vollständige Modellantwort]"


def execute_case(client,tier,case):
    client.connect()
    history=[]
    elapsed=0.0
    for turn in case["history"][::2]:
        # True model-generated multi-turn chains: earlier assistant turns
        # come from the SAME model and session, never the fixture's stub.
        code,result,spent=client.post(tier,turn["content"],history)
        elapsed+=spent
        if code!=200 or not str(result.get("text","")).strip():
            return code,result,elapsed,"chain_setup_failed"
        history.extend([
            {"role":"user","content":turn["content"]},
            {"role":"assistant","content":_safe_history_piece(result["text"])}
        ])
    from neral_v42_quality import prompt_for_exact_case
    code,result,spent=client.post(tier,prompt_for_exact_case(case),history,
               web_query=case.get("web_query"))
    return code,result,elapsed+spent,None


def _summary(rows,manifest,*,expected):
    stats=defaultdict(list)
    for record in rows:
        stats[(record["tier"],record["category"])].append(record)
    categories=[]
    for tier in TIERS:
        for category in CATEGORIES:
            rs=stats[(tier,category)]
            rates=[x["latency_s"] for x in rs]
            graded=[x for x in rs if x["grade"] in ("pass","fail")]
            categories.append({
                "tier":tier,"category":category,"n":len(rs),
                "graded":len(graded),"passed":sum(x["grade"]=="pass" for x in rs),
                "failed":sum(x["grade"]=="fail" for x in rs),
                "ungraded":sum(x["grade"]=="ungraded" for x in rs),
                "pass_rate_graded_pct":(round(100*sum(x["grade"]=="pass" for x in rs)/len(graded),2)
                                        if graded else None),
                "mean_latency_s":round(statistics.mean(rates),3) if rates else None,
                "max_latency_s":round(max(rates),3) if rates else None,
                "model_generated_count":sum(x.get("model_generated") is True for x in rs),
                "model_origins":dict(sorted(__import__("collections").Counter(
                    str(x.get("model_origin") or "unknown") for x in rs).items())),
                "tool_routes":dict(sorted(__import__("collections").Counter(
                    str(x.get("tool_used") or "none") for x in rs).items())),
                "errors":dict(sorted(__import__("collections").Counter(
                    x["reason"] for x in rs if x["grade"]!="pass").items())),
            })
    return {
        "status":"COMPLETE" if len(rows)==expected else "PARTIAL",
        "sealed_sha256":manifest["sha256"],"cases_per_tier":expected//3,
        "observations":len(rows),"expected":expected,
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "measurement_note":"Ungraded writing/explanation needs human review; HTTP200 is not correctness",
        "no_training_capture":True,
        "categories":categories
    }


def run(sealed:Path,manifest_path:Path,private_details:Path,summary_path:Path,
        *,limit_cases:int=200):
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    actual=validate_suite(sealed)
    if manifest["sha256"]!=actual["sha256"] or limit_cases<1 or limit_cases>200:
        raise ValueError("Sealed SHA256 mismatch or invalid case limit")
    cases=[json.loads(x) for x in sealed.read_text(encoding="utf-8").splitlines() if x.strip()]
    # Fail BEFORE sending ANY prompt if the code-grade Docker image is missing.
    from neral_docker_runtime import ensure_docker_ready
    if not ensure_docker_ready():
        raise RuntimeError("Local offline Docker daemon unavailable")
    check=subprocess.run(["docker","image","inspect",IMAGE],
                         capture_output=True,timeout=15,check=False)
    if check.returncode!=0:
        raise RuntimeError("Preinstalled offline code-grading Docker image unavailable")
    OwnerClient().get("/")
    # Validate opt-out once, before reading any private case or mutating data.
    preflight=OwnerClient()
    preflight.connect()
    probe_status,probe_result,_=preflight.post("Amber","Hallo!",[])
    if probe_status!=200 or probe_result.get("benchmark_isolated") is not True:
        raise RuntimeError("Three-tier service missing benchmark isolation; refuse live run")
    expected=3*min(limit_cases,len(cases))
    # Tier-major scheduling avoids unnecessary model unload/reload on 12GB VRAM.
    plan=[(tier,case) for tier in TIERS for case in cases[:limit_cases]]
    if private_details.exists():
        raise FileExistsError("Results already exist; never silently overwrite a blind run")
    private_details.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    with private_details.open("x",encoding="utf-8") as output:
        for index,(tier,case) in enumerate(plan,1):
            client=OwnerClient()
            try:
                code,response,elapsed,setup_failure=execute_case(client,tier,case)
                if code!=200:
                    grade,reason="fail",("http_"+str(code))
                elif setup_failure:
                    grade,reason="fail",setup_failure
                else:
                    grade,reason=grade_response(case,response)
            except (OSError,ValueError,RuntimeError) as e:
                # Missing benchmark isolation is fatal, not a scorable failure.
                if "isolated, no-training" in str(e):
                    raise
                code,response,elapsed,grade,reason=0,{},0.0,"fail","benchmark_exception"
            record={
                "id":case["id"],"tier":tier,"category":case["category"],
                "prompt_sha256":sha256(case["prompt"].encode("utf-8")).hexdigest(),
                "http_status":code,"latency_s":round(elapsed,4),
                "grade":grade,"reason":reason,
                "model_text":str(response.get("text",""))[:16000],
                "model_origin":response.get("model_origin"),
                "model":response.get("model"),
                "effective_tier":response.get("tier"),
                "model_generated":response.get("model_generated"),
                "tool_used":response.get("tool_used"),
                "verification_scope":response.get("verification_scope"),
                "error_code":response.get("error_code"),
                "adapter_loaded":response.get("adapter_loaded"),
            }
            output.write(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n")
            output.flush();os.fsync(output.fileno())
            rows.append(record)
            if index%25==0 or index==expected:
                print(f"NERAL_BLIND200_PROGRESS {index}/{expected} "+
                      f"last={tier}:{case['category']}:{grade}",flush=True)
    after=OwnerClient().get("/")
    if after["status"]!=200:
        raise RuntimeError("Port 9690 failed post-benchmark HTTP200")
    summary=_summary(rows,manifest,expected=expected)
    summary["private_details_sha256"]=sha256(private_details.read_bytes()).hexdigest()
    summary_path.write_text(json.dumps(summary,ensure_ascii=False,sort_keys=True,indent=2)+"\n",
                            encoding="utf-8")
    print("NERAL_BLIND200_COMPLETE observations="+str(len(rows))+
          " sealed_sha256="+manifest["sha256"]+
          " HTTP200_PORT9690",flush=True)
    return summary


def main():
    import argparse
    p=argparse.ArgumentParser()
    for key in ("sealed","manifest","private-details","summary"):
        p.add_argument("--"+key,type=Path,required=True)
    p.add_argument("--limit-cases",type=int,default=200)
    a=p.parse_args()
    run(a.sealed,a.manifest,a.private_details,a.summary,limit_cases=a.limit_cases)


if __name__=="__main__":
    main()
