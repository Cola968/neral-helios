"""Run NERAL V3.1's frozen 100-case quality gate against LIVE localhost V3.

One real Windows RTX4070, no model downloads, no service restart, no training,
no raw chat transcript persisted. Uses synthetic prompts only. Creates and
deletes one temporary local account; aborts if source, service or cleanup fails.
"""
from __future__ import annotations

import argparse
import http.cookiejar
import json
from pathlib import Path
import secrets
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, ProxyHandler, Request, build_opener

from scripts.neral_v31_quality_gate import (
    build_cases, manifest_sha256, score_case, summarize, validate_case_manifest,
)

ORIGIN = "http://127.0.0.1:9690"
AUTHORIZATION = "NERAL_V31_OWNER_APPROVED_READONLY100"
TIERS = {
    "math": "Terra",
    "logic": "Helios",
    "programming": "Helios",
    "web": "Luna",       # V3 may auto-escalate a sourced web comparison to Helios
    "knowledge": "Luna",
}

class LiveLocalAPI:
    def __init__(self):
        jar=http.cookiejar.CookieJar()
        self.opener=build_opener(ProxyHandler({}),HTTPCookieProcessor(jar))
        self.csrf=None

    def call(self,path,method="GET",payload=None,timeout=180):
        if not path.startswith("/") or path.startswith("//") or "://" in path:
            raise ValueError("Only the fixed localhost service is permitted")
        data=(json.dumps(payload,ensure_ascii=False).encode("utf-8")
              if payload is not None else None)
        headers={"Accept":"application/json"}
        if data is not None:
            headers["Content-Type"]="application/json"
        if method != "GET" and self.csrf:
            headers["X-Neral-CSRF"]=self.csrf
        request=Request(ORIGIN+path,method=method,data=data,headers=headers)
        try:
            with self.opener.open(request,timeout=timeout) as res:
                raw=res.read(131072)
                return res.status, raw
        except HTTPError as err:
            return err.code,err.read(131072)

    def json(self,path,method="GET",payload=None,timeout=180):
        status,raw=self.call(path,method,payload,timeout)
        try:
            parsed=json.loads(raw)
        except (ValueError,UnicodeDecodeError):
            parsed={}
        return status,parsed if isinstance(parsed,dict) else {}

def require_gpu():
    response=subprocess.run(
        ["nvidia-smi","--query-gpu=name,memory.total",
         "--format=csv,noheader,nounits"],
        capture_output=True,text=True,timeout=12,check=True,
    )
    rows=response.stdout.strip().splitlines()
    if len(rows)!=1 or "RTX 4070" not in rows[0]:
        raise RuntimeError("Expected local RTX4070; no benchmark run")
    return "RTX 4070"

def run(expected_sha,web_delay=40.0,case_limit=100):
    if not 0 <= web_delay <= 120:
        raise ValueError("Web delay outside bounded rate-limit interval")
    if case_limit != 100:
        raise ValueError("No partial 100-case benchmark may be reported as complete")
    cards=build_cases()
    validate_case_manifest(cards)
    frozen_hash=manifest_sha256(cards)
    if not expected_sha or len(expected_sha)!=40 or \
       any(c not in "0123456789abcdef" for c in expected_sha):
        raise ValueError("Full source SHA required")
    current=subprocess.run(["git","rev-parse","HEAD"],text=True,
                           capture_output=True,timeout=5,check=True).stdout.strip()
    if current!=expected_sha:
        raise RuntimeError("Source commit does not match owner-authorized benchmark")
    gpu=require_gpu()
    client=LiveLocalAPI()
    status,_=client.call("/")
    if status!=200:
        raise RuntimeError("Local V3 homepage did not answer HTTP200")
    status,info=client.json("/api/neral/preview/session")
    if (status!=200 or info.get("evidence_agent_enabled") is not True or
            info.get("three_tier_enabled") is not True or
            not isinstance(info.get("csrf"),str)):
        raise RuntimeError("Original local V3 evidence-enabled service missing")
    client.csrf=info["csrf"]
    status,detail=client.json("/api/neral/chat/three-tier/status")
    if status!=200 or detail.get("enabled") is not True:
        raise RuntimeError("Actual V3 three-tier runtime not available")

    username="v31bench"+secrets.token_hex(6)
    password="V31-Local-"+secrets.token_urlsafe(26)
    registered=False
    cleaned=False
    rows=[]
    fatal=None
    started=time.monotonic()
    try:
        status,auth=client.json("/api/neral/account/register","POST",
                                {"username":username,"password":password},timeout=30)
        if status!=201 or auth.get("authenticated") is not True or \
           not isinstance(auth.get("csrf"),str):
            raise RuntimeError("Synthetic evaluation account registration failed")
        registered=True
        client.csrf=auth["csrf"]
        print("NERAL_V31_CASES_SEALED count=100 sha256="+frozen_hash,flush=True)
        last_web_request=None
        for i,case in enumerate(cards):
            if case["category"]=="web" and last_web_request is not None:
                # V3 uses 2+ provider searches per comparison; conservative
                # spacing avoids burning the provider's 6/min opt-in budget.
                spacing=web_delay-(time.monotonic()-last_web_request)
                if spacing>0:
                    time.sleep(spacing)
            tier=TIERS[case["category"]]
            data={"tier":tier,"prompt":case["question"],"history":[],
                  "force_sandbox":False}
            if case["category"]=="web":
                data["web_query"]=case["web_query"]
                last_web_request=time.monotonic()
            started_case=time.monotonic()
            try:
                api_status,answer=client.json(
                    "/api/neral/chat/three-tier","POST",data,timeout=190)
                if api_status==409:
                    # Concurrent local user activity is not a model failure.
                    # Reattempt the same synthetic case with a bounded wait.
                    time.sleep(4)
                    api_status,answer=client.json(
                        "/api/neral/chat/three-tier","POST",data,timeout=190)
            except (OSError,URLError,TimeoutError):
                api_status,answer=599,{}
            elapsed=round(time.monotonic()-started_case,4)
            answer={**answer,"seconds":elapsed,"_requested_tier":tier}
            measured=score_case(case,answer,api_status)
            rows.append(measured)
            print("NERAL_V31_CASE "+json.dumps({
                "i":i+1,"category":case["category"],"passed":measured["passed"],
                "critical":measured["critical_unverified"],
                "seconds":elapsed,"http":api_status,
            },sort_keys=True),flush=True)
    except (RuntimeError,ValueError,OSError) as error:
        fatal=type(error).__name__
    finally:
        if registered:
            try:
                status,deleted=client.json("/api/neral/account","DELETE",
                                           {"password":password},timeout=40)
                cleaned=status==200 and deleted.get("deleted") is True
            except (OSError,URLError,TimeoutError):
                cleaned=False
    if fatal is not None:
        raise RuntimeError("Benchmark aborted: "+fatal)
    if not cleaned:
        raise RuntimeError("Synthetic account cleanup unconfirmed")
    report=summarize(cards,rows)
    report.update({
        "source_sha":expected_sha,
        "real_local_gpu":gpu,
        "live_service":True,
        "account_deleted":cleaned,
        "source_manifest_frozen_before_first_answer":True,
        "raw_model_responses_saved":False,
        "user_chats_read":False,
        "duration_seconds":round(time.monotonic()-started,2),
    })
    return {"summary":report,"rows":rows}

def main(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument("--authorize",required=True)
    parser.add_argument("--source-sha",required=True)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--web-delay",default=40,type=float)
    args=parser.parse_args(argv)
    if args.authorize!=AUTHORIZATION:
        raise SystemExit("No explicit V3.1 local benchmark authorization")
    try:
        result=run(args.source_sha,web_delay=args.web_delay)
        args.out.parent.mkdir(parents=True,exist_ok=True)
        with args.out.open("x",encoding="utf-8") as output:
            json.dump(result,output,ensure_ascii=False,sort_keys=True,indent=2)
        stats=result["summary"]
        print("NERAL_V31_100_SCALAR "+json.dumps({
            "status":stats["status"],
            "score":stats["combined_passed"],
            "objective":stats["objective_oracle"],
            "web":stats["web_citation_integrity"],
            "critical_unverified":stats["critical_unverified"],
            "threshold_met":stats["threshold_90_of_100_met"],
            "latency_seconds":stats["latency_seconds"],
            "latency_by_executed_tier":stats["latency_by_executed_tier"],
            "account_deleted":stats["account_deleted"],
            "source_sha":stats["source_sha"],
        },sort_keys=True),flush=True)
        # This is exploratory: score below 90 is a finding, not a CI failure.
        return 0
    except Exception as err:
        # No secrets, prompts, network response bodies or model completions in logs.
        print("NERAL_V31_100_BLOCKED "+type(err).__name__,flush=True)
        return 1

if __name__=="__main__":
    sys.exit(main())
