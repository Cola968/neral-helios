"""One synthetic public search from the real Windows desktop network.

No user's questions, credentials, personal data or discovered URLs are
persisted. Never treats unverified provider snippets as page content.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time


def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    args=p.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("Search smoke report must be unique")
    start=time.perf_counter()
    report={
        "status":"LIVE_WEB_SEARCH_FAILED",
        "query_kind":"fixed_public_synthetic",
        "query_sent_from_windows_runner":False,
        "result_count":0,
        "provider":None,
        "providers_contacted":[],
        "personal_user_query_sent":False,
        "model_generated":False,
        "model_weights_changed":False,
    }
    try:
        from neral_web_search import WebSearch
        value=WebSearch(max_per_minute=1).search("Berlin Brandenburg Airport")
        results=value.get("results",[])
        report["query_sent_from_windows_runner"]=value.get("external_request") is True
        report["result_count"]=len(results) if isinstance(results,list) else 0
        report["provider"]=value.get("provider")
        report["providers_contacted"]=value.get("providers_contacted",[])
        if report["query_sent_from_windows_runner"] and report["result_count"]>0:
            report["status"]="LIVE_WEB_SEARCH_RESULTS_FOUND"
    except Exception as exc:
        report["error_class"]=type(exc).__name__
    report["elapsed_s"]=round(time.perf_counter()-start,3)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as f:
        json.dump(report,f,sort_keys=True,indent=2)
    print("NERAL_WINDOWS_WEBSEARCH_SCALAR "+json.dumps(report,sort_keys=True),flush=True)
    return 0 if report["status"]=="LIVE_WEB_SEARCH_RESULTS_FOUND" else 1


if __name__=="__main__":
    raise SystemExit(main())
