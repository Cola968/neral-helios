"""Classify CPU pytest failures against exact bf736d5d main baseline."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def junit_report(filename):
    root=ET.parse(filename).getroot()
    suites=list(root.iter("testsuite"))
    tests=sum(int(s.get("tests","0")) for s in suites)
    failures=[]
    errors=0
    for case in root.iter("testcase"):
        failed=case.find("failure") is not None
        error=case.find("error") is not None
        if error: errors+=1
        if failed or error:
            failures.append(case.get("classname","")+"::"+case.get("name",""))
    return {"tests":tests,"failures":sorted(set(failures)),"errors":errors}


def compare_failures(main:dict,candidate:dict)->dict:
    base=set(main.get("failures",[])); current=set(candidate.get("failures",[]))
    new=sorted(current-base)
    return {"baseline_tests":main["tests"],"candidate_tests":candidate["tests"],
        "baseline_failures":len(base),"candidate_failures":len(current),
        "shared_baseline_failures":sorted(current & base),
        "fixed_baseline_failures":sorted(base-current),
        "new_regressions":new,
        "accepted":not new and main["tests"]>=1000 and candidate["tests"]>=main["tests"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--extract",type=Path)
    p.add_argument("--baseline",type=Path)
    p.add_argument("--candidate",type=Path)
    p.add_argument("--out",type=Path)
    args=p.parse_args()
    if args.extract:
        print(json.dumps(junit_report(args.extract),sort_keys=True))
        return
    if not args.baseline or not args.candidate or not args.out:
        p.error("baseline, candidate and out required")
    base=json.loads(args.baseline.read_text())
    cand=json.loads(args.candidate.read_text())
    report=compare_failures(base,cand)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,sort_keys=True))
    if not report["accepted"]:raise SystemExit(2)

if __name__=="__main__":main()
