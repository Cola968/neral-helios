"""Conservative, read-only independent reference audit of NeralTalk350.

Arithmetic is recomputed without eval. Code is checked by Python AST only,
never executed. Unrecognized prompts/answers remain unverified, not pass.
Only aggregate counters are returned (no individual IDs, texts or code).
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from fractions import Fraction
import json
import re
from pathlib import Path

CLEAN_SHA="fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31"
STATES=("pass","fail","unverified")


def verdict(family,value,expected):
    status="unverified" if value is None else ("pass" if value==expected else "fail")
    return {"family":family,"status":status}


def last_number(answer):
    m=re.search(r"=\s*(-?\d+)\s*[.!]?\s*$",answer.strip())
    return int(m.group(1)) if m else None


def inspect_math(prompt,answer):
    m=re.match(r"^(?:Berechne|Compute)\s+(-?\d+)\s*([+-])\s*(-?\d+)\b",prompt)
    if m:
        a,op,b=int(m[1]),m[2],int(m[3])
        v=re.match(r"^\s*(-?\d+)\s*([+-])\s*(-?\d+)\s*=\s*(-?\d+)\b",answer)
        if not v:return verdict("arithmetic",None,0)
        if (int(v[1]),v[2],int(v[3]))!=(a,op,b):return verdict("arithmetic",False,True)
        return verdict("arithmetic",int(v[4]),a+b if op=="+" else a-b)
    m=re.match(r"^(?:Wie viel ist|What is)\s+(\d+)\s+(?:mal|times)\s+(\d+)\b",prompt)
    if m:
        a,b=map(int,m.groups())
        v=re.match(r"^\s*(\d+)\s*[·×*]\s*(\d+)\s*=\s*(-?\d+)\b",answer)
        if not v:return verdict("multiplication",None,0)
        if (int(v[1]),int(v[2]))!=(a,b):return verdict("multiplication",False,True)
        return verdict("multiplication",int(v[3]),a*b)
    m=re.match(r"^(?:Löse die Gleichung|Solve)\s+(-?\d+)x\s*\+\s*\((-?\d+)\)\s*=\s*(-?\d+)\b",prompt)
    if m:
        a,b,c=map(int,m.groups())
        if a==0:return verdict("linear_equation",None,0)
        v=re.search(r"\bx\s*=\s*(-?\d+)\b",answer)
        return verdict("linear_equation",int(v[1]) if v else None,Fraction(c-b,a))
    m=re.match(r"^(?:Bestimme|Find)\s+(\d+)%\s+(?:von|of)\s+(\d+)\b",prompt)
    if m:
        pct,base=map(int,m.groups())
        return verdict("percentage",last_number(answer),Fraction(pct*base,100))
    m=re.match(r"^(?:Berechne|Compute)\s+(\d+)/(\d+)\s*[·×]\s*(\d+)\b",prompt)
    if m:
        n,d,mult=map(int,m.groups())
        if d==0:return verdict("fraction_product",None,0)
        v=re.search(r"=\s*(-?\d+)(?:/(\d+))?\s*[.!]?\s*$",answer.strip())
        if not v or (v[2] and int(v[2])==0):return verdict("fraction_product",None,0)
        observed=Fraction(int(v[1]),int(v[2]) if v[2] else 1)
        return verdict("fraction_product",observed,Fraction(n*mult,d))
    if ("Welche Kraft wirkt" in prompt or "What force acts" in prompt) and "F = m·a" in prompt:
        m=re.search(r"\bm\s*=\s*(\d+)\s*kg\s+(?:und|and)\s+a\s*=\s*(\d+)\s*m/s²",prompt)
        if m:
            v=re.search(r"=\s*(-?\d+)\s*N\s*[.!]?\s*$",answer)
            return verdict("force",int(v[1]) if v else None,int(m[1])*int(m[2]))
    m=re.match(r"^(?:Setze die Zahlenfolge|Continue the sequence)\s+(-?\d+),\s*(-?\d+),\s*(-?\d+),\s*(-?\d+),",prompt)
    if m:
        a,b,c,d=map(int,m.groups())
        if b-a!=c-b or c-b!=d-c:return verdict("arithmetic_sequence",None,0)
        v=re.search(r"(?:nächste Zahl ist|next number is)\s+(-?\d+)",answer,re.I)
        step=re.search(r"(?:Differenz ist immer|difference is always)\s+(-?\d+)",answer,re.I)
        if not v or not step:return verdict("arithmetic_sequence",None,0)
        return verdict("arithmetic_sequence",(int(step[1]),int(v[1])),(b-a,d+b-a))
    return verdict("unsupported",None,0)


def inspect_code(prompt,answer):
    # Dataset source is untrusted; never exec or import generated code.
    tick=chr(96)
    p=re.search(r"(?:Python-Funktion|Python function)\s+"+re.escape(tick)+r"(\w+)\(a,\s*b\)"+re.escape(tick),prompt)
    if not p or not ("addiert" in prompt or "adds two numbers" in prompt):
        return verdict("unsupported",None,0)
    name=p[1]
    example=re.search(r"(?:mit|with)\s+(-?\d+)\s+(?:und|and)\s+(-?\d+)\.?\s*$",prompt)
    code=re.fullmatch(r"\s*"+re.escape(tick*3)+r"python\s*\n(.*?)\n"+re.escape(tick*3)+r"\s*",answer,re.S)
    if not code or not example:return verdict("python_add",None,0)
    try:
        tree=ast.parse(code[1])
    except (SyntaxError,ValueError):
        return verdict("python_add",False,True)
    if len(tree.body)!=2 or not isinstance(tree.body[0],ast.FunctionDef):
        return verdict("python_add",False,True)
    f=tree.body[0]
    if f.name!=name or [a.arg for a in f.args.args]!=["a","b"] or len(f.body)!=1:
        return verdict("python_add",False,True)
    ret=f.body[0]
    ok=(isinstance(ret,ast.Return) and isinstance(ret.value,ast.BinOp)
        and isinstance(ret.value.op,ast.Add)
        and isinstance(ret.value.left,ast.Name) and ret.value.left.id=="a"
        and isinstance(ret.value.right,ast.Name) and ret.value.right.id=="b")
    if not ok:return verdict("python_add",False,True)
    x,y=map(int,example.groups())
    # For this narrow reference, also check the sample call and its comment.
    call=re.search(r"print\(\s*"+re.escape(name)+r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*\)\s*#\s*(-?\d+)\s*$",code[1],re.M)
    if not call:return verdict("python_add",None,0)
    return verdict("python_add",tuple(map(int,call.groups())),(x,y,x+y))


def inspect_reference(domain,prompt,answer):
    if not all(isinstance(v,str) for v in (domain,prompt,answer)):
        raise ValueError("expected string reference fields")
    if domain=="math":return inspect_math(prompt,answer)
    if domain=="code":return inspect_code(prompt,answer)
    if domain=="format":
        m=re.match(r"^Formuliere präziser:\s*['‘]Die Werte (-?\d+) und (-?\d+) sind irgendwie unterschiedlich[.]['’]",prompt)
        if m:
            a,b=map(int,m.groups())
            result=re.search(r"unterscheiden sich um\s+(-?\d+)",answer)
            return verdict("absolute_difference",int(result[1]) if result else None,abs(a-b))
    return verdict("unsupported",None,0)


def audit_splits(splits):
    if not isinstance(splits,dict) or set(splits)!={"train","val","test"}:
        raise ValueError("expected train/val/test splits")
    result={"schema":"arctus350-reference-answer-integrity-v1",
            "scope":"only explicitly recognized templates; not full answer correctness",
            "privacy":"aggregate statuses only; no prompts, rows, IDs or code",
            "splits":{}}
    for split in ("train","val","test"):
        rows=splits[split]
        if not isinstance(rows,list) or not rows:raise ValueError("empty split")
        total=Counter({k:0 for k in STATES})
        by_domain={}
        by_family={}
        for row in rows:
            if not isinstance(row,dict) or not isinstance(row.get("messages"),list) or not isinstance(row.get("domain"),str):
                raise ValueError("invalid normalized dataset row")
            messages=row["messages"]
            if len(messages)==2 and messages[0].get("role")=="user" and messages[1].get("role")=="assistant":
                observed=inspect_reference(row["domain"],messages[0].get("content"),messages[1].get("content"))
            else:
                observed=verdict("unsupported_multiturn",None,0)
            st,family=observed["status"],observed["family"]
            if st not in STATES:raise ValueError("invalid audit status")
            total[st]+=1
            by_domain.setdefault(row["domain"],Counter({k:0 for k in STATES}))[st]+=1
            by_family.setdefault(family,Counter({k:0 for k in STATES}))[st]+=1
        result["splits"][split]={
            "rows":len(rows),"status":dict(total),
            "by_domain":{x:dict(v) for x,v in sorted(by_domain.items())},
            "by_family":{x:dict(v) for x,v in sorted(by_family.items())},
        }
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():raise FileExistsError("refuse overwrite of audit report")
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    manifest,splits=validate_dataset_root(args.data)
    if manifest["manifest_sha256"]!=CLEAN_SHA:
        raise ValueError("clean dataset SHA256 mismatch")
    validate_prompt_holdouts(splits)
    report=audit_splits(splits)
    report["dataset_manifest_sha256"]=CLEAN_SHA
    args.out.write_text(json.dumps(report,indent=2,ensure_ascii=False,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
