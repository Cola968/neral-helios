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


def _python_ast_fenced(answer):
    """Return AST for a fenced Python snippet; no compilation or execution."""
    fence=re.escape(chr(96)*3)
    code=re.fullmatch(r"\s*"+fence+r"python\s*\n(.*?)\n"+fence+r"\s*",answer,re.S)
    if not code:return None
    try:
        return ast.parse(code[1])
    except (SyntaxError,ValueError):
        return False


def _static_python_match(answer,reference):
    tree=_python_ast_fenced(answer)
    if tree is None:return None
    if tree is False:return False
    expected=ast.parse(reference)
    return ast.dump(tree,include_attributes=False)==ast.dump(expected,include_attributes=False)


def _last_python_output_comment(answer):
    comments=re.findall(r"#[ \t]*(-?\d+)[ \t]*$",answer,re.M)
    return int(comments[-1]) if comments else None


def inspect_code(prompt,answer):
    """Check known deterministic Python exercises, never run training code."""
    tick=re.escape(chr(96))
    # Example 1: addition function plus the worked example.
    m=re.search(r"(?:Python-Funktion|Python function)\s+"+tick+r"(\w+)\(a,\s*b\)"+tick,prompt)
    if m:
        name=m[1]
        example=re.search(r"(?:mit|with)\s+(-?\d+)\s+(?:und|and)\s+(-?\d+)\.?\s*$",prompt)
        if "addiert" in prompt or "adds two numbers" in prompt:
            if not example:return verdict("python_add",None,0)
            x,y=map(int,example.groups())
            reference="def "+name+"(a, b):\n    return a + b\n\nprint("+name+"("+str(x)+", "+str(y)+"))"
            structure=_static_python_match(answer,reference)
            comment=_last_python_output_comment(answer)
            return verdict("python_add",None if structure is None or comment is None else (structure,comment),(True,x+y))
        if "größeren Wert zurückgibt" in prompt or "returns the larger value" in prompt:
            if not example:return verdict("python_max",None,0)
            x,y=map(int,example.groups())
            reference="def "+name+"(a, b):\n    return a if a >= b else b\n\nprint("+name+"("+str(x)+", "+str(y)+"))"
            structure=_static_python_match(answer,reference)
            comment=_last_python_output_comment(answer)
            return verdict("python_max",None if structure is None or comment is None else (structure,comment),(True,max(x,y)))
    # Example 2: list(range(start, stop)) and the independently computed list.
    if re.match(r"^(?:Erkläre kurz, was|Briefly explain what)\s+",prompt):
        m=re.search(tick+r"list\(range\((\d+),\s*(\d+)\)\)"+tick,prompt)
        if m:
            start,stop=map(int,m.groups())
            result=re.search(r"(?:Liste ist|list is)\s+"+tick+r"(\[[0-9,\s-]*\])"+tick,answer,re.I)
            if not result or len(result[1])>300 or stop-start>100:
                return verdict("python_range",None,0)
            try: observed=ast.literal_eval(result[1])
            except (SyntaxError,ValueError):return verdict("python_range",None,0)
            if not isinstance(observed,list) or any(type(x) is not int for x in observed):
                return verdict("python_range",None,0)
            return verdict("python_range",observed,list(range(start,stop)))
    # Example 3: equality versus assignment in an if condition.
    if "Python-Fehler" in prompt or "Fix the Python error" in prompt:
        m=re.search(tick+r"if\s+([A-Za-z_]\w*)\s*=\s*(-?\d+):\s*print\(\1\)"+tick,prompt)
        if m:
            name,value=m[1],int(m[2])
            result=re.search(tick+r"if\s+([A-Za-z_]\w*)\s*==\s*(-?\d+):\s*print\(\1\)"+tick,answer)
            observed=(result[1],int(result[2])) if result else None
            return verdict("python_equality_fix",observed,(name,value))
    # Example 4: check whether an integer is even (AST exact structure).
    if "gerade ist" in prompt or "is even" in prompt:
        m=re.search(tick+r"([A-Za-z_]\w*)\s*=\s*(-?\d+)"+tick,prompt)
        if m:
            name,value=m[1],int(m[2])
            reference=(name+" = "+str(value)+"\nis_even = "+name+
                       " % 2 == 0\nprint(is_even)")
            return verdict("python_even",_static_python_match(answer,reference),True)
    # Example 5: interpret the sum() print expression.
    if "Python-Code aus" in prompt or "Python code print" in prompt:
        m=re.search(tick+r"values\s*=\s*(\[[0-9,\s-]*\]);\s*print\(sum\(values\)\)"+tick,prompt)
        if m and len(m[1])<300:
            try: values=ast.literal_eval(m[1])
            except (SyntaxError,ValueError):return verdict("python_sum",None,0)
            if not isinstance(values,list) or any(type(x) is not int for x in values):
                return verdict("python_sum",None,0)
            output=re.search(r"(?:Ausgabe ist|output is)\s+"+tick+r"(-?\d+)"+tick,answer,re.I)
            return verdict("python_sum",int(output[1]) if output else None,sum(values))
    # Example 6: dictionary literal and the exact requested value.
    if "Dictionary" in prompt or "dictionary" in prompt:
        m=re.search(r"(?:Schlüssel|key)\s+"+tick+r"([A-Za-z_]\w*)"+tick+
                    r"\s+(?:und dem Wert|and value)\s+(-?\d+)",prompt)
        if m:
            key,value=m[1],int(m[2])
            reference="data = {"+repr(key)+": "+str(value)+"}\nprint(data["+repr(key)+"])"
            match=_static_python_match(answer,reference)
            comment=_last_python_output_comment(answer)
            return verdict("python_dictionary",None if match is None or comment is None else (match,comment),(True,value))
    # Example 7: square function with a numerical sample.
    if "Quadrat zurückgibt" in prompt or "returns the square" in prompt:
        m=re.search(r"(?:Python-Funktion|Python function)\s+"+tick+
                    r"([A-Za-z_]\w*)\(([A-Za-z_]\w*)\)"+tick,prompt)
        example=re.search(r"(?:Beispiel|example)\s+([A-Za-z_]\w*)\s*=\s*(-?\d+)",prompt)
        if m and example:
            fn,var=m[1],m[2]
            if example[1]!=var:return verdict("python_square",None,0)
            n=int(example[2])
            reference=("def "+fn+"("+var+"):\n    return "+var+" * "+var+
                       "\n\nprint("+fn+"("+str(n)+"))")
            match=_static_python_match(answer,reference)
            comment=_last_python_output_comment(answer)
            return verdict("python_square",None if match is None or comment is None else (match,comment),(True,n*n))
    return verdict("unsupported",None,0)


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
