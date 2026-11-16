"""Evidence-gated training JSONL export. Raw status flags cannot certify a result."""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
from .symbolic import verify_symbolic
from .graphs import verify_graph
from .constraints import solve_constraints


def canonical_hash(task:dict)->str:
    if not isinstance(task,dict): raise ValueError("task")
    normalized=json.loads(json.dumps(task,sort_keys=True,ensure_ascii=False))
    code=normalized.get("spec",{}).get("code") if isinstance(normalized.get("spec"),dict) else None
    if isinstance(code,str):
        normalized["spec"]["code"]=ast.dump(ast.parse(code),include_attributes=False)
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,ensure_ascii=False,separators=(",",":")).encode()).hexdigest()


def canonical_answer(answer)->str:
    return json.dumps(answer,ensure_ascii=False,sort_keys=True,separators=(",",":"))


def reverify(row:dict)->dict|None:
    """Recompute from structured task instead of trusting a 'verified' log property."""
    if not isinstance(row,dict) or not isinstance(row.get("task"),dict):return None
    task=row["task"]; tool=task.get("tool"); spec=task.get("spec")
    if not isinstance(spec,dict): return None
    try:
        if tool=="sympy": computed=verify_symbolic(spec)
        elif tool=="networkx": computed=verify_graph(spec)
        elif tool=="z3": computed=solve_constraints(spec)
        else: return None  # Python exit, AST repair, arbitrary traceback are not proof
        if not computed.get("verified") or computed.get("status")=="unsat" and row.get("answer") is not None:
            return None
        expected=canonical_answer(computed["result"])
        if canonical_answer(row.get("answer"))!=expected:return None
        return computed
    except (ValueError,TypeError,KeyError,RuntimeError,ImportError):
        return None


def export_verified(rows:list[dict],folder:str|Path)->dict:
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    valid=[]; hashes=set()
    for row in rows:
        cert=reverify(row)
        if cert is None:continue
        sha=canonical_hash(row["task"])
        if sha in hashes:continue
        hashes.add(sha)
        # No private chain-of-thought: only reproducible solver/tool steps.
        task=row["task"]
        record={"id":sha,"prompt":str(row.get("prompt",""))[:1000],
          "task":task,"answer":row["answer"],"evidence":{"tool":task["tool"],"method":cert.get("method"),
          "independently_recomputed":True},"trace":[{"tool":task["tool"],"spec":task["spec"],"result":cert["result"]}]}
        valid.append(record)
    valid.sort(key=lambda r:r["id"])
    counts={"input":len(rows),"verified":len(valid),"rejected":len(rows)-len(valid),"splits":{}}
    buckets={"train":[],"val":[],"test":[]}
    for row in valid:
        # Group by a *semantic problem variant* rather than by random row.
        family=str(next((v.get("group") for v in rows if isinstance(v,dict) and v.get("task")==row["task"]),row["task"]["tool"]))
        bucket=["train","train","train","train","train","train","val","test"][int(hashlib.sha256(family.encode()).hexdigest(),16)%8]
        buckets[bucket].append(row)
    # No test data is included in the training-only consolidated outputs.
    for name,bucket in buckets.items():
        counts["splits"][name]=len(bucket)
        for tier in ("helios_terra_v2","luna_distill_v1"):
            with (folder/f"{tier}.{name}.jsonl").open("w",encoding="utf-8") as handle:
                for row in bucket:
                    payload=row if tier.startswith("helios") else {"id":row["id"],"prompt":row["prompt"],
                     "answer":row["answer"],"tool":row["task"]["tool"],"verified":True}
                    handle.write(json.dumps(payload,ensure_ascii=False)+"\n")
    for tier in ("helios_terra_v2","luna_distill_v1"):
        (folder/f"{tier}.jsonl").write_text((folder/f"{tier}.train.jsonl").read_text(encoding="utf-8"),encoding="utf-8")
    (folder/"manifest.json").write_text(json.dumps(counts,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    return counts


def export_log_file(path:str|Path,destination:str|Path)->dict:
    rows=[]
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                if len(line)>20000: continue
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: continue
    return export_verified(rows,destination)
