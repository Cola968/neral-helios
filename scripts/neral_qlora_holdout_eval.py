"""CPU-only evaluator for 24 curated, train-excluded NERAL holdout problems.

Evaluates syntactically valid, source-grounded predicted JSON without
conflating solver correctness with semantic translation. No model calls.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from neral_formal_constraints import PREFIX, try_solve_formal_constraints
from neral_turn0_extractor import parse_extracted_constraints


def compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)

def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def load_holdout(path: Path) -> list[dict]:
    rows=[json.loads(s) for s in path.read_text(encoding="utf-8").splitlines()
          if s.strip()]
    if len(rows)!=24 or len({x["id"] for x in rows})!=24:
        raise ValueError("Exactly 24 unique curated holdout cases are required")
    for row in rows:
        if row["split"]!="holdout" or row["source"]!="hand_curated_not_training":
            raise ValueError("Untrusted holdout provenance")
        envelope={"kind":"formal","complete":True,
                  "schema":row["schema"],"evidence":row["evidence"]}
        schema=json.loads(compact(row["schema"]))
        if isinstance(schema.get("variables"),list):
            schema["variables"]={r["name"]:r["domain"] for r in schema["variables"]}
        solved=try_solve_formal_constraints(PREFIX+compact(schema))
        rejections=[]
        parsed=parse_extracted_constraints(row["question"],compact(envelope),
                                          rejection_reason_sink=rejections)
        if type(solved) is not int or solved!=row["expected_count"] or parsed!=solved or rejections!=["accepted"]:
            raise ValueError("Curated holdout oracle/source grounding mismatch: "+row["id"])
    return rows

def assert_disjoint(holdout:list[dict],train_path:Path)->None:
    prompts={sha(x["question"]) for x in holdout}
    schemas={sha(compact(x["schema"])) for x in holdout}
    for line in train_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item=json.loads(line)
        question_hash=item["meta"]["question_sha256"]
        answer=json.loads(item["messages"][-1]["content"])
        if question_hash in prompts or sha(compact(answer["schema"])) in schemas:
            raise ValueError("Holdout appeared in train corpus")

def evaluate(holdout:list[dict],predictions:Path)->dict:
    """Input predictions JSONL: {"id": "holdout-...", "response": "{...}"}.

    The caller is responsible for actually running Qwen/NERAL; this scorer
    is not a simulated performance measurement.
    """
    submitted={}
    for line in predictions.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row=json.loads(line)
        if set(row)!={"id","response"} or not isinstance(row["response"],str):
            raise ValueError("Predictions must have exactly id and response")
        if row["id"] in submitted:
            raise ValueError("Duplicate prediction")
        submitted[row["id"]]=row["response"]
    if set(submitted)!={item["id"] for item in holdout}:
        raise ValueError("Missing or unexpected prediction IDs")
    correct=0
    by_family=Counter()
    total_family=Counter()
    parser_rejected=0
    for row in holdout:
        total_family[row["family"]]+=1
        status=[]
        parsed=parse_extracted_constraints(row["question"],
                 submitted[row["id"]], rejection_reason_sink=status)
        is_correct=type(parsed) is int and parsed==row["expected_count"]
        correct+=is_correct
        by_family[row["family"]]+=is_correct
        parser_rejected+=parsed is None
    return {"status":"24_case_curated_holdout_not_blind",
            "correct":correct,"total":len(holdout),
            "accuracy":correct/len(holdout),
            "parser_rejected":parser_rejected,
            "by_family":{k:{"correct":by_family[k],"total":total_family[k]}
                         for k in sorted(total_family)}}

def main()->int:
    p=argparse.ArgumentParser()
    p.add_argument("--holdout",required=True,type=Path)
    p.add_argument("--train",type=Path,help="Optional: verify train split exclusion")
    p.add_argument("--predictions",type=Path,help="Real model JSONL predictions")
    args=p.parse_args()
    rows=load_holdout(args.holdout)
    if args.train is not None:
        assert_disjoint(rows,args.train)
    if args.predictions is None:
        print(compact({"status":"holdout_oracle_verified",
                       "cases":len(rows),"model_score":"not_measured"}))
        return 0
    print(compact(evaluate(rows,args.predictions)))
    return 0
if __name__=="__main__":
    raise SystemExit(main())
