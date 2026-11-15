"""CPU-only regrade of saved original answers from #43709097701.
Never prints questions, answers or gold values; never invokes models or GPU.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

RUN_ID = "43709097701"
SRC = "737413f03b347676108d507ff3fe7693975dd269"
SEALED = "74cf83ce3e9b3644b306556438a3e51373589ffdaf254dc84abb74e08fd04258"
CATS = ("mathematics", "logic", "programming", "mixed")
FOLDER = "neral-pr542-sealed100-43709097701-1"
STRICT = re.compile(r"\s*(?:(?:FINAL_ANSWER|NERAL_FINAL_TEXT):\s*)?(-?\d{1,10})\s*")
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d{1,11})(?!\w|\.\d)")
LABEL = re.compile(r"(?:FINAL_ANSWER|NERAL_FINAL_TEXT|final\s+(?:answer|result)|"
    r"(?:the\s+)?(?:answer|result|total|count)\s*(?:is|equals|:|=)|"
    r"(?:there\s+(?:are|is)\s+(?:exactly\s+)?)|"
    r"(?:thus|therefore|hence)\s*[:,]?\s*(?:the\s+)?(?:answer\s+is\s+)?|"
    r"ergebnis\s*(?:ist|:)|antwort\s*(?:ist|:))",re.IGNORECASE)

def val(x):
    x=x.replace(",","").replace(" ","").removeprefix("+")
    if not re.fullmatch(r"-?\d{1,11}",x):return None
    n=int(x)
    return n if abs(n)<=1000000000 else None

def extract(t):
    if type(t) is not str or not t.strip(): return None,"empty"
    t=t.replace("−","-").replace("–","-").replace("\u00a0"," ").strip()
    z=STRICT.fullmatch(t)
    if z:return int(z.group(1)),"strict"
    fence=chr(96)*3
    if len(t)>32768 or t.count(fence)%2:return None,"ambiguous"
    tail=t[-650:]
    # Strong end-answer evidence wins over intermediate calculations.
    boxes=list(re.finditer(r"\\boxed\s*\{\s*([-+]?\d{1,10})\s*\}",tail))
    if boxes:
        return val(boxes[-1].group(1)),"boxed"
    labels=[]
    for m in LABEL.finditer(tail):
        following=tail[m.end():m.end()+140]
        num=NUM.search(following)
        if num and len(following[:num.start()].strip(" *:_\\\t\n\r{}[]()"))<=20:
            n=val(num.group())
            if n is not None:labels.append((m.start(),n))
    if labels:return labels[-1][1],"labeled"
    # Final isolated integer line, permitting Markdown emphasis and LaTex.
    last=t.splitlines()[-1].strip().strip(" *_$;.!"+chr(96))
    last=last.removeprefix(r"\(").removesuffix(r"\)")
    if re.fullmatch(r"[-+]?\d{1,11}(?:,\d{3})*",last):
        return val(last),"standalone"
    # Single numerical result in closing prose; abstain on ambiguous lines.
    paragraph=t.split("\n\n")[-1]
    if fence not in paragraph:
        sentence=re.split(r"(?<=[.!?])\s+(?=[A-Z])",paragraph)[-1][-240:]
        nums=[val(z.group()) for z in NUM.finditer(sentence)]
        if len(nums)==1 and re.search(
            r"\b(?:ways|solutions|outcomes|colorings|rectangles|seconds|"
            r"trips|strings|orders|integers|result|answer|total|is|are|"
            r"equals|gives|makes|hence|therefore|thus)\b",sentence,re.I):
            return nums[0],"one_number_in_final_sentence"
    return None,"ambiguous"

def locate():
    roots=[]
    for key in ("RUNNER_TEMP","RUNNER_WORKSPACE","GITHUB_WORKSPACE",
                "TEMP","TMP","USERPROFILE"):
        v=os.environ.get(key)
        if v:
            p=Path(v)
            roots.extend([(key,p),(key+"_parent",p.parent)])
    for disk in ("C:","D:"):
        for name in ("actions-runner","actions_runner","runner","gh-runner",
                     "neral-actions-runner","NERAL-PR542-GPU-RAW"):
            roots.append(("runner_root",Path(disk+"\\")/name))
    tried=set()
    for kind,root in roots:
        for p in (root,root/"_temp",root/"_work"/"_temp",
                  root/"AppData"/"Local"/"Temp"):
            candidate=p/FOLDER/"private-results.jsonl"
            if str(candidate).lower() in tried:continue
            tried.add(str(candidate).lower())
            try:
                if candidate.is_file() and candidate.stat().st_size>1000:
                    return kind,candidate,len(tried)
            except OSError:pass
    return None,None,len(tried)

def strict(t,g):
    m=STRICT.fullmatch(t) if type(t) is str else None
    return bool(m and int(m.group(1))==g)

def grade(path):
    data=[]
    with path.open("r",encoding="utf8") as stream:
        for line in stream:
            if line.strip():
                x=json.loads(line)
                m=x["metadata"]
                data.append((x["id"],m["category"],int(x["gold_integer"]),
                             x.get("base_reply"),x.get("candidate_reply"),m))
    if len(data)!=100 or len({v[0] for v in data})!=100:
        raise ValueError("Wrong original source run count")
    for category in CATS:
        have={v[0] for v in data if v[1]==category}
        expected={f"blind100-{category}-{n:02d}" for n in range(1,26)}
        if have!=expected:raise ValueError("Wrong original source case IDs")
    old_b=sum(strict(v[3],v[2]) for v in data)
    old_c=sum(strict(v[4],v[2]) for v in data)
    old_proof=sum(v[5]["source_verified"] is True for v in data)
    old_loss=sum(strict(v[3],v[2]) and not strict(v[4],v[2]) for v in data)
    if (old_b,old_c,old_proof,old_loss)!=(2,12,10,0):
        raise ValueError("Raw content does not reproduce 43709097701 scalar evidence")
    rows=[]
    for ident,cat,gold,base,cand,meta in data:
        bv,bmode=extract(base)
        cv,cmode=extract(cand)
        refusal=meta["selection_policy"]=="explicit_safe_refusal"
        if refusal:cv,cmode=None,"safe_refusal"
        rows.append(dict(id=ident,category=cat,
             b=(bv is not None and bv==gold),
             c=(cv is not None and cv==gold),bm=bmode,cm=cmode,
             proof=bool(meta["source_verified"]),refusal=refusal))
    breakdown={}
    for cat in CATS:
        z=[x for x in rows if x["category"]==cat]
        breakdown[cat]={"qwen":sum(v["b"] for v in z),
                         "neral":sum(v["c"] for v in z),
                         "wins":sum(v["c"] and not v["b"] for v in z),
                         "regressions":sum(v["b"] and not v["c"] for v in z),
                         "count":len(z)}
    r=[v for v in rows if v["b"] and not v["c"]]
    proof=[v for v in rows if v["proof"]]
    return {
      "status":"offline_cpu_regrade_complete",
      "source_run":RUN_ID,"source_commit":SRC,"sealed_sha256":SEALED,
      "case_count":len(rows),
      "categories":breakdown,
      "qwen_correct":sum(v["b"] for v in rows),
      "neral_correct":sum(v["c"] for v in rows),
      "net_gain":sum(int(v["c"])-int(v["b"]) for v in rows),
      "regression_count":len(r),
      "regression_ids":[v["id"] for v in r],
      "regressions_among_13_refusals":[v["id"] for v in r if v["refusal"]],
      "safe_refusals":sum(v["refusal"] for v in rows),
      "safe_refusals_qwen_correct_ids":[v["id"] for v in rows if v["refusal"] and v["b"]],
      "source_proof_total":len(proof),
      "source_proof_neral_correct":sum(v["c"] for v in proof),
      "source_proof_qwen_correct":sum(v["b"] for v in proof),
      "source_proof_incremental_wins":sum(v["c"] and not v["b"] for v in proof),
      "source_proof_regressions":sum(v["b"] and not v["c"] for v in proof),
      "qwen_ambiguous_ids":[v["id"] for v in rows if v["bm"]=="ambiguous"],
      "neral_ambiguous_ids":[v["id"] for v in rows if v["cm"]=="ambiguous"],
      "extraction_modes":{"qwen":dict(Counter(v["bm"] for v in rows)),
                          "neral":dict(Counter(v["cm"] for v in rows))},
      "rubric":"Fixed deterministic final-answer extraction on both arms; ambiguous answers not counted",
      "caveat":"Automated review is not manual semantic adjudication",
      "inference_calls":0,"gpu_calls":0,"model_training":False,
      "raw_outputs_uploaded":False,"merged_to_main":False
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--self-test",action="store_true")
    a=p.parse_args()
    if a.self_test:
        for t,expect in (("42",42),("FINAL_ANSWER: 42",42),
             ("The answer is 42.",42),("Therefore \\boxed{42}",42),
             ("First 10. The final answer is **42**.",42),
             ("4+10=14. The result is 42.",42)):
            assert extract(t)[0]==expect,("synthetic test failure",t)
        assert extract("10, 20, 30")[0] is None
        print("OFFLINE_FIXED_RUBRIC_SYNTHETIC_TESTS_PASS")
        return
    kind,path,checked=locate()
    if path is None:
        result={"status":"original_raw_missing",
          "source_run":RUN_ID,"sealed_sha256":SEALED,
          "candidate_paths_checked":checked,
          "note":"Old run stored private-results.jsonl under RUNNER_TEMP; file not found in original-run paths",
          "inference_calls":0,"gpu_calls":0,
          "merged_to_main":False,"raw_outputs_uploaded":False}
    else:
        result=grade(path)
        result["location_class"]=kind
    if a.out.exists():raise FileExistsError("Result path already occupied")
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open("x",encoding="utf8",newline="\n") as f:
        json.dump(result,f,sort_keys=True,indent=2)
        f.write("\n")
    print("OFFLINE_CPU_REVIEW_STATUS "+result["status"])
    print("OFFLINE_GPU_CALLS 0")
    if result["status"]=="offline_cpu_regrade_complete":
        print("OFFLINE_CPU_SCALARS "+json.dumps(result,sort_keys=True))
    else:
        print("OFFLINE_SOURCE_FILE_MISSING; NO_REGRADE_POSSIBLE")

if __name__=="__main__":
    main()
