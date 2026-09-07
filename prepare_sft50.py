"""Prepare contamination-screened Neral-50M instruction data from local declared sources.

No network calls and no external model calls are made. Every source is a local JSONL file
listed in sft50_sources/manifest.json with an exact SHA-256 and provenance/license metadata.
All rows sharing the same normalized user prompt are assigned to the same split before any
training quota is selected, preventing identical-question leakage across train/validation/test.
Frozen Neral benchmark literals are screened from prompt and answer text.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

ALLOWED_LANGUAGES={"de","en"}
ALLOWED_DOMAINS={
    "chat","explanation","code","math","science","focus","translation",
    "reasoning","summary","safety","format","knowledge",
}
ID_RE=re.compile(r"[A-Za-z0-9_.-]{1,96}")
SECRET_OR_CONTACT=re.compile(
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:sk-|gh[pousr]_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]{16,}|"
    r"AKIA[A-Z0-9]{16}|-----BEGIN .*PRIVATE KEY",
    re.I,
)
CONTROL_RE=re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
FROZEN_PY_ASSIGNMENTS={
    "benchmark_v3.py":{"PAIR_CASES","FREE_CHAT","BASE_PREFIXES"},
    "benchmark_v4.py":{"PAIRS","ENTROPY_PREFIXES","BASE_GENERATION"},
}
SPLIT_CONTRACT_VERSION=2


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())


def prompt_group_hash(prompt: str) -> str:
    normalized=normalize(prompt)
    if not normalized:raise ValueError("normalized prompt is empty")
    return hashlib.sha256(("SFT50:PROMPT_GROUP:v2:"+normalized).encode("utf8")).hexdigest()


def split_for_group(group_sha256: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{64}",group_sha256):raise ValueError("invalid prompt group SHA-256")
    bucket=int(group_sha256[:8],16)%100
    return "holdout" if bucket<2 else "test" if bucket<5 else "val" if bucket<10 else "train"


def split_for_prompt(prompt: str) -> str:
    return split_for_group(prompt_group_hash(prompt))


def split_for(source_id,row_id):
    """Historical row-level splitter kept for reproducibility tests; new data uses prompt groups."""
    bucket=int(hashlib.sha256(f"SFT50:{source_id}:{row_id}".encode()).hexdigest()[:8],16)%100
    return "holdout" if bucket<2 else "test" if bucket<5 else "val" if bucket<10 else "train"


def _collect_strings(value,out):
    if isinstance(value,str):out.append(value)
    elif isinstance(value,(list,tuple,set)):
        for item in value:_collect_strings(item,out)
    elif isinstance(value,dict):
        for k,v in value.items():_collect_strings(k,out);_collect_strings(v,out)


def benchmark_literals(repo_root: Path):
    phrases=set();files={}
    for rel in ("benchmarks50/regression.jsonl","benchmarks50/raw_math_v1.jsonl"):
        path=repo_root/rel
        if not path.is_file():continue
        files[rel]=sha256_file(path)
        for line in path.read_text(encoding="utf8").splitlines():
            if not line.strip():continue
            row=json.loads(line)
            values=[];_collect_strings(row,values)
            for value in values:
                n=normalize(value)
                if len(n)>=20:phrases.add(n)
    for rel,names in FROZEN_PY_ASSIGNMENTS.items():
        path=repo_root/rel
        if not path.is_file():continue
        files[rel]=sha256_file(path)
        tree=ast.parse(path.read_text(encoding="utf8"),filename=str(path));values=[]
        for node in tree.body:
            if not isinstance(node,(ast.Assign,ast.AnnAssign)):continue
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            ids={t.id for t in targets if isinstance(t,ast.Name)}
            if not ids&names:continue
            try:value=ast.literal_eval(node.value)
            except (ValueError,TypeError,SyntaxError):continue
            _collect_strings(value,values)
        for value in values:
            n=normalize(value)
            if len(n)>=20:phrases.add(n)
    return sorted(phrases,key=len,reverse=True),files


def clean_field(value,kind):
    if not isinstance(value,str):return None,f"{kind}_not_text"
    value=unicodedata.normalize("NFC",value).strip()
    if "\ufffd" in value or CONTROL_RE.search(value):return None,f"{kind}_encoding_or_control"
    if SECRET_OR_CONTACT.search(value):return None,f"{kind}_contact_or_secret"
    raw=len(value.encode("utf8"))
    minimum=4 if kind=="answer" else 8
    maximum=3072 if kind=="answer" else 3000
    if raw<minimum:return None,f"{kind}_too_short"
    if raw>maximum:return None,f"{kind}_too_long"
    if re.search(r"(.)\1{15,}",value):return None,f"{kind}_character_run"
    return value,None


def pair_from_row(row):
    if isinstance(row.get("prompt"),str) and isinstance(row.get("answer"),str):
        return row["prompt"],row["answer"]
    messages=row.get("messages")
    if isinstance(messages,list) and len(messages)==2:
        a,b=messages
        if isinstance(a,dict) and isinstance(b,dict) and a.get("role")=="user" and b.get("role")=="assistant":
            return a.get("content"),b.get("content")
    return None,None


def validate_manifest(manifest,root: Path):
    if manifest.get("version")!=1:raise ValueError("SFT manifest version must be 1")
    sources=manifest.get("sources")
    if not isinstance(sources,list) or not sources:raise ValueError("SFT manifest needs sources")
    seen=set();total=0.0;resolved=[];base=root.resolve()
    for src in sources:
        sid=src.get("id")
        if not isinstance(sid,str) or not ID_RE.fullmatch(sid) or sid in seen:raise ValueError(f"invalid/duplicate source id: {sid!r}")
        seen.add(sid)
        if src.get("language") not in ALLOWED_LANGUAGES:raise ValueError(f"{sid}: language must be de/en")
        if src.get("domain") not in ALLOWED_DOMAINS:raise ValueError(f"{sid}: unsupported domain")
        frac=src.get("target_fraction")
        if type(frac) not in (int,float) or not 0<float(frac)<=1:raise ValueError(f"{sid}: invalid target_fraction")
        total+=float(frac)
        for key in ("file","sha256","license","license_url","provenance_url"):
            if not isinstance(src.get(key),str) or not src[key].strip():raise ValueError(f"{sid}: missing {key}")
        if not re.fullmatch(r"[0-9a-fA-F]{64}",src["sha256"]):raise ValueError(f"{sid}: invalid SHA-256")
        path=(root/src["file"]).resolve()
        if base not in path.parents:raise ValueError(f"{sid}: path escapes source root")
        if not path.is_file():raise FileNotFoundError(path)
        actual=sha256_file(path)
        if actual.lower()!=src["sha256"].lower():raise ValueError(f"{sid}: SHA-256 mismatch")
        resolved.append(src|{"path":path,"target_fraction":float(frac),"sha256":actual})
    if not math.isclose(total,1.0,abs_tol=1e-6):raise ValueError(f"target_fraction must sum to 1.0, got {total}")
    return resolved


def load_rows(sources,phrases,reject,max_seq=4096):
    exact=set();rows=[]
    for src in sources:
        ids=set()
        with src["path"].open("r",encoding="utf8") as f:
            for line_no,line in enumerate(f,1):
                if not line.strip():continue
                try:row=json.loads(line)
                except json.JSONDecodeError as e:raise ValueError(f"{src['id']} line {line_no}: invalid JSON") from e
                rid=row.get("id")
                if not isinstance(rid,(str,int)):raise ValueError(f"{src['id']} line {line_no}: missing id")
                rid=str(rid)
                if rid in ids:raise ValueError(f"{src['id']}: duplicate id {rid}")
                ids.add(rid)
                prompt,answer=pair_from_row(row)
                prompt,reason=clean_field(prompt,"prompt")
                if reason:reject[reason]+=1;continue
                answer,reason=clean_field(answer,"answer")
                if reason:reject[reason]+=1;continue
                if len((prompt+"\n"+answer).encode("utf8"))>3800:reject["pair_too_long"]+=1;continue
                formatted='User: '+prompt+'\nAssistant: '+answer+'\n'
                if len(formatted.encode('utf8'))+1>max_seq:
                    reject['pair_exceeds_training_sequence']+=1;continue
                domain=src["domain"]
                if src.get("allow_row_domains") is True and row.get("domain") is not None:
                    if row.get("domain") not in ALLOWED_DOMAINS:reject["invalid_row_domain"]+=1;continue
                    domain=row["domain"]
                nprompt=normalize(prompt);nanswer=normalize(answer)
                if any(p in nprompt or p in nanswer for p in phrases):reject["frozen_benchmark_contamination"]+=1;continue
                digest=hashlib.sha256((nprompt+"\0"+nanswer).encode()).hexdigest()
                if digest in exact:reject["exact_pair_duplicate"]+=1;continue
                exact.add(digest)
                group=prompt_group_hash(prompt)
                rows.append({
                    "id":rid,"source_id":src["id"],"language":src["language"],"domain":domain,
                    "prompt":prompt,"answer":answer,"split":split_for_group(group),
                    "prompt_group_sha256":group,"pair_sha256":digest,
                    "selection_hash":hashlib.sha256(f"SFT50:968:{src['id']}:{rid}".encode()).hexdigest(),
                    "prompt_bytes":len(prompt.encode("utf8")),"answer_bytes":len(answer.encode("utf8")),
                })
    return rows


def audit_prompt_group_splits(rows):
    """Return an explicit cross-split prompt-group audit and fail on any leakage."""
    locations=collections.defaultdict(set);counts=collections.Counter()
    for row in rows:
        group=row.get("prompt_group_sha256");split=row.get("split")
        if not isinstance(group,str) or not re.fullmatch(r"[0-9a-f]{64}",group):raise ValueError("row missing valid prompt_group_sha256")
        if split not in {"train","val","test","holdout"}:raise ValueError("row missing valid split")
        locations[group].add(split);counts[split]+=1
    overlaps={group:sorted(splits) for group,splits in locations.items() if len(splits)>1}
    audit={"prompt_groups":len(locations),"rows":len(rows),"rows_by_split":dict(counts),"cross_split_prompt_group_overlap":len(overlaps),"overlap_examples":dict(list(sorted(overlaps.items()))[:20])}
    if overlaps:raise ValueError(f"normalized prompt leakage across splits: {len(overlaps)} groups")
    return audit


def select_train(rows,sources,target_examples):
    by=collections.defaultdict(list)
    for row in rows:
        if row["split"]=="train":by[row["source_id"]].append(row)
    selected=[];meta={};remaining=target_examples
    for i,src in enumerate(sources):
        candidates=sorted(by[src["id"]],key=lambda r:r["selection_hash"])
        quota=remaining if i==len(sources)-1 else round(target_examples*src["target_fraction"])
        quota=max(1,quota);remaining-=quota
        if len(candidates)<quota:raise ValueError(f"{src['id']}: only {len(candidates)} clean train examples; quota is {quota}")
        chosen=candidates[:quota];selected.extend(chosen)
        meta[src["id"]]={"target_examples":quota,"selected_examples":len(chosen),"available_train_examples":len(candidates)}
    if len(selected)!=target_examples:raise AssertionError("selected train count mismatch")
    return selected,meta


def auto_train_target(rows,sources,recommended=None,cap=24000):
    available=collections.Counter(r["source_id"] for r in rows if r["split"]=="train")
    limits=[]
    for src in sources:
        n=available[src["id"]]
        if n<1:raise ValueError(f"{src['id']}: no clean train examples after filtering/split")
        limits.append(math.floor(n/src["target_fraction"]))
    capacity=max(0,min(limits)-len(sources))
    if recommended is not None:
        try:capacity=min(capacity,int(recommended))
        except (TypeError,ValueError):raise ValueError("recommended_train_examples must be an integer")
    target=min(cap,capacity)
    if target<1000:raise ValueError(f"Only {target} safely mixable train examples remain after cleaning")
    return target


def write_outputs(rows,selected,sources,out: Path,target,reject,selection,bench_files,source_manifest_sha):
    out.mkdir(parents=True,exist_ok=True);selected_keys={(r["source_id"],r["id"]) for r in selected};summary={};written=[]
    for split in ("train","val","test","holdout"):
        if split=="train":chosen=[r for r in rows if (r["source_id"],r["id"]) in selected_keys]
        else:chosen=[r for r in rows if r["split"]==split]
        chosen=sorted(chosen,key=lambda r:(r["source_id"],r["selection_hash"]))
        clean=[{k:v for k,v in r.items() if k not in {"selection_hash","split"}} for r in chosen]
        path=out/f"{split}.jsonl";path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in clean),encoding="utf8")
        summary[split]={"examples":len(clean),"answer_bytes":sum(r["answer_bytes"] for r in clean),"sha256":sha256_file(path),"by_language":dict(collections.Counter(r["language"] for r in clean)),"by_domain":dict(collections.Counter(r["domain"] for r in clean)),"prompt_groups":len({r["prompt_group_sha256"] for r in clean})}
        written.extend(dict(r,split=split) for r in clean)
    overlap_audit=audit_prompt_group_splits(written)
    public_sources=[{k:v for k,v in s.items() if k!="path"} for s in sources]
    manifest={
        "version":1,"purpose":"Neral-50M assistant-only instruction tuning","target_train_examples":target,
        "sources":public_sources,"selection":selection,"splits":summary,"rejected":dict(reject),
        "source_manifest_sha256":source_manifest_sha,"frozen_benchmark_files":bench_files,
        "format":"User prompt + Assistant answer; assistant answer and EOS only are loss targets during training",
        "split_contract_version":SPLIT_CONTRACT_VERSION,
        "split":"SHA256(SFT50:PROMPT_GROUP:v2:normalize(prompt)): 90/5/3/2 train/val/test/holdout before train quota selection; identical normalized prompts cannot cross splits",
        "prompt_group_overlap_audit":overlap_audit,
        "external_model_calls":False,"raw_source_uploaded":False,
        "contamination_note":"Exact normalized long literals from frozen 100-prompt, RawMath, v3 and v4 suites are rejected; not exhaustive semantic detection.",
    }
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    return manifest


def build(source_root: Path,out: Path,target_train_examples: int,repo_root: Path,max_seq=512):
    if target_train_examples<0:raise ValueError("target_train_examples cannot be negative")
    manifest_path=source_root/"manifest.json";manifest=json.loads(manifest_path.read_text(encoding="utf8"))
    sources=validate_manifest(manifest,source_root);phrases,bench_files=benchmark_literals(repo_root);reject=collections.Counter()
    if not 32<=max_seq<=4096:raise ValueError('max_seq must be 32..4096')
    rows=load_rows(sources,phrases,reject,max_seq);audit_prompt_group_splits(rows)
    target=target_train_examples or auto_train_target(rows,sources,manifest.get("recommended_train_examples"))
    if target<1000:raise ValueError("Use at least 1000 train examples for Stage D")
    selected,selection=select_train(rows,sources,target)
    result=write_outputs(rows,selected,sources,out,target,reject,selection,bench_files,sha256_file(manifest_path))
    result['max_training_sequence']=max_seq
    (out/'manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source-root",default="sft50_sources");ap.add_argument("--out",default="sft50_data");ap.add_argument('--max-seq',type=int,default=512);ap.add_argument("--target-train-examples",type=int,default=0,help="0=auto-size safely after filtering; otherwise exact quota");args=ap.parse_args()
    result=build(Path(args.source_root),Path(args.out),args.target_train_examples,Path(__file__).resolve().parent,args.max_seq)
    print(json.dumps({"target_train_examples":result["target_train_examples"],"train":result["splits"]["train"],"val":result["splits"]["val"],"rejected":result["rejected"],"prompt_group_overlap_audit":result["prompt_group_overlap_audit"]},indent=2,ensure_ascii=False))


if __name__=="__main__":main()
