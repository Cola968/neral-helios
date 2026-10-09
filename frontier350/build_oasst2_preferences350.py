from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
try:
    from .oasst2_human350 import load_archive,indexes,path_to_root,valid_human_path,to_messages,split_for_root_prompt,contaminated,sha256_file,LICENSE,SOURCE,reviewed_human
except ImportError:
    from oasst2_human350 import load_archive,indexes,path_to_root,valid_human_path,to_messages,split_for_root_prompt,contaminated,sha256_file,LICENSE,SOURCE,reviewed_human

def _frozen(repo_root: Path) -> list[str]:
    import sys
    rr=str(Path(repo_root).resolve())
    if rr not in sys.path: sys.path.insert(0,rr)
    import prepare_sft50
    phrases,_=prepare_sft50.benchmark_literals(Path(repo_root))
    try:
        import prepare_sft50_i1
        extra,_=prepare_sft50_i1.targeted_literals(Path(repo_root)); phrases=sorted(set(phrases)|set(extra),key=len,reverse=True)
    except Exception: pass
    return phrases

def extract_preferences(archive: Path, *, repo_root: Path, verify_sha: bool=True, max_context_turns: int=7, max_bytes: int=600):
    if max_context_turns<1 or max_context_turns%2!=1: raise ValueError("max_context_turns must be odd")
    rows=load_archive(archive,verify_sha=verify_sha); by_id,children=indexes(rows); frozen=_frozen(repo_root)
    out={"train":[],"val":[],"test":[]}; seen=set(); skipped_holdout=0
    for pid,kids in sorted(children.items()):
        parent=by_id.get(pid)
        if not parent or parent.get("role")!="prompter" or not reviewed_human(parent): continue
        candidates=[r for r in kids if r.get("role")=="assistant" and r.get("lang")==parent.get("lang") and reviewed_human(r) and isinstance(r.get("rank"),int)]
        if len(candidates)<2: continue
        candidates=sorted(candidates,key=lambda r:(int(r["rank"]),r["message_id"]))
        if int(candidates[0]["rank"])==int(candidates[-1]["rank"]): continue
        context=path_to_root(pid,by_id)
        if not valid_human_path(context,end_role="prompter"): continue
        root_text=context[0]["text"]
        use=context[-max_context_turns:] if len(context)>max_context_turns else context
        if use[0].get("role")!="prompter" or use[-1].get("role")!="prompter": continue
        chosen,rejected=candidates[0],candidates[-1]
        texts=[r["text"].strip() for r in use]+[chosen["text"].strip(),rejected["text"].strip()]
        if sum(len(x.encode("utf8")) for x in texts)>max_bytes or contaminated(texts,frozen): continue
        digest=hashlib.sha256(json.dumps([to_messages(use),chosen["text"].strip(),rejected["text"].strip()],ensure_ascii=False,sort_keys=True).encode("utf8")).hexdigest()
        if digest in seen: continue
        seen.add(digest)
        split=split_for_root_prompt(root_text)
        if split=="holdout": skipped_holdout+=1; continue
        out[split].append({"id":f"oasst2-pref:{pid}","language":parent["lang"],"source":"oasst2-human-ranked","license":LICENSE,"context":to_messages(use),"chosen":chosen["text"].strip(),"rejected":rejected["text"].strip(),"chosen_rank":int(chosen["rank"]),"rejected_rank":int(rejected["rank"]),"pair_sha256":digest})
    return out,{"accepted":{k:len(v) for k,v in out.items()},"holdout_skipped":skipped_holdout}

def write_preferences(archive: Path, out_root: Path, repo_root: Path, *, verify_sha: bool=True, max_context_turns: int=7, max_bytes: int=600):
    out_root=Path(out_root)
    if out_root.exists() and any(out_root.iterdir()): raise FileExistsError("preference output must be fresh")
    out_root.mkdir(parents=True,exist_ok=True)
    splits,stats=extract_preferences(archive,repo_root=repo_root,verify_sha=verify_sha,max_context_turns=max_context_turns,max_bytes=max_bytes)
    hashes={}
    for split,rows in splits.items():
        p=out_root/f"{split}.jsonl"; p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8"); hashes[split]=sha256_file(p)
    manifest={"schema":1,"format":"neral-preference350-oasst2-human-v1","source":SOURCE,"license":LICENSE,"archive_sha256":sha256_file(archive),"verify_sha":verify_sha,"synthetic_messages_allowed":False,"external_model_calls":False,"stats":stats,"split_sha256":hashes}
    raw=json.dumps(manifest,sort_keys=True,separators=(",",":")).encode("utf8");manifest["manifest_sha256"]=hashlib.sha256(raw).hexdigest()
    (out_root/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return manifest

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--archive",required=True);ap.add_argument("--out",required=True);ap.add_argument("--repo-root",default=".");ap.add_argument("--max-context-turns",type=int,default=7);ap.add_argument("--max-bytes",type=int,default=600);a=ap.parse_args();print(json.dumps(write_preferences(Path(a.archive),Path(a.out),Path(a.repo_root),max_context_turns=a.max_context_turns,max_bytes=a.max_bytes),ensure_ascii=False))
if __name__=="__main__": main()
