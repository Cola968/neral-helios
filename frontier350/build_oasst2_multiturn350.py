from __future__ import annotations
import argparse, json, os, tempfile
from pathlib import Path
try:
    from .chat350_data import validate_dataset_root, read_jsonl
    from .oasst2_human350 import load_archive,indexes,path_to_root,valid_human_path,to_messages,split_for_root_prompt,contaminated,sha256_file,LICENSE,SOURCE
    from .chat350_text_hygiene import all_texts_safe
except ImportError:
    from chat350_data import validate_dataset_root, read_jsonl
    from oasst2_human350 import load_archive,indexes,path_to_root,valid_human_path,to_messages,split_for_root_prompt,contaminated,sha256_file,LICENSE,SOURCE
    from chat350_text_hygiene import all_texts_safe

def _frozen(repo_root: Path) -> list[str]:
    import sys
    rr=str(Path(repo_root).resolve())
    if rr not in sys.path: sys.path.insert(0,rr)
    import prepare_sft50
    phrases,_=prepare_sft50.benchmark_literals(Path(repo_root))
    try:
        import prepare_sft50_i1
        extra,_=prepare_sft50_i1.targeted_literals(Path(repo_root))
        phrases=sorted(set(phrases)|set(extra),key=len,reverse=True)
    except Exception:
        pass
    return phrases

def extract(archive: Path, *, repo_root: Path, verify_sha: bool=True, max_turns: int=8, max_bytes: int=360):
    if max_turns<4 or max_turns%2: raise ValueError("max_turns must be even and >=4")
    rows=load_archive(archive,verify_sha=verify_sha); by_id,children=indexes(rows); frozen=_frozen(repo_root)
    by_root={}
    for row in rows:
        if row.get("role")!="assistant": continue
        path=path_to_root(row["message_id"],by_id)
        if len(path)<4 or not valid_human_path(path,end_role="assistant"): continue
        root=path[0]["message_id"]
        use=path[-max_turns:] if len(path)>max_turns else path
        if use[0].get("role")!="prompter" or use[-1].get("role")!="assistant": continue
        texts=[r["text"].strip() for r in use]
        safe,_=all_texts_safe(texts)
        if not safe: continue
        if sum(len(x.encode("utf8")) for x in texts)>max_bytes: continue
        if contaminated(texts,frozen): continue
        rank_sum=sum(int(r["rank"]) if isinstance(r.get("rank"),int) else 999 for r in use if r.get("role")=="assistant")
        key=(-len(use),rank_sum,row["message_id"])
        prev=by_root.get(root)
        if prev is None or key<prev[0]: by_root[root]=(key,use)
    out={"train":[],"val":[],"test":[]}; rejected_holdout=0
    for root,(_,path) in sorted(by_root.items()):
        split=split_for_root_prompt(path_to_root(root,by_id)[0]["text"])
        if split=="holdout": rejected_holdout+=1; continue
        m=to_messages(path)
        out[split].append({"id":f"oasst2-multiturn:{root}:{path[-1]['message_id']}","language":path[0]["lang"],"domain":"chat","source":"oasst2-human-multiturn","license":LICENSE,"messages":m})
    return out,{"candidate_roots":len(by_root),"holdout_skipped":rejected_holdout,"accepted":{k:len(v) for k,v in out.items()}}

def _message_key(text: str, *, min_normalized_chars: int=20):
    import hashlib,unicodedata
    value=" ".join(unicodedata.normalize("NFKC",str(text)).casefold().split())
    if len(value)<min_normalized_chars:
        return None
    return hashlib.sha256(value.encode("utf8")).hexdigest()

def filter_cross_split_message_leakage(base, additions, *, min_normalized_chars: int=20):
    locations={}
    for split,rows in base.items():
        for row in rows:
            for msg in row.get("messages",[]):
                key=_message_key(msg.get("content",""),min_normalized_chars=min_normalized_chars)
                if key:
                    locations.setdefault(key,set()).add(split)
    kept={k:[] for k in ("train","val","test")}
    filtered=0
    for split in ("train","val","test"):
        for row in additions.get(split,[]):
            keys=[_message_key(m.get("content",""),min_normalized_chars=min_normalized_chars) for m in row.get("messages",[])]
            keys=[k for k in keys if k]
            conflict=any(locations.get(k,set())-{split} for k in keys)
            if conflict:
                filtered+=1
                continue
            kept[split].append(row)
            for key in keys:
                locations.setdefault(key,set()).add(split)
    return kept,{"cross_split_message_filtered":filtered,"message_key_min_normalized_chars":min_normalized_chars}

def augment(archive: Path, dataset_root: Path, repo_root: Path, *, verify_sha: bool=True, max_turns: int=8, max_bytes: int=360):
    root=Path(dataset_root); base={s:read_jsonl(root/f"{s}.jsonl") for s in ("train","val","test")}
    add,stats=extract(archive,repo_root=repo_root,verify_sha=verify_sha,max_turns=max_turns,max_bytes=max_bytes)
    add,leakage_stats=filter_cross_split_message_leakage(base,add)
    stats.update(leakage_stats)
    with tempfile.TemporaryDirectory(dir=root.parent) as td:
        tmp=Path(td)
        for split in ("train","val","test"):
            rows=base[split]+add[split]
            (tmp/f"{split}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
        manifest,_=validate_dataset_root(tmp)
        for split in ("train","val","test"): os.replace(tmp/f"{split}.jsonl",root/f"{split}.jsonl")
    prov={"schema":1,"source":SOURCE,"license":LICENSE,"archive_sha256":sha256_file(archive),"verify_sha":verify_sha,"max_turns":max_turns,"max_bytes":max_bytes,"stats":stats,"post_augmentation_manifest":manifest}
    (root/"multiturn_provenance.json").write_text(json.dumps(prov,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return prov

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--archive",required=True);ap.add_argument("--dataset-root",required=True);ap.add_argument("--repo-root",default=".");ap.add_argument("--max-turns",type=int,default=8);ap.add_argument("--max-bytes",type=int,default=360);a=ap.parse_args()
    print(json.dumps(augment(Path(a.archive),Path(a.dataset_root),Path(a.repo_root),max_turns=a.max_turns,max_bytes=a.max_bytes),ensure_ascii=False))
if __name__=="__main__": main()
