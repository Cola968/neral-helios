from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

try:
    from .chat350_data import validate_dataset_root
except ImportError:
    from chat350_data import validate_dataset_root

SPLITS=("train","val","test")

def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):
            h.update(c)
    return h.hexdigest()

def convert(screened_root: Path, out: Path):
    screened_root=Path(screened_root); out=Path(out)
    manifest_path=screened_root/"manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    old=json.loads(manifest_path.read_text(encoding="utf8"))
    if old.get("version")!=1:
        raise ValueError("unsupported screened SFT manifest version")
    if old.get("external_model_calls") is not False:
        raise ValueError("screened source used external model calls")
    overlap=(old.get("prompt_group_overlap_audit") or {}).get("cross_split_prompt_group_overlap")
    if overlap not in (0,None):
        raise ValueError("screened source reports prompt leakage")
    sources=old.get("sources")
    if not isinstance(sources,list) or not sources:
        raise ValueError("screened source manifest has no sources")
    smap={}
    for s in sources:
        sid=s.get("id")
        if not sid or sid in smap:
            raise ValueError("invalid/duplicate source id")
        for k in ("license","provenance_url","sha256"):
            if not isinstance(s.get(k),str) or not s[k].strip():
                raise ValueError(f"{sid}: missing {k}")
        smap[sid]=s
    out.mkdir(parents=True,exist_ok=True)
    counts={}
    for split in SPLITS:
        src=screened_root/f"{split}.jsonl"
        if not src.is_file():
            raise FileNotFoundError(src)
        rows=[]; seen=set()
        for ln,line in enumerate(src.read_text(encoding="utf8").splitlines(),1):
            if not line.strip():
                continue
            r=json.loads(line); sid=r.get("source_id")
            if sid not in smap:
                raise ValueError(f"{split}:{ln}: unknown source_id {sid!r}")
            if not all(isinstance(r.get(k),str) and r[k].strip() for k in ("id","language","domain","prompt","answer")):
                raise ValueError(f"{split}:{ln}: malformed screened row")
            rid=f"{sid}:{r['id']}"
            if rid in seen:
                raise ValueError(f"duplicate converted id in {split}: {rid}")
            seen.add(rid)
            s=smap[sid]
            rows.append({
                "id":rid,"language":r["language"],"domain":r["domain"],
                "source":sid,"license":s["license"],
                "prompt":r["prompt"],"answer":r["answer"],
            })
        if not rows:
            raise ValueError(f"empty converted split: {split}")
        p=out/f"{split}.jsonl"
        p.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf8")
        counts[split]={"rows":len(rows),"sha256":sha256_file(p)}
    new_manifest,_=validate_dataset_root(out)
    provenance={
        "schema":1,"format":"neraltalk350-from-screened-sft-v1",
        "source_screened_manifest_sha256":sha256_file(manifest_path),
        "source_split_sha256":{s:sha256_file(screened_root/f"{s}.jsonl") for s in SPLITS},
        "output":counts,"neraltalk_manifest":new_manifest,
        "source_licenses":{sid:{"license":s["license"],"provenance_url":s["provenance_url"],"source_file_sha256":s["sha256"]} for sid,s in sorted(smap.items())},
        "holdout_imported":False,
        "external_model_calls":False,
        "policy":"Only already screened Neral SFT train/val/test rows are adapted; historical holdout and frozen benchmark data are never imported."
    }
    (out/"provenance.json").write_text(json.dumps(provenance,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    return provenance

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--screened-root",required=True)
    ap.add_argument("--out",required=True)
    a=ap.parse_args()
    print(json.dumps(convert(Path(a.screened_root),Path(a.out)),ensure_ascii=False))

if __name__=="__main__":
    main()
