"""Bootstrap a broader local B2 corpus from pinned public datasets.

This is a convenience data-preparation utility, not training code. It uses no external model
or API key. It streams pinned Hugging Face dataset revisions, writes bounded local JSONL
source files, records exact local SHA-256 values and emits the source manifest consumed by
prepare_pretrain50_b2.py. Raw documents stay local and are ignored by Git.

The default mix deliberately prioritizes broad DE/EN language and educational text. Code and
focus-domain capability are addressed later by dedicated, provenance-safe stages rather than
pulling license-heterogeneous code into the first automatic bootstrap.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

FINEWEB_REPO="HuggingFaceFW/fineweb"
FINEWEB_REV="v1.4.0"
FINEWEB2_REPO="HuggingFaceFW/fineweb-2"
# Verified upstream v2.1.0 branch; immutable commit (v2.1.1 does not exist).
FINEWEB2_REV="66d1a3b989bad0c42b88179a7fb60c47ce87af74"
FINEWEB_EDU_REPO="HuggingFaceFW/fineweb-edu"
FINEWEB_EDU_REV="v1.4.0"
OASST_REV="5c38bf14fe3c7bb31bb8e13eb4f87b2c6fd78d93"
ODC_LICENSE="ODC-By-1.0"
ODC_LICENSE_URL="https://opendatacommons.org/licenses/by/1-0/"

SOURCE_SPECS=(
    {
        "id":"fineweb-en-general","repo":FINEWEB_REPO,"config":"sample-10BT","revision":FINEWEB_REV,
        "language":"en","domain":"general","target_fraction":.35,"staging_bytes":55_000_000,
        "license":ODC_LICENSE,"license_url":ODC_LICENSE_URL,
        "provenance_url":f"https://huggingface.co/datasets/{FINEWEB_REPO}/tree/{FINEWEB_REV}",
    },
    {
        "id":"fineweb2-de-general","repo":FINEWEB2_REPO,"config":"deu_Latn","revision":FINEWEB2_REV,
        "language":"de","domain":"general","target_fraction":.40,"staging_bytes":65_000_000,
        "license":ODC_LICENSE,"license_url":ODC_LICENSE_URL,
        "provenance_url":f"https://huggingface.co/datasets/{FINEWEB2_REPO}/tree/{FINEWEB2_REV}",
    },
    {
        "id":"fineweb-edu-en","repo":FINEWEB_EDU_REPO,"config":"sample-10BT","revision":FINEWEB_EDU_REV,
        "language":"en","domain":"education","target_fraction":.20,"staging_bytes":35_000_000,
        "license":ODC_LICENSE,"license_url":ODC_LICENSE_URL,
        "provenance_url":f"https://huggingface.co/datasets/{FINEWEB_EDU_REPO}/tree/{FINEWEB_EDU_REV}",
    },
)
DIALOGUE_FRACTION=.05
DIALOGUE_STAGING_BYTES=12_000_000


def sha256_file(path: Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def safe_id(value,index):
    value=str(value or "").strip()
    if value and len(value)<=160 and re.fullmatch(r"[A-Za-z0-9_.:/@+-]+",value):return value
    return f"row-{index:09d}"


def row_text(row):
    value=row.get("text")
    return value if isinstance(value,str) else None


def row_url(row):
    value=row.get("url")
    return value if isinstance(value,str) and value.startswith(("http://","https://")) else None


def stream_public_source(spec,out_path: Path,staging_bytes=None):
    target=int(staging_bytes or spec["staging_bytes"])
    receipt=out_path.with_suffix('.receipt.json')
    identity={k:spec[k] for k in ('repo','config','revision')}
    if out_path.is_file() and receipt.is_file():
        saved=json.loads(receipt.read_text(encoding='utf8'))
        if (saved.get('identity')==identity and saved.get('target_bytes')==target
                and saved.get('sha256')==sha256_file(out_path)):
            return saved
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise RuntimeError("Install the optional data dependency: pip install -r requirements-data.txt") from e
    ds=load_dataset(spec["repo"],spec["config"],split="train",streaming=True,revision=spec["revision"])
    out_path.parent.mkdir(parents=True,exist_ok=True);written=0;docs=0;seen=set();reject=collections.Counter()
    partial=out_path.with_suffix('.partial.jsonl')
    with partial.open("w",encoding="utf8",newline="\n") as f:
        for index,row in enumerate(ds):
            text=row_text(row)
            if not text:reject["missing_text"]+=1;continue
            raw=text.encode("utf8",errors="strict")
            if len(raw)<400:reject["short"]+=1;continue
            digest=hashlib.sha256(raw).hexdigest()
            if digest in seen:reject["duplicate"]+=1;continue
            seen.add(digest)
            rid=safe_id(row.get("id") or row.get("file_path") or row.get("url"),index)
            item={"id":rid,"text":text,"url":row_url(row),"source_revision":spec["revision"]}
            f.write(json.dumps(item,ensure_ascii=False,separators=(",",":"))+"\n");written+=len(raw);docs+=1
            if written>=target:break
    if written<target:raise ValueError(f"{spec['id']}: stream ended at {written:,} UTF-8 bytes before target {target:,}")
    partial.replace(out_path)
    result={"documents":docs,"text_bytes":written,"reject":dict(reject),"sha256":sha256_file(out_path),
            "identity":identity,"target_bytes":target}
    tmp=receipt.with_suffix('.tmp')
    tmp.write_text(json.dumps(result,indent=2),encoding='utf8');tmp.replace(receipt)
    return result


def build_oasst_dialogue(out_path: Path,staging_bytes=DIALOGUE_STAGING_BYTES):
    from fetch_oasst1_sft50 import UPSTREAM_REVISION, choose_pairs, download, sha256_file as source_sha, UPSTREAM_SHA256, UPSTREAM_URL
    import gzip
    cache=out_path.parent/"oasst1_ready.messages.jsonl.gz"
    if not cache.is_file():download(UPSTREAM_URL,cache)
    actual=source_sha(cache)
    if actual!=UPSTREAM_SHA256:raise ValueError(f"Pinned OASST1 SHA mismatch: {actual}")
    rows=[]
    with gzip.open(cache,"rt",encoding="utf8") as f:
        for line in f:
            if line.strip():rows.append(json.loads(line))
    pairs=choose_pairs(rows,10**9,1);flat=[]
    for row in pairs["en"]:flat.append((hashlib.sha256(row["id"].encode()).hexdigest(),row))
    flat.sort();written=0;docs=0
    with out_path.open("w",encoding="utf8",newline="\n") as f:
        for _,row in flat:
            text=f"User: {row['prompt'].strip()}\nAssistant: {row['answer'].strip()}"
            raw=text.encode("utf8")
            if len(raw)<300:continue
            item={"id":row["id"],"text":text,"url":f"https://huggingface.co/datasets/OpenAssistant/oasst1/tree/{UPSTREAM_REVISION}"}
            f.write(json.dumps(item,ensure_ascii=False,separators=(",",":"))+"\n");written+=len(raw);docs+=1
            if written>=staging_bytes:break
    if written<min(staging_bytes,5_000_000):raise ValueError(f"OASST1 English dialogue bootstrap too small: {written:,} bytes")
    return {"documents":docs,"text_bytes":written,"sha256":sha256_file(out_path),"upstream_sha256":actual}


def make_manifest(root: Path,source_reports):
    sources=[]
    for spec in SOURCE_SPECS:
        path=root/f"{spec['id']}.jsonl";report=source_reports[spec["id"]]
        sources.append({
            "id":spec["id"],"file":path.name,"sha256":report["sha256"],"language":spec["language"],"domain":spec["domain"],
            "target_fraction":spec["target_fraction"],"license":spec["license"],"license_url":spec["license_url"],"provenance_url":spec["provenance_url"],
            "dataset_revision":spec["revision"],"staged_text_bytes":report["text_bytes"],"staged_documents":report["documents"],
            "terms_note":"Dataset use is also subject to Common Crawl Terms of Use; preserve provenance/attribution in release documentation.",
        })
    dialogue=source_reports["oasst1-dialogue-en"]
    sources.append({
        "id":"oasst1-dialogue-en","file":"oasst1-dialogue-en.jsonl","sha256":dialogue["sha256"],"language":"en","domain":"dialogue","target_fraction":DIALOGUE_FRACTION,
        "license":"Apache-2.0","license_url":f"https://huggingface.co/datasets/OpenAssistant/oasst1/blob/{OASST_REV}/LICENSE",
        "provenance_url":f"https://huggingface.co/datasets/OpenAssistant/oasst1/tree/{OASST_REV}","dataset_revision":OASST_REV,
        "staged_text_bytes":dialogue["text_bytes"],"staged_documents":dialogue["documents"],"upstream_sha256":dialogue["upstream_sha256"],
    })
    if not math.isclose(sum(s["target_fraction"] for s in sources),1.0,abs_tol=1e-9):raise AssertionError("bootstrap fractions must sum to 1")
    manifest={
        "version":1,"bootstrap_version":2,"purpose":"Neral-50M B2 automatic broader-data bootstrap",
        "sources":sources,
        "notes":[
            "Raw staged data remains local and is not committed.",
            "The downstream B2 builder independently verifies hashes, filters text, deduplicates, splits documents and screens frozen benchmark literals.",
            "FineWeb/FineWeb2/FineWeb-Edu are ODC-By datasets derived from Common Crawl; their dataset cards state that Common Crawl Terms of Use also apply.",
            "Dialogue source contains English only so manifest language metadata remains exact; German conversational behavior is supplied by Stage-D SFT.",
        ],
    }
    (root/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8")
    return manifest


def bootstrap(root: Path,scale=1.0):
    if not .05<=scale<=4.0:raise ValueError("scale must be between 0.05 and 4.0")
    root.mkdir(parents=True,exist_ok=True);reports={}
    for spec in SOURCE_SPECS:
        path=root/f"{spec['id']}.jsonl";reports[spec["id"]]=stream_public_source(spec,path,round(spec["staging_bytes"]*scale))
        print(json.dumps({"source":spec["id"],**reports[spec["id"]]},ensure_ascii=False),flush=True)
    reports["oasst1-dialogue-en"]=build_oasst_dialogue(root/"oasst1-dialogue-en.jsonl",round(DIALOGUE_STAGING_BYTES*scale))
    manifest=make_manifest(root,reports)
    print(json.dumps({"manifest":str(root/"manifest.json"),"sources":len(manifest["sources"]),"staged_text_bytes":sum(r["text_bytes"] for r in reports.values())},ensure_ascii=False))
    return manifest


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default="b2_sources");ap.add_argument("--scale",type=float,default=1.0);args=ap.parse_args()
    bootstrap(Path(args.out),args.scale)


if __name__=="__main__":main()
