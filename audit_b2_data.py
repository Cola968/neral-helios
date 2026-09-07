"""Fail-closed audit for a prepared Neral-50M B2 corpus.

The audit never reads raw source text and never calls a model. It verifies the generated
manifest, packed-file hashes, document index integrity, exact normalized-dedup evidence,
source quotas, provenance fields, and hashes of every frozen benchmark source used by the
contamination screen. It is intended to run immediately before B2 training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from prepare_pretrain50_b2 import benchmark_contamination, sha256_file

SHA256_RE=re.compile(r"^[0-9a-f]{64}$",re.I)
SPLITS=("train","val","test","holdout")
LANGUAGES=("de","en")


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf8"))


def audit(root: Path, repo_root: Path):
    manifest_path=root/"manifest.json"
    docs_path=root/"documents.jsonl"
    if not manifest_path.is_file():raise FileNotFoundError(manifest_path)
    if not docs_path.is_file():raise FileNotFoundError(docs_path)
    m=_load_json(manifest_path)
    if m.get("version")!=1:raise ValueError("B2 corpus manifest version must be 1")
    target=int(m.get("target_train_tokens",0));actual=int(m.get("actual_train_tokens",0))
    if target<1_000_000 or actual<target:raise ValueError(f"B2 train-token target not met: {actual} < {target}")

    sources=m.get("sources")
    if not isinstance(sources,list) or not sources:raise ValueError("B2 manifest has no sources")
    fractions=0.0;source_ids=set()
    for src in sources:
        sid=src.get("id")
        if not isinstance(sid,str) or not sid or sid in source_ids:raise ValueError(f"Invalid/duplicate source id: {sid!r}")
        source_ids.add(sid);fractions+=float(src.get("target_fraction",0))
        for key in ("sha256","license","license_url","provenance_url"):
            if not isinstance(src.get(key),str) or not src[key].strip():raise ValueError(f"{sid}: missing provenance field {key}")
        if not SHA256_RE.fullmatch(src["sha256"]):raise ValueError(f"{sid}: invalid source SHA-256")
        if not src["license_url"].startswith("https://") or not src["provenance_url"].startswith("https://"):
            raise ValueError(f"{sid}: provenance URLs must be HTTPS")
    if not math.isclose(fractions,1.0,abs_tol=1e-6):raise ValueError(f"source fractions sum to {fractions}, expected 1.0")

    selection=m.get("selection",{})
    if set(selection)!=source_ids:raise ValueError("selection/source id mismatch")
    for sid,meta in selection.items():
        if int(meta.get("selected_tokens",0))<int(meta.get("target_tokens",0)):
            raise ValueError(f"{sid}: source quota not met")
        if int(meta.get("selected_documents",0))<=0:raise ValueError(f"{sid}: selected no training documents")

    splits=m.get("splits",{})
    packed_token_sum=0
    for split in SPLITS:
        if split not in splits:raise ValueError(f"missing split {split}")
        for lang in LANGUAGES:
            meta=splits[split].get(lang)
            if not isinstance(meta,dict):raise ValueError(f"missing {split}/{lang} metadata")
            path=root/f"{split}_{lang}.bin"
            if not path.is_file():raise FileNotFoundError(path)
            expected=meta.get("sha256")
            if not isinstance(expected,str) or not SHA256_RE.fullmatch(expected):raise ValueError(f"bad packed hash for {split}/{lang}")
            if sha256_file(path).lower()!=expected.lower():raise ValueError(f"packed SHA-256 mismatch for {split}/{lang}")
            size_tokens=path.stat().st_size//2
            if path.stat().st_size%2 or size_tokens!=int(meta.get("tokens",-1)):
                raise ValueError(f"packed token count mismatch for {split}/{lang}")
            if split=="train":packed_token_sum+=size_tokens
    if packed_token_sum!=actual:raise ValueError(f"actual_train_tokens mismatch: manifest={actual}, packed={packed_token_sum}")

    expected_docs=m.get("documents_sha256")
    if not isinstance(expected_docs,str) or not SHA256_RE.fullmatch(expected_docs):raise ValueError("invalid documents SHA-256")
    if sha256_file(docs_path).lower()!=expected_docs.lower():raise ValueError("documents.jsonl SHA-256 mismatch")

    seen_doc_keys=set();seen_norm=set();last_end=defaultdict(int);doc_counts=defaultdict(int)
    for line_no,line in enumerate(docs_path.read_text(encoding="utf8").splitlines(),1):
        if not line.strip():continue
        row=json.loads(line);key=(row.get("source_id"),str(row.get("id")))
        if key in seen_doc_keys:raise ValueError(f"duplicate document key at line {line_no}: {key}")
        seen_doc_keys.add(key)
        digest=row.get("normalized_sha256")
        if not isinstance(digest,str) or not SHA256_RE.fullmatch(digest):raise ValueError(f"invalid normalized hash at line {line_no}")
        if digest in seen_norm:raise ValueError(f"exact normalized duplicate survived at line {line_no}")
        seen_norm.add(digest)
        split=row.get("split");lang=row.get("language")
        if split not in SPLITS or lang not in LANGUAGES:raise ValueError(f"invalid split/language at line {line_no}")
        start=int(row.get("start",-1));end=int(row.get("end",-1));bucket=(split,lang)
        if start!=last_end[bucket] or end<=start:raise ValueError(f"non-contiguous/invalid document range at line {line_no}")
        if end>int(splits[split][lang]["tokens"]):raise ValueError(f"document exceeds packed stream at line {line_no}")
        last_end[bucket]=end;doc_counts[bucket]+=1
    for split in SPLITS:
        for lang in LANGUAGES:
            meta=splits[split][lang]
            if last_end[(split,lang)]!=int(meta["tokens"]):raise ValueError(f"document index does not cover {split}/{lang}")
            if doc_counts[(split,lang)]!=int(meta["documents"]):raise ValueError(f"document count mismatch for {split}/{lang}")

    _,current_bench_hashes=benchmark_contamination(repo_root)
    recorded=m.get("contamination_source_sha256")
    if recorded!=current_bench_hashes:
        raise ValueError("frozen benchmark sources changed after B2 contamination scan; rebuild corpus before training")

    report={
        "ok":True,"target_train_tokens":target,"actual_train_tokens":actual,
        "sources":len(sources),"documents":len(seen_doc_keys),"unique_normalized_documents":len(seen_norm),
        "source_manifest_sha256":m.get("source_manifest_sha256"),"documents_sha256":expected_docs,
        "contamination_source_sha256":recorded,
    }
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--data",default="pretrain50_b2_data");ap.add_argument("--out")
    a=ap.parse_args();report=audit(Path(a.data),Path(__file__).resolve().parent)
    text=json.dumps(report,indent=2,ensure_ascii=False)
    if a.out:Path(a.out).write_text(text+"\n",encoding="utf8")
    print(text)


if __name__=="__main__":main()
