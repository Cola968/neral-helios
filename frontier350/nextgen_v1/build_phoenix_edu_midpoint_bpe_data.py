from __future__ import annotations

"""Build the midpoint education/general/DE Phoenix DEVELOPMENT refresh corpus.

Controlled source-mix line search between the measured 55/15/30 leader and the
75/10/15 education-heavy ablation: 65% FineWeb-Edu EN / 12.5% FineWeb EN /
22.5% FineWeb-2 DE. Same admitted source families, immutable revisions, tokenizer,
dedupe and development-only manifest contract. No FineMath and no synthetic data.
"""

import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

from build_frontier_data import SourceSpec,assert_sources_allowed_by_policy
from build_frontier_bpe_data import LocalBPE,PersistentSeen,StreamingShardWriter,build_source
from source_lock350 import LOCK_PATH,locked_revisions

CURRENT_ANCHOR_TOKENS=2_073_161_216
CURRENT_ANCHOR_SHA256="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
EXPECTED_TOKENIZER_SHA256="4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
MIX_PROFILE="edu-midpoint-65-12p5-22p5-v1"
SOURCES=(
    SourceSpec("fineweb_edu_en","HuggingFaceFW/fineweb-edu","sample-10BT","train","odc-by","text",0.65,"en","education"),
    SourceSpec("fineweb_en","HuggingFaceFW/fineweb","sample-10BT","train","odc-by","text",0.125,"en","general"),
    SourceSpec("fineweb2_de","HuggingFaceFW/fineweb-2","deu_Latn","train","odc-by","text",0.225,"de","general"),
)

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def quota(total:int,index:int)->int:
    if index==len(SOURCES)-1:return total-sum(int(total*s.share) for s in SOURCES[:-1])
    return int(total*SOURCES[index].share)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True)
    p.add_argument("--train-tokens",type=int,default=20_000_000)
    p.add_argument("--valid-tokens",type=int,default=500_000);p.add_argument("--shard-tokens",type=int,default=5_000_000)
    a=p.parse_args(argv)
    if a.train_tokens!=20_000_000 or a.valid_tokens!=500_000:
        p.error("controlled midpoint experiment is locked to 20M train / 0.5M validation corpus")
    if abs(sum(s.share for s in SOURCES)-1.0)>1e-12:raise RuntimeError("shares must sum to one")
    source_policy=assert_sources_allowed_by_policy(SOURCES)
    root=a.root.absolute()
    if root.exists():raise RuntimeError("output must be create-only")
    tokenizer=LocalBPE(a.tokenizer.resolve(strict=True));contract=tokenizer.contract()
    if contract["tokenizer_sha256"]!=EXPECTED_TOKENIZER_SHA256:raise ValueError("tokenizer SHA mismatch")
    revisions=locked_revisions(SOURCES);source_lock_sha=sha256(LOCK_PATH)

    root.mkdir(parents=True,exist_ok=False)
    tw=StreamingShardWriter(root/"train","train",a.shard_tokens)
    vw=StreamingShardWriter(root/"valid","valid",min(a.shard_tokens,a.valid_tokens))
    seen=PersistentSeen(root/"dedupe.sqlite3");reports=[]
    try:
        for i,spec in enumerate(SOURCES):
            reports.append({"partition":"validation",**build_source(spec,revisions[spec.name],quota(a.valid_tokens,i),vw,seen,tokenizer)})
        for i,spec in enumerate(SOURCES):
            reports.append({"partition":"train",**build_source(spec,revisions[spec.name],quota(a.train_tokens,i),tw,seen,tokenizer)})
        documents=seen.count()
    finally:seen.close()
    valid_files=vw.finish();train_files=tw.finish()
    actual_train=sum(x["tokens"] for x in train_files);actual_valid=sum(x["tokens"] for x in valid_files)
    if actual_train!=a.train_tokens or actual_valid!=a.valid_tokens:raise RuntimeError("token accounting mismatch")

    manifest={
        "schema":1,"architecture":"frontier350-v1","corpus_role":"foundation-pretraining-refresh-development",
        "classification":"DEVELOPMENT_ONLY_NOT_PRODUCTION_APPROVED",
        "continuation_profile":"phoenix-quality-refresh-v2-approved-sources",
        "development_mix_profile":MIX_PROFILE,
        "continuation_from_tokens":CURRENT_ANCHOR_TOKENS,
        "continuation_from_checkpoint_sha256":CURRENT_ANCHOR_SHA256,
        "new_train_tokens":actual_train,"validation_tokens":actual_valid,
        "target_cumulative_tokens_if_fully_consumed":CURRENT_ANCHOR_TOKENS+actual_train,
        "tokenizer":contract,"source_lock_sha256":source_lock_sha,
        "source_mix":{s.name:s.share for s in SOURCES},"source_policy":source_policy,
        "finemath_candidate_included":False,"resolved_revisions":revisions,
        "licenses":sorted({s.license for s in SOURCES}),"sources":reports,
        "train_files":train_files,"valid_files":valid_files,"deduplicated_documents":documents,
        "validation_built_first_shared_dedupe":True,"train_validation_exact_document_disjoint":True,
        "benchmark_content_included":None,
        "benchmark_screening_status":"NOT_PERFORMED_BY_THIS_BUILDER_REQUIRES_SEPARATE_AUDIT",
        "historical_2b_document_overlap_checked":False,
        "historical_overlap_warning":"Complete historical document-hash registry unavailable; no novelty claim versus 2.073B anchor.",
        "independent_holdout_checked":False,"human_source_review_complete":False,"production_eligible":False,
        "created_unix":time.time(),
    }
    out=root/"manifest.json";out.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"event":"phoenix_edu_midpoint_corpus_built","manifest_sha256":sha256(out),
                      "mix_profile":MIX_PROFILE,"source_mix":manifest["source_mix"],
                      "train_tokens":actual_train,"validation_tokens":actual_valid,
                      "documents":documents,"production_eligible":False},sort_keys=True),flush=True)

if __name__=="__main__":
    main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
