from __future__ import annotations

"""Build a bounded FineMath DEVELOPMENT-only Phoenix refresh corpus.

Recipe: 55% FineWeb-Edu EN / 10% FineWeb EN / 15% FineWeb-2 DE / 20% FineMath.
FineMath stays blocked from ordinary training. This builder requires a clean >=10k-row
candidate-admission result and filters every quality-passing document against frozen local
benchmark/public-development literals before tokenization. The screen is deliberately
non-exhaustive and is never a production-contamination claim.
"""

import argparse,hashlib,json,os,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))

from build_frontier_data import SourceSpec,assert_sources_allowed_by_policy,content_hash,extract_text,normalize,quality_ok,stream_rows
from build_frontier_bpe_data import LocalBPE,PersistentSeen,StreamingShardWriter
from source_lock350 import LOCK_PATH,locked_revisions
from hf_source_policy50 import assert_bounded_development_candidate
from prepare_pretrain50_b2 import benchmark_contamination,normalize_for_dedup
from phoenix_knowledge_repair_sft import public_prompts

CURRENT_ANCHOR_TOKENS=2_073_161_216
CURRENT_ANCHOR_SHA256="f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7"
TOKENIZER_SHA="4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
PROFILE="phoenix-quality-refresh-v3-finemath-development"
SCREEN_STATUS="FULL_STREAM_REJECTION_FROZEN_SUITES_AND_PUBLIC_DEV_V1_NONEXHAUSTIVE"
SOURCES=(
    SourceSpec("fineweb_edu_en","HuggingFaceFW/fineweb-edu","sample-10BT","train","odc-by","text",0.55,"en","education"),
    SourceSpec("fineweb_en","HuggingFaceFW/fineweb","sample-10BT","train","odc-by","text",0.10,"en","general"),
    SourceSpec("fineweb2_de","HuggingFaceFW/fineweb-2","deu_Latn","train","odc-by","text",0.15,"de","general"),
    SourceSpec("finemath","HuggingFaceTB/finemath","finemath-3plus","train","odc-by","text",0.20,"en","math"),
)

def sha256(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk)
    return h.hexdigest()

def quota(total:int,index:int)->int:
    if index==len(SOURCES)-1:return total-sum(int(total*s.share) for s in SOURCES[:-1])
    return int(total*SOURCES[index].share)

def shingles(text:str)->set[str]:
    words=text.split()
    return {" ".join(words[i:i+5]) for i in range(max(0,len(words)-4))}

def screening_contract(repo_root:Path):
    frozen,hashes=benchmark_contamination(repo_root)
    public=tuple(normalize_for_dedup(x) for x in public_prompts())
    literals=sorted({x for x in (*frozen,*public) if len(x)>=20},key=len,reverse=True)
    near=[shingles(x) for x in literals if len(x.split())>=8]
    manifest_sha=hashlib.sha256(json.dumps(
        {"frozen_files":hashes,"public_prompts":sorted(public)},
        sort_keys=True,separators=(",",":")
    ).encode()).hexdigest()
    return literals,near,manifest_sha,len(public)

def leakage_reason(text:str,literals:list[str],near_sets:list[set[str]])->str|None:
    norm=normalize_for_dedup(text)
    if any(p in norm for p in literals):return "frozen_or_public_exact"
    doc=shingles(norm)
    if doc:
        for ps in near_sets:
            if ps and len(ps&doc)/len(ps)>=0.80:return "frozen_or_public_near"
    return None

def build_source_screened(spec,revision,target,writer,seen,tokenizer,literals,near_sets,screen_counts):
    raw=accepted=rejected=emitted=truncated=0;reasons={};started=time.time()
    for row in stream_rows(spec,revision):
        raw+=1
        text=normalize(extract_text(spec,row))
        ok,reason=quality_ok(text)
        if not ok:
            rejected+=1;reasons[reason]=reasons.get(reason,0)+1;continue
        screen_counts["documents_screened"]+=1
        leak=leakage_reason(text,literals,near_sets)
        if leak:
            rejected+=1;reasons[leak]=reasons.get(leak,0)+1
            if leak.endswith("exact"):screen_counts["exact_rejections"]+=1
            else:screen_counts["near_rejections"]+=1
            continue
        if not seen.add_if_new(content_hash(text)):
            rejected+=1;reasons["duplicate"]=reasons.get("duplicate",0)+1;continue
        ids=tokenizer.encode_document(text);remaining=target-emitted
        if len(ids)>remaining:
            ids=ids[:remaining];truncated+=1
        if ids:
            writer.add(ids);emitted+=len(ids);accepted+=1
        if emitted>=target:break
        if raw%10_000==0:
            print(json.dumps({"source":spec.name,"rows":raw,"accepted":accepted,"tokens":emitted,"target":target,
                              "screened":screen_counts["documents_screened"]},sort_keys=True),flush=True)
    if emitted!=target:raise RuntimeError(f"{spec.name} token quota mismatch {emitted}/{target}")
    return {"source":{"name":spec.name,"repo":spec.repo,"config":spec.config,"split":spec.split,"license":spec.license,
                       "text_field":spec.text_field,"share":spec.share,"language":spec.language,"kind":spec.kind},
            "revision":revision,"raw_rows":raw,"accepted_rows":accepted,"rejected_rows":rejected,
            "rejection_reasons":reasons,"emitted_tokens":emitted,"budget_truncated_documents":truncated,
            "synthetic_eos_targets":0,"elapsed_s":time.time()-started}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,required=True);p.add_argument("--tokenizer",type=Path,required=True)
    p.add_argument("--admission-result",type=Path,required=True)
    p.add_argument("--train-tokens",type=int,default=20_000_000);p.add_argument("--valid-tokens",type=int,default=500_000)
    p.add_argument("--shard-tokens",type=int,default=5_000_000)
    a=p.parse_args(argv)
    if a.train_tokens!=20_000_000 or a.valid_tokens!=500_000:p.error("locked to 20M train / 0.5M validation")
    if abs(sum(s.share for s in SOURCES)-1.0)>1e-12:raise RuntimeError("source shares must sum to one")
    if not a.admission_result.is_file() or a.admission_result.is_symlink():raise ValueError("regular FineMath admission result required")
    admission=json.loads(a.admission_result.read_text(encoding="utf-8"))
    fm_policy=assert_bounded_development_candidate("HuggingFaceTB/finemath",admission)

    ordinary=SOURCES[:3];ordinary_policy=assert_sources_allowed_by_policy(ordinary)
    source_policy=dict(ordinary_policy)
    source_policy["finemath"]={"dataset_id":fm_policy.dataset_id,"role":fm_policy.role,"license":fm_policy.license,
                               "content_origin":fm_policy.content_origin,"development_only":True}

    root=a.root.absolute()
    if root.exists():raise RuntimeError("output must be create-only")
    tokenizer=LocalBPE(a.tokenizer.resolve(strict=True))
    if tokenizer.contract()["tokenizer_sha256"]!=TOKENIZER_SHA:raise ValueError("tokenizer SHA mismatch")
    revisions=locked_revisions(SOURCES)
    if admission.get("revision")!=revisions["finemath"]:raise ValueError("FineMath admission/source-lock revision mismatch")
    lock_sha=sha256(LOCK_PATH);admission_sha=sha256(a.admission_result)
    literals,near_sets,screen_manifest_sha,public_count=screening_contract(Path(__file__).resolve().parents[2])

    root.mkdir(parents=True,exist_ok=False);train_root,valid_root=root/"train",root/"valid"
    tw=StreamingShardWriter(train_root,"train",a.shard_tokens);vw=StreamingShardWriter(valid_root,"valid",min(a.shard_tokens,a.valid_tokens))
    seen=PersistentSeen(root/"dedupe.sqlite3");reports=[]
    counts={"documents_screened":0,"exact_rejections":0,"near_rejections":0}
    try:
        for i,spec in enumerate(SOURCES):
            reports.append({"partition":"validation",**build_source_screened(spec,revisions[spec.name],quota(a.valid_tokens,i),
                            vw,seen,tokenizer,literals,near_sets,counts)})
        for i,spec in enumerate(SOURCES):
            reports.append({"partition":"train",**build_source_screened(spec,revisions[spec.name],quota(a.train_tokens,i),
                            tw,seen,tokenizer,literals,near_sets,counts)})
        documents=seen.count()
    finally:seen.close()
    valid_files=vw.finish();train_files=tw.finish()
    actual_train=sum(x["tokens"] for x in train_files);actual_valid=sum(x["tokens"] for x in valid_files)
    if actual_train!=a.train_tokens or actual_valid!=a.valid_tokens:raise RuntimeError("token accounting mismatch")

    manifest={
        "schema":1,"architecture":"frontier350-v1","corpus_role":"foundation-pretraining-refresh-development",
        "classification":"DEVELOPMENT_ONLY_NOT_PRODUCTION_APPROVED","continuation_profile":PROFILE,
        "continuation_from_tokens":CURRENT_ANCHOR_TOKENS,"continuation_from_checkpoint_sha256":CURRENT_ANCHOR_SHA256,
        "new_train_tokens":actual_train,"validation_tokens":actual_valid,
        "target_cumulative_tokens_if_fully_consumed":CURRENT_ANCHOR_TOKENS+actual_train,
        "tokenizer":tokenizer.contract(),"source_lock_sha256":lock_sha,
        "source_mix":{s.name:s.share for s in SOURCES},"source_policy":source_policy,
        "finemath_candidate_included":True,"finemath_candidate_admission":admission,
        "finemath_candidate_admission_sha256":admission_sha,
        "resolved_revisions":revisions,"licenses":sorted({s.license for s in SOURCES}),"sources":reports,
        "train_files":train_files,"valid_files":valid_files,"deduplicated_documents":documents,
        "validation_built_first_shared_dedupe":True,"train_validation_exact_document_disjoint":True,
        "benchmark_content_included":None,"benchmark_screening_status":SCREEN_STATUS,
        "full_stream_leakage_screen":{**counts,"screen_contract_sha256":screen_manifest_sha,
                                      "public_dev_prompts":public_count,
                                      "interpretation":"Exact/near rejection against committed frozen-suite/public-dev literals only; non-exhaustive."},
        "historical_2b_document_overlap_checked":False,
        "historical_overlap_warning":"Complete historical document-hash registry unavailable; no novelty claim versus 2.073B anchor.",
        "independent_holdout_checked":False,"human_source_review_complete":False,"production_eligible":False,
        "created_unix":time.time(),
    }
    path=root/"manifest.json";path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps({"event":"phoenix_finemath_dev_corpus_built","manifest_sha256":sha256(path),
                      "source_mix":manifest["source_mix"],"train_tokens":actual_train,"validation_tokens":actual_valid,
                      "documents":documents,"screen":manifest["full_stream_leakage_screen"],
                      "production_eligible":False},sort_keys=True),flush=True)

if __name__=="__main__":
    main();sys.stdout.flush();sys.stderr.flush();os._exit(0)
