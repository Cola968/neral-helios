"""Build Neral-50M B2 corpus from locally supplied, provenance-declared JSONL sources.

The builder never downloads data and never calls another model. Every input file must be
listed with an exact SHA-256, license metadata, language, domain and target fraction. Raw
source text and packed binaries remain local and are excluded from Git.
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

import numpy as np

SPECIAL_BOS=1
SPECIAL_EOS=2
BYTE_OFFSET=4
ALLOWED_LANGUAGES={"de","en"}
ALLOWED_DOMAINS={"general","education","science_math","code","dialogue","focus","structured"}
ID_RE=re.compile(r"[a-zA-Z0-9_.-]{1,80}")
SECRET_OR_CONTACT=re.compile(
    r"[\w.+-]+@[\w.-]+\.[a-z]{2,}|(?:sk-|gh[pousr]_|github_pat_|xox[baprs]-)[A-Za-z0-9_-]{16,}|-----BEGIN .*PRIVATE KEY",
    re.I,
)
FROZEN_PY_ASSIGNMENTS={
    "benchmark_v3.py":{"PAIR_CASES","FREE_CHAT","BASE_PREFIXES"},
    "benchmark_v4.py":{"PAIRS","ENTROPY_PREFIXES","BASE_GENERATION"},
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
    return h.hexdigest()


def normalize_for_dedup(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())


def _basic_text_checks(raw):
    if not isinstance(raw,str):return None,"not_text"
    if "\ufffd" in raw or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]",raw):return None,"encoding_or_control"
    if SECRET_OR_CONTACT.search(raw):return None,"contact_or_secret_pattern"
    raw=unicodedata.normalize("NFC",raw.replace("\r\n","\n").replace("\r","\n"))
    if len(raw)>2_000_000:return None,"document_too_large"
    return raw,None


def clean_text(raw: str, domain: str="general"):
    """Domain-aware deterministic cleaning.

    Prose is aggressively paragraph-filtered. Code preserves leading indentation and line
    breaks; dialogue/structured sources preserve short turn/record lines. This avoids
    destroying exactly the syntax/turn structure those B2 domains are meant to teach.
    """
    raw,reason=_basic_text_checks(raw)
    if reason:return None,reason
    if domain not in ALLOWED_DOMAINS:return None,"unsupported_domain"

    if domain=="code":
        lines=[line.rstrip() for line in raw.split("\n")]
        # Collapse excessive blank runs while preserving indentation and ordinary blank lines.
        out=[];blank=0
        for line in lines:
            if not line.strip():
                blank+=1
                if blank<=2:out.append("")
                continue
            blank=0;out.append(line)
        text="\n".join(out).strip("\n")
        if len(text)<200:return None,"insufficient_clean_code"
        printable=sum(c.isprintable() or c in "\n\t" for c in text)/max(1,len(text))
        alpha=sum(c.isalpha() for c in text)/max(1,len(text))
        if printable<.98 or alpha<.12:return None,"low_quality_code"
        # Generated/minified garbage often contains enormous unbroken lines.
        if max((len(line) for line in text.splitlines()),default=0)>20_000:return None,"extreme_code_line"
        return text,None

    if domain in {"dialogue","structured"}:
        lines=[]
        for line in raw.split("\n"):
            line=" ".join(line.split())
            if not line:continue
            if re.search(r"javascript:|cookie settings|all rights reserved",line,re.I):continue
            if re.search(r"(.)\1{14,}",line):continue
            lines.append(line)
        text="\n".join(lines)
        if len(text)<250:return None,"insufficient_clean_structured"
        if sum(c.isalpha() for c in text)/max(1,len(text))<.35:return None,"low_alpha_structured"
        return text,None

    paragraphs=[];seen=set()
    for part in raw.splitlines():
        part=" ".join(part.split())
        if len(part)<80:continue
        if re.search(r"<[^>]+>|\{\{|javascript:|cookie settings|all rights reserved",part,re.I):continue
        if sum(c.isalpha() for c in part)/max(1,len(part))<.45:continue
        if re.search(r"(.)\1{9,}",part):continue
        key=normalize_for_dedup(part)
        if key in seen:continue
        seen.add(key);paragraphs.append(part)
    text="\n\n".join(paragraphs)
    if len(text)<300:return None,"insufficient_clean_prose"
    return text,None


def minhash_signature(normalized: str, count: int=32):
    words=normalized.split()
    if len(words)<12:return tuple()
    shingles={" ".join(words[i:i+5]) for i in range(len(words)-4)}
    hashes=sorted(int.from_bytes(hashlib.blake2b(x.encode("utf8"),digest_size=8).digest(),"big") for x in shingles)
    return tuple(hashes[:count])


def _collect_strings(value, out):
    if isinstance(value,str):out.append(value)
    elif isinstance(value,(tuple,list,set)):
        for item in value:_collect_strings(item,out)
    elif isinstance(value,dict):
        for k,v in value.items():_collect_strings(k,out);_collect_strings(v,out)


def _python_frozen_literals(path: Path, names: set[str]):
    """Extract only literal benchmark constants with ast.literal_eval; never execute code."""
    tree=ast.parse(path.read_text(encoding="utf8"),filename=str(path));strings=[]
    for node in tree.body:
        if not isinstance(node,(ast.Assign,ast.AnnAssign)):continue
        targets=node.targets if isinstance(node,ast.Assign) else [node.target]
        target_names={t.id for t in targets if isinstance(t,ast.Name)}
        if not target_names&names:continue
        try:value=ast.literal_eval(node.value)
        except (ValueError,TypeError,SyntaxError):continue
        _collect_strings(value,strings)
    return strings


def benchmark_contamination(repo_root: Path):
    """Return normalized frozen-suite literals plus exact source hashes for audit.

    This catches exact long prompt/continuation fragments from the committed free-generation,
    RawMath, v3 and v4 suites. It is intentionally not described as exhaustive semantic
    contamination detection.
    """
    phrases=set();files={}
    json_paths=[repo_root/"benchmarks50/regression.jsonl",repo_root/"benchmarks50/raw_math_v1.jsonl"]
    for path in json_paths:
        if not path.exists():continue
        files[str(path.relative_to(repo_root))]=sha256_file(path)
        for line in path.read_text(encoding="utf8").splitlines():
            if not line.strip():continue
            row=json.loads(line)
            for value in row.values():
                if isinstance(value,str):
                    n=normalize_for_dedup(value)
                    if len(n)>=20:phrases.add(n)
    for filename,names in FROZEN_PY_ASSIGNMENTS.items():
        path=repo_root/filename
        if not path.exists():continue
        files[filename]=sha256_file(path)
        for value in _python_frozen_literals(path,names):
            n=normalize_for_dedup(value)
            if len(n)>=20:phrases.add(n)
    return sorted(phrases,key=len,reverse=True),files


def benchmark_phrases(repo_root: Path):
    return benchmark_contamination(repo_root)[0]


def validate_manifest(manifest: dict, source_root: Path):
    if manifest.get("version")!=1:raise ValueError("B2 manifest version must be 1")
    sources=manifest.get("sources")
    if not isinstance(sources,list) or not sources:raise ValueError("B2 manifest needs sources")
    ids=set();fractions=0.0;normalized=[];root=source_root.resolve()
    for src in sources:
        sid=src.get("id")
        if not isinstance(sid,str) or not ID_RE.fullmatch(sid) or sid in ids:raise ValueError(f"Invalid/duplicate source id: {sid!r}")
        ids.add(sid);lang=src.get("language");domain=src.get("domain")
        if lang not in ALLOWED_LANGUAGES:raise ValueError(f"{sid}: language must be de/en")
        if domain not in ALLOWED_DOMAINS:raise ValueError(f"{sid}: unsupported domain {domain!r}")
        frac=src.get("target_fraction")
        if type(frac) not in (int,float) or not 0<float(frac)<=1:raise ValueError(f"{sid}: invalid target_fraction")
        fractions+=float(frac)
        for key in ("license","license_url","provenance_url","sha256","file"):
            if not isinstance(src.get(key),str) or not src[key].strip():raise ValueError(f"{sid}: missing {key}")
        if not re.fullmatch(r"[a-fA-F0-9]{64}",src["sha256"]):raise ValueError(f"{sid}: invalid sha256")
        path=(source_root/src["file"]).resolve()
        if root not in path.parents:raise ValueError(f"{sid}: source path escapes source root")
        if not path.is_file():raise FileNotFoundError(path)
        actual=sha256_file(path)
        if actual.lower()!=src["sha256"].lower():raise ValueError(f"{sid}: SHA-256 mismatch")
        normalized.append(src|{"path":path,"target_fraction":float(frac),"sha256":actual})
    if not math.isclose(fractions,1.0,abs_tol=1e-6):raise ValueError(f"target_fraction must sum to 1.0, got {fractions}")
    return normalized


def split_for(source_id: str, row_id: str):
    bucket=int(hashlib.sha256(f"{source_id}:{row_id}".encode()).hexdigest()[:8],16)%100
    return "holdout" if bucket<2 else "test" if bucket<5 else "val" if bucket<10 else "train"


def token_count(text: str):return len(text.encode("utf8"))+2

def encode(text: str):
    raw=np.frombuffer(text.encode("utf8"),dtype=np.uint8).astype(np.uint16)+BYTE_OFFSET
    return np.concatenate((np.array([SPECIAL_BOS],dtype=np.uint16),raw,np.array([SPECIAL_EOS],dtype=np.uint16)))


def load_clean_rows(sources, contamination, reject):
    exact=set();signatures=[];buckets=collections.defaultdict(set);rows=[]
    for src in sources:
        seen_ids=set()
        with src["path"].open("r",encoding="utf8") as f:
            for line_number,line in enumerate(f,1):
                if not line.strip():continue
                try:row=json.loads(line)
                except json.JSONDecodeError as e:raise ValueError(f"{src['id']} line {line_number}: invalid JSON") from e
                rid=row.get("id");raw=row.get("text")
                if not isinstance(rid,(str,int)):raise ValueError(f"{src['id']} line {line_number}: id missing")
                rid=str(rid)
                if rid in seen_ids:raise ValueError(f"{src['id']}: duplicate row id {rid}")
                seen_ids.add(rid)
                text,reason=clean_text(raw,src["domain"])
                if reason:reject[reason]+=1;continue
                norm=normalize_for_dedup(text);digest=hashlib.sha256(norm.encode()).hexdigest()
                if digest in exact:reject["exact_normalized_duplicate"]+=1;continue
                if any(p in norm for p in contamination):reject["frozen_benchmark_contamination"]+=1;continue
                sig=minhash_signature(norm);candidate=set()
                for h in sig[:4]:candidate.update(buckets[h])
                near=False
                if sig:
                    a=set(sig)
                    for idx in candidate:
                        b=set(signatures[idx]);j=len(a&b)/max(1,len(a|b))
                        if j>=0.80:near=True;break
                if near:reject["near_duplicate_minhash"]+=1;continue
                index=len(signatures);signatures.append(sig)
                for h in sig[:4]:buckets[h].add(index)
                exact.add(digest)
                rows.append({"id":rid,"source_id":src["id"],"language":src["language"],"domain":src["domain"],"title":row.get("title"),"url":row.get("url") or src["provenance_url"],"text":text,"normalized_sha256":digest,"split":split_for(src["id"],rid),"tokens":token_count(text),"selection_hash":hashlib.sha256(f"B2:968:{src['id']}:{rid}".encode()).hexdigest()})
    return rows


def select_train(rows, sources, target_train_tokens):
    selected=[];selection={};by_source=collections.defaultdict(list)
    for row in rows:
        if row["split"]=="train":by_source[row["source_id"]].append(row)
    for src in sources:
        candidates=sorted(by_source[src["id"]],key=lambda r:r["selection_hash"]);quota=math.floor(target_train_tokens*src["target_fraction"]);used=0;chosen=[]
        for row in candidates:
            if used>=quota:break
            chosen.append(row);used+=row["tokens"]
        if used<quota:raise ValueError(f"{src['id']}: only {used:,} clean train tokens available; quota is {quota:,}")
        selected.extend(chosen);selection[src["id"]]={"target_tokens":quota,"selected_tokens":used,"selected_documents":len(chosen),"available_documents":len(candidates)}
    return selected,selection


def write_outputs(rows,train_selected,sources,out: Path,target,reject,selection,source_manifest_sha,benchmark_files):
    out.mkdir(parents=True,exist_ok=True);train_keys={(r["source_id"],r["id"]) for r in train_selected};docs=[];summary={}
    for split in ("train","val","test","holdout"):
        summary[split]={}
        for lang in ("de","en"):
            if split=="train":chosen=[r for r in rows if r["split"]==split and r["language"]==lang and (r["source_id"],r["id"]) in train_keys]
            else:chosen=[r for r in rows if r["split"]==split and r["language"]==lang]
            chosen=sorted(chosen,key=lambda r:(r["source_id"],r["selection_hash"]));chunks=[];offset=0;bytes_count=0;domain_tokens=collections.Counter();source_tokens=collections.Counter()
            for r in chosen:
                z=encode(r["text"]);chunks.append(z);n=len(z);b=len(r["text"].encode("utf8"));docs.append({k:v for k,v in r.items() if k not in {"text","selection_hash","tokens"}}|{"start":offset,"end":offset+n,"bytes":b});offset+=n;bytes_count+=b;domain_tokens[r["domain"]]+=n;source_tokens[r["source_id"]]+=n
            data=np.concatenate(chunks) if chunks else np.empty((0,),dtype=np.uint16);path=out/f"{split}_{lang}.bin";data.astype("<u2").tofile(path)
            summary[split][lang]={"documents":len(chosen),"tokens":int(len(data)),"bytes":bytes_count,"sha256":sha256_file(path),"domain_tokens":dict(domain_tokens),"source_tokens":dict(source_tokens)}
    docs_path=out/"documents.jsonl";docs_path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in docs),encoding="utf8");public_sources=[{k:v for k,v in s.items() if k!="path"} for s in sources]
    manifest={"version":1,"purpose":"Neral-50M B2 broader continued pretraining","target_train_tokens":target,"actual_train_tokens":sum(summary["train"][l]["tokens"] for l in ("de","en")),"source_manifest_sha256":source_manifest_sha,"sources":public_sources,"selection":selection,"splits":summary,"documents_sha256":sha256_file(docs_path),"rejected":dict(reject),"packing":"BOS + UTF-8 bytes offset by 4 + EOS; document index enables boundary-safe blocks","split":"SHA256(source_id:row_id): 90/5/3/2 train/val/test/holdout before train quota selection","dedup":"NFKC/casefold exact + bounded 32-minhash word-5-shingle near-duplicate check >=0.80","cleaning":"Domain-aware: prose paragraph filtering; code preserves indentation/newlines; dialogue/structured preserve short turn/record lines","contamination":"Exact normalized long literals (>=20 chars) from frozen free-generation, RawMath, benchmark v3 and v4 are rejected; not a claim of exhaustive semantic contamination detection","contamination_source_sha256":benchmark_files,"external_model_calls":False,"raw_data_uploaded":False}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf8");return manifest


def build(source_root: Path,out: Path,target_train_tokens: int,repo_root: Path):
    if target_train_tokens<1_000_000:raise ValueError("B2 target should be at least 1,000,000 tokens")
    manifest_path=source_root/"manifest.json";manifest=json.loads(manifest_path.read_text(encoding="utf8"));sources=validate_manifest(manifest,source_root);contamination,benchmark_files=benchmark_contamination(repo_root);reject=collections.Counter();rows=load_clean_rows(sources,contamination,reject);train_selected,selection=select_train(rows,sources,target_train_tokens)
    return write_outputs(rows,train_selected,sources,out,target_train_tokens,reject,selection,sha256_file(manifest_path),benchmark_files)


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source-root",default="b2_sources");ap.add_argument("--out",default="pretrain50_b2_data");ap.add_argument("--target-train-tokens",type=int,default=100_000_000);args=ap.parse_args();report=build(Path(args.source_root),Path(args.out),args.target_train_tokens,Path(__file__).resolve().parent);print(json.dumps({"actual_train_tokens":report["actual_train_tokens"],"rejected":report["rejected"],"selection":report["selection"]},indent=2,ensure_ascii=False))


if __name__=="__main__":main()
