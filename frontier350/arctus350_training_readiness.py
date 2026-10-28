"""Read-only CPU-only training-readiness audit for approved clean NeralTalk350.

Counts languages, domains, recorded sources/licenses and exact original BPE
length eligibility (128/256). No raw rows, IDs, texts or tokens in output.
Counts cannot establish correctness, license legality or instruction quality.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

CLEAN_MANIFEST_SHA="fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31"
TOKENIZER_SHA="4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
SPLITS=("train","val","test")


def aggregate_eligibility(splits,encode):
    if not isinstance(splits,dict) or set(splits)!=set(SPLITS):
        raise ValueError("expected train, val and test")
    out={"schema":"arctus350-readiness-v1","split":{},"privacy":"metadata aggregates only; no raw text, ids or tokens"}
    for split in SPLITS:
        rows=splits[split]
        if not isinstance(rows,list) or not rows:
            raise ValueError("empty or malformed split: "+split)
        languages=collections.Counter()
        domains=collections.Counter()
        sources=collections.Counter()
        licenses=collections.Counter()
        domains_128=collections.Counter()
        domains_256=collections.Counter()
        eligible_128=eligible_256=0
        for row in rows:
            if not isinstance(row,dict):
                raise ValueError("malformed row")
            for field in ("language","domain","source","license"):
                if not isinstance(row.get(field),str) or not row[field].strip():
                    raise ValueError("invalid required metadata: "+field)
            languages[row["language"]]+=1
            domains[row["domain"]]+=1
            sources[row["source"]]+=1
            licenses[row["license"]]+=1
            for max_tokens in (128,256):
                try:
                    tokens=encode(row,max_tokens)
                except ValueError:
                    continue
                if not tokens:
                    raise ValueError("encoded row must have at least one token")
                if max_tokens==128:
                    eligible_128+=1
                    domains_128[row["domain"]]+=1
                else:
                    eligible_256+=1
                    domains_256[row["domain"]]+=1
        out["split"][split]={
            "rows":len(rows),
            "language":dict(sorted(languages.items())),
            "domain":dict(sorted(domains.items())),
            "source":dict(sorted(sources.items())),
            "license":dict(sorted(licenses.items())),
            "eligible_128":eligible_128,
            "eligible_256":eligible_256,
            "eligible_domain_128":dict(sorted(domains_128.items())),
            "eligible_domain_256":dict(sorted(domains_256.items())),
        }
    out["interpretation"]="Metadata and context length only: not training approval, not proof of legal licensing, independent semantics or model capability."
    return out


def sha256(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--tokenizer",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError("refuse overwrite readiness evidence")
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .chat350_sft_format import encode_messages
    from tokenizers import Tokenizer
    if sha256(args.tokenizer)!=TOKENIZER_SHA:
        raise ValueError("locked 16K tokenizer SHA mismatch")
    manifest,splits=validate_dataset_root(args.data)
    if manifest["manifest_sha256"]!=CLEAN_MANIFEST_SHA:
        raise ValueError("derived clean dataset manifest SHA mismatch")
    validate_prompt_holdouts(splits)
    tok=Tokenizer.from_file(str(args.tokenizer))
    def encode(row,max_seq):
        return encode_messages(tok,row["messages"],max_seq_len=max_seq).input_ids
    report=aggregate_eligibility(splits,encode)
    report["source_dataset_sha256"]=CLEAN_MANIFEST_SHA
    report["tokenizer_sha256"]=TOKENIZER_SHA
    args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2,sort_keys=True)+"\n",encoding="utf8")
    print(json.dumps(report,ensure_ascii=False,sort_keys=True),flush=True)


if __name__=="__main__":
    main()
