"""Produce original-Qwen-QLoRA compatible, not-yet-approved v4 train files.

This bridge has no optimizer, model or network dependency. It never exports the
320 test cases into the trainer. No trained-model or human-review claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from scripts.neral_v4_curriculum import verify

EXPORT_ROWS = {"train":1920,"validation":320}
MAX_TRAINER_FILE = 1_048_576
GENERATOR = Path(__file__).with_name("neral_v4_curriculum.py")


def _sha(b:bytes)->str:
    return hashlib.sha256(b).hexdigest()


def _json(b:object)->bytes:
    return (json.dumps(b,ensure_ascii=False,sort_keys=True,
                       separators=(",",":"))+"\n").encode("utf8")


def _expected(source:Path)->dict[str,bytes]:
    # Reconstructs all examples from the independent deterministic oracle
    # rather than trusting the JSONL or any attacker-controlled manifest.
    status=verify(source)
    if status.get("valid") is not True or status.get("training_allowed") is not False:
        raise ValueError("Candidate not verified or preapproved data unexpectedly present")
    manifest=json.loads((source/"manifest.json").read_text(encoding="utf8"))
    if manifest.get("rows")!={"train":1920,"val":320,"test":320}:
        raise ValueError("Unexpected source split inventory")
    if not GENERATOR.is_file() or GENERATOR.is_symlink():
        raise ValueError("Reviewed generator source missing")
    source_code_hash=_sha(GENERATOR.read_bytes())
    outputs={}
    digests={}
    for old,new in (("train","train"),("val","validation")):
        rows=[]
        reviews=[]
        data=(source/(old+".jsonl")).read_text(encoding="utf8").splitlines()
        if len(data)!=EXPORT_ROWS[new]:
            raise ValueError("Candidate split changed")
        for n,line in enumerate(data,1):
            original=json.loads(line)
            if (not isinstance(original,dict) or set(original)!={"prompt","completion"}
                    or len(original["prompt"])!=2 or len(original["completion"])!=1):
                raise ValueError("Unexpected original QLoRA split shape")
            messages=original["prompt"]+original["completion"]
            if [message.get("role") for message in messages]!=[
                    "system","user","assistant"]:
                raise ValueError("Unexpected message roles")
            if (any(set(msg)!={"role","content"} or not isinstance(msg["content"],str)
                    or not msg["content"].strip() for msg in messages)):
                raise ValueError("Blank or additional message fields")
            encoded=_json({"messages":messages}).decode("utf8").rstrip("\n")
            rows.append(encoded+"\n")
            reviews.append({
                "line":n,
                "row_sha256":_sha(encoded.encode("utf8")),
                "evidence":["local-original-generator","deterministic-answer-verification",
                            "isolated-train-validation-split"],
                "finding":"Generator-verified label, NOT an independent human semantic review",
            })
        result="".join(rows).encode("utf8")
        if len(result)>=MAX_TRAINER_FILE:
            raise ValueError("Pinned original trainer has a 1 MiB dataset cap")
        review={
            "schema":1,
            "dataset_sha256":_sha(result),
            "sources":{"scripts/neral_v4_curriculum.py":source_code_hash},
            "rows":reviews,
            "reviewer":"CPU-oracle only; needs model tokenizer and human quality review",
            "training_allowed":False,
        }
        outputs[new+".jsonl"]=result
        outputs[new+".review.json"]=_json(review)
        digests[new]=_sha(result)
    readiness={
        "schema":1,
        "status":"NERAL_V4_TRAINER_FORMAT_ONLY_NOT_TRAINING_APPROVAL",
        "rows":dict(EXPORT_ROWS),
        "test_rows_excluded":320,
        "data_sha256":digests,
        "generator_sha256":source_code_hash,
        "training_allowed":False,
        "tokenizer_verified":False,
        "training_performed":False,
        "adapter_created":False,
        "human_quality_reviewed":False,
    }
    outputs["readiness.json"]=_json(readiness)
    return outputs


def check_export(source:Path,output:Path)->dict:
    source,output=Path(source),Path(output)
    if (output.is_symlink() or not output.is_dir() or
            set(p.name for p in output.iterdir()) !=
            {"train.jsonl","train.review.json","validation.jsonl",
             "validation.review.json","readiness.json"}):
        raise ValueError("Unexpected local trainer export")
    generated=_expected(source)
    for name,expected in generated.items():
        path=output/name
        if path.is_symlink() or not path.is_file() or path.read_bytes()!=expected:
            raise ValueError("Trainer example/review changed")
    report=json.loads(generated["readiness.json"])
    return {k:report[k] for k in ("rows","test_rows_excluded",
        "training_allowed","tokenizer_verified","training_performed",
        "adapter_created","human_quality_reviewed","data_sha256")}


def prepare(source:Path,output:Path)->dict:
    source,output=Path(source).resolve(),Path(output)
    if not source.is_dir() or source.is_symlink() or output.is_symlink() or output.exists():
        raise ValueError("Source must exist; export destination must be new")
    resolved=output.resolve()
    if (resolved==source or source in resolved.parents or
            resolved in source.parents or not resolved.parent.is_dir()):
        raise ValueError("Trainer output must be a separate directory")
    generated=_expected(source)
    output.mkdir(exist_ok=False)
    for name,data in generated.items():
        with (output/name).open("xb") as handle:
            handle.write(data)
    return check_export(source,output)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument("--source",required=True,type=Path)
    p.add_argument("--out",required=True,type=Path)
    a=p.parse_args(argv)
    report=prepare(a.source,a.out)
    print(json.dumps({"rows":report["rows"],
                      "test_rows_excluded":report["test_rows_excluded"],
                      "training_allowed":False},sort_keys=True))


if __name__=="__main__":
    main()
