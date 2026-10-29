"""Offline native Arctus350 NeralTalk SFT numeric-label integrity preflight.

Scope: narrowly recognizable exact arithmetic prompts and bare-number targets.
Unscored free-form text must NEVER be interpreted as numerically correct.
Checks only train/val; the test split is sealed and is NEVER iterated.
No model weights, tokens, trainer execution, web, or external provider calls.
Only aggregate counts are emitted: no prompts, answers, IDs or hashes leak.
"""
from __future__ import annotations

from fractions import Fraction
import argparse
import hashlib
import json
from pathlib import Path
import re
import unicodedata

SPLITS=("train","val")
NUMBER=r"\d{1,9}(?:[.,]\d{1,8})?"
NUMERIC_REPLY=re.compile(r"[+-]?\d{1,9}(?:[.,]\d{1,8})?(?:/\d{1,9})?\Z")
BINARY=re.compile(
    rf"({NUMBER})\s*(\+|-|\*|/|plus|minus|mal|times|geteilt\s+durch|divided\s+by)\s*({NUMBER})\Z"
)
PREFIX=re.compile(r"\A(?:berechne|calculate|compute|wie\s+viel\s+ist|wieviel\s+ist|what\s+is)\s+",re.I)
DE_ADD=re.compile(rf"addiere\s+({NUMBER})\s+und\s+({NUMBER})\Z",re.I)
EN_ADD=re.compile(rf"add\s+({NUMBER})\s+and\s+({NUMBER})\Z",re.I)
SUFFIX=re.compile(
    r"(?:\?\s*)?(?:gib\s+nur\s+die\s+zahl\s+aus|reply\s+only\s+with\s+the\s+number)\.?\Z",re.I
)
_OPS={
    "+":"+","plus":"+","-":"-","minus":"-","*":"*",
    "times":"*","mal":"*","/":"/","geteilt durch":"/","divided by":"/",
}


class LabelIntegrityError(ValueError):
    """Fail-closed native SFT label issue; no raw text is included."""


def _canon(text):
    return " ".join(unicodedata.normalize("NFKC",text).casefold().split())


def _number(text):
    return Fraction(text.replace(",","."))


def _arithmetic(prompt):
    """Return mathematically proved answer only if whole input matches grammar."""
    if not isinstance(prompt,str) or not 1<=len(prompt)<=256 or any(ord(c)<32 for c in prompt):
        return None
    s=unicodedata.normalize("NFKC",prompt).casefold().strip()
    s=SUFFIX.sub("",s).rstrip(" ?!.")
    for pattern in (DE_ADD,EN_ADD):
        found=pattern.fullmatch(s)
        if found:
            return _number(found[1])+_number(found[2])
    s=PREFIX.sub("",s,count=1)
    found=BINARY.fullmatch(s)
    if not found:
        return None
    a,op,b=_number(found[1])," ".join(found[2].split()),_number(found[3])
    op=_OPS[op]
    if op=="+":return a+b
    if op=="-":return a-b
    if op=="*":return a*b
    if b!=0:return a/b
    return None


def _numeric_reply(text):
    if not isinstance(text,str) or len(text)>30:
        return None
    value=text.strip()
    if not NUMERIC_REPLY.fullmatch(value):
        return None
    try:
        return _number(value)
    except (ValueError,ZeroDivisionError):
        return None


def _messages(row):
    if (not isinstance(row,dict) or not isinstance(row.get("id"),str) or
        not row["id"] or not isinstance(row.get("domain"),str) or
        not isinstance(row.get("messages"),list)):
        raise LabelIntegrityError("invalid normalized SFT row")
    messages=row["messages"]
    if not 2<=len(messages)<=32 or len(messages)%2:
        raise LabelIntegrityError("invalid number of chat turns")
    for idx,m in enumerate(messages):
        if (not isinstance(m,dict) or m.get("role")!=("user" if idx%2==0 else "assistant")
            or not isinstance(m.get("content"),str) or not m["content"].strip()):
            raise LabelIntegrityError("invalid normalized chat message")
    return messages


def audit_numeric_supervision(splits):
    """Audit finite, strict direct math; do not touch held-out test rows."""
    if not isinstance(splits,dict) or set(splits)!={"train","val","test"}:
        raise LabelIntegrityError("expected train/val/test split dictionary")
    output={
        "schema":"arctus350-math-label-integrity-v1",
        "status":"AUDIT_ONLY_NOT_MODEL_EVAL",
        "test":"SEALED_NOT_INSPECTED",
        "model_accuracy":"NOT_MEASURED",
        "training_started":False,
        "proof_of_training_data_quality":False,
        "scope":"strict exact arithmetic references only; unsupported math and free text remain unscored",
    }
    healthy=True
    for split in SPLITS:
        rows=splits[split]
        if not isinstance(rows,list) or not rows:
            raise LabelIntegrityError("train and val require non-empty normalized rows")
        seen_numeric_context={}
        contradictory=set()
        observed={
            "rows":len(rows),"math_rows":0,"verified_direct_arithmetic":0,
            "invalid_labels":0,"unscored_math_rows":0,
            "conflicting_numeric_contexts":0,
        }
        for row in rows:
            messages=_messages(row)
            if row["domain"].casefold().strip()!="math":
                continue
            observed["math_rows"]+=1
            recognized=False
            for answer_idx in range(1,len(messages),2):
                prompt=messages[answer_idx-1]["content"]
                answer=messages[answer_idx]["content"]
                numeric=_numeric_reply(answer)
                if numeric is not None:
                    # Distinguish a repeated question in DIFFERENT dialogue states.
                    # A short greeting repeated across conversations is not automatically
                    # a contradiction or evidence of poor model training.
                    context="\u241e".join(
                        ("u" if m["role"]=="user" else "a")+":"+_canon(m["content"])
                        for m in messages[:answer_idx])
                    fingerprint=hashlib.sha256(context.encode("utf-8")).digest()
                    previous=seen_numeric_context.setdefault(fingerprint,numeric)
                    if previous!=numeric:
                        contradictory.add(fingerprint)
                expected=_arithmetic(prompt)
                if expected is None or numeric is None:
                    continue
                recognized=True
                if expected==numeric:
                    observed["verified_direct_arithmetic"]+=1
                else:
                    observed["invalid_labels"]+=1
            if not recognized:
                observed["unscored_math_rows"]+=1
        observed["conflicting_numeric_contexts"]=len(contradictory)
        if observed["invalid_labels"] or observed["conflicting_numeric_contexts"]:
            healthy=False
        output[split]=observed
    output["pass"]=healthy
    output["decision"]="PRETRAIN_GATE_PASS_LIMITED_SCOPE" if healthy else "REJECT_DIRTY_SFT"
    output["limitations"]=(
        "Only full-match small arithmetic with bare numeric labels is checked. "
        "Unscored examples can still be false, copyrighted, sensitive, leaked or low quality. "
        "No generated text, model inference, checkpoint quality or release is verified."
    )
    return output


def require_clean_numeric_supervision(splits):
    """The real native trainer uses this BEFORE CUDA or optimizer steps."""
    report=audit_numeric_supervision(splits)
    if not report["pass"]:
        raise LabelIntegrityError("invalid numeric supervision or conflicting math labels")
    return report


def _sha256_file(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def audit_directory(root, out):
    """Read existing NeralTalk train/VAL only; never inspect TEST data.

    Writes a *new* local aggregate record exclusively. This audit does not
    verify that the corpora are licensed, uncontaminated or safe in all respects.
    """
    from .chat350_data import read_jsonl
    root,out=Path(root),Path(out)
    if out.exists():
        raise FileExistsError("do not overwrite any previous research evidence")
    train_path,val_path=root/"train.jsonl",root/"val.jsonl"
    # No reference to root/'test.jsonl' and no test-set iteration.
    records={"train":read_jsonl(train_path),"val":read_jsonl(val_path),"test":None}
    report=audit_numeric_supervision(records)
    report["train_sha256"]=_sha256_file(train_path)
    report["val_sha256"]=_sha256_file(val_path)
    report["source"]="existing local NeralTalk350 train/val only, original bytes SHA-256"
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("x",encoding="utf-8") as dest:
        json.dump(report,dest,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)
        dest.write("\n")
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True,help="local NeralTalk350 dataset directory")
    parser.add_argument("--out",type=Path,required=True,help="NEW local privacy-safe aggregate JSON path")
    args=parser.parse_args()
    report=audit_directory(args.data,args.out)
    print(f"Arctus350 label integrity: {report['decision']}; "
          f"train verified={report['train']['verified_direct_arithmetic']}; "
          f"VAL verified={report['val']['verified_direct_arithmetic']}; "
          "TEST=SEALED_NOT_INSPECTED",flush=True)
    raise SystemExit(0 if report["pass"] else 2)


if __name__=="__main__":
    main()
