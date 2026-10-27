"""NERAL-P1 independent human-reviewed personality rubric.

No model inference, no dataset training, no automatic 'LLM judges itself'
grading, and no claims of attested humans or actual model provenance.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

SUITE = Path(__file__).with_name("eval") / "neral_persona_style_holdout_v1.jsonl"
CATEGORIES = frozenset({
    "identity","uncertainty","coding","initiative",
    "tone","safety","transparency","language",
})
FIELDS = frozenset({"id","category","prompt","rubric","train_allowed","source"})
MODEL_IDS = frozenset({"qwen3.5:4b","qwen3.5:9b","NERAL-P1-4B-LoRA"})


class StyleReviewError(ValueError):
    pass


def _hash(value) -> str:
    content=json.dumps(value,ensure_ascii=False,sort_keys=True,
                       separators=(",",":"),allow_nan=False).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def _norm(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+"," ",text.casefold(),flags=re.UNICODE).split())


def load_style_holdout(path: Path | str = SUITE) -> list[dict]:
    file=Path(path)
    if not file.is_file() or file.stat().st_size > 100_000:
        raise StyleReviewError("Frozen style holdout missing or oversized")
    try:
        cases=[json.loads(line) for line in file.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    except (ValueError,UnicodeError) as exc:
        raise StyleReviewError("Invalid holdout JSONL") from exc
    return cases


def validate_style_holdout(cases: list[dict],train_examples: list[dict]) -> dict:
    if not isinstance(cases,list) or not 1 <= len(cases) <= 100:
        raise StyleReviewError("Invalid style suite size")
    if not isinstance(train_examples,list):
        raise StyleReviewError("Invalid training examples")
    seen_ids=set()
    prompts=set()
    categories=set()
    training_prompts=set()
    for row in train_examples:
        messages=row.get("messages") if isinstance(row,dict) else None
        if not isinstance(messages,list) or not messages or not isinstance(messages[0],dict):
            raise StyleReviewError("Invalid train source")
        user=messages[0].get("content")
        if not isinstance(user,str):
            raise StyleReviewError("Invalid training prompt")
        training_prompts.add(_norm(user))
    for case in cases:
        if not isinstance(case,dict) or set(case)!=FIELDS:
            raise StyleReviewError("Untrusted style test metadata")
        ident=case["id"]
        if not isinstance(ident,str) or not re.fullmatch(r"nstyle-[0-9]{3}",ident) or ident in seen_ids:
            raise StyleReviewError("Unknown or repeated holdout ID")
        seen_ids.add(ident)
        cat=case["category"]
        if not isinstance(cat,str) or cat not in CATEGORIES:
            raise StyleReviewError("Invalid style category")
        categories.add(cat)
        if case["train_allowed"] is not False or case["source"]!="ai-authored-original-heldout-2026-10-28":
            raise StyleReviewError("Style evaluation cannot be used for training")
        for field in ("prompt","rubric"):
            text=case[field]
            if not isinstance(text,str) or not text.strip() or len(text)>2000:
                raise StyleReviewError("Empty or oversized review rubric")
        norm=_norm(case["prompt"])
        if norm in training_prompts or norm in prompts:
            raise StyleReviewError("Training/holdout overlap or duplicate")
        prompts.add(norm)
    return {"count":len(cases),"categories":sorted(categories),
            "sha256":_hash(cases),"training_overlap":False,
            "status":"unscored_holdout"}


def make_style_review(cases: list[dict], answers: dict[str,str], *,
                      model_id: str) -> dict:
    """Build an ungraded review sheet from user-supplied model transcripts."""
    audit=validate_style_holdout(cases,[])
    if model_id not in MODEL_IDS:
        raise StyleReviewError("Model label not approved")
    if not isinstance(answers,dict) or set(answers)!={c["id"] for c in cases}:
        raise StyleReviewError("Every heldout question requires one transcript")
    rows=[]
    for case in cases:
        text=answers[case["id"]]
        if not isinstance(text,str) or not text.strip() or len(text)>10000:
            raise StyleReviewError("Invalid or empty model transcript")
        rows.append({
            "id":case["id"],"answer":text,
            "rubric":case["rubric"], "rating":None,
            "reviewer":None, "notes":"",
        })
    return {"suite_sha256":audit["sha256"],"model_id":model_id,
            "provenance":"self_reported_manual_transcript",
            "rows":rows}


def summarize_style_review(cases: list[dict], review: dict) -> dict:
    audit=validate_style_holdout(cases,[])
    if not isinstance(review,dict) or set(review)!={
        "suite_sha256","model_id","provenance","rows"
    }:
        raise StyleReviewError("Untrusted review report")
    if review["suite_sha256"]!=audit["sha256"]:
        raise StyleReviewError("Review used wrong frozen holdout suite")
    if review["model_id"] not in MODEL_IDS:
        raise StyleReviewError("Invalid model identity")
    if review["provenance"]!="self_reported_manual_transcript":
        raise StyleReviewError("Unrecognized review provenance")
    rows=review["rows"]
    if not isinstance(rows,list) or len(rows)!=len(cases):
        raise StyleReviewError("Missing or duplicate reviewed cases")
    by_id={c["id"]:c for c in cases}
    seen=set()
    total=0
    rated=0
    for row in rows:
        if not isinstance(row,dict) or set(row)!={
            "id","answer","rubric","rating","reviewer","notes"
        }:
            raise StyleReviewError("Review row malformed")
        ident=row["id"]
        if ident not in by_id or ident in seen or row["rubric"]!=by_id[ident]["rubric"]:
            raise StyleReviewError("Tampered rubric or duplicate case")
        seen.add(ident)
        if not isinstance(row["answer"],str) or not row["answer"].strip() or len(row["answer"])>10000:
            raise StyleReviewError("Missing actual model answer")
        if not isinstance(row["notes"],str) or len(row["notes"])>2000:
            raise StyleReviewError("Invalid reviewer notes")
        rating=row["rating"]
        if rating is None:
            if row["reviewer"] is not None:
                raise StyleReviewError("Unscored response cannot claim reviewer")
            continue
        if type(rating) is not int or not 0 <= rating <= 2 or row["reviewer"]!="human":
            raise StyleReviewError("Rating must be human-reported ordinal 0/1/2")
        rated+=1
        total+=rating
    complete=rated==len(cases)
    return {
        "metric":"NERAL-P1 human style rubric (0-2, self-reported)",
        "model_id":review["model_id"],
        "suite_sha256":audit["sha256"],
        "status":"human_reported_unverified" if complete else "not_measured",
        "score_pct":round(100*total/(2*len(cases)),2) if complete else None,
        "rated_cases":rated,
        "total_cases":len(cases),
        "warning":(
            "These ratings and model transcripts are user-supplied, not "
            "identity-verified or cryptographically attested. Not NIS2, "
            "not a general intelligence score; never train on the holdout."
        ),
    }


def main() -> None:
    import argparse
    parser=argparse.ArgumentParser(description="Human-only NERAL personality rubric; no inference")
    modes=parser.add_subparsers(dest="action",required=True)
    prepare=modes.add_parser("prepare")
    prepare.add_argument("--answers",required=True,type=Path)
    prepare.add_argument("--model",required=True,choices=sorted(MODEL_IDS))
    prepare.add_argument("--output",required=True,type=Path)
    review=modes.add_parser("score")
    review.add_argument("--sheet",required=True,type=Path)
    args=parser.parse_args()
    cases=load_style_holdout()
    if args.action=="prepare":
        if args.answers.stat().st_size>200_000:
            raise StyleReviewError("Answer file too large")
        answers=json.loads(args.answers.read_text(encoding="utf-8"))
        report=make_style_review(cases,answers,model_id=args.model)
        data=(json.dumps(report,ensure_ascii=False,indent=2)+"\n").encode("utf-8")
        with args.output.open("xb") as f:
            f.write(data)
        print("Created unscored human review: "+str(args.output))
    else:
        if args.sheet.stat().st_size>512_000:
            raise StyleReviewError("Review file too large")
        report=json.loads(args.sheet.read_text(encoding="utf-8"))
        summary=summarize_style_review(cases,report)
        print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
