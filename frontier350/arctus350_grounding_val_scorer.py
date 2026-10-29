"""Offline VAL-only scorer for *reported* native Arctus350 grounding outputs.

No inference, no network, no model weights, no test-set scoring.
IMPORTANT: a claimed SHA-256 cannot prove that observations came from a
checkpoint. Output is always UNVERIFIED_REPORTED_OUTPUTS, never promotion.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from .arctus350_grounding_curriculum_v1 import make_splits, verify_row

_SCHEMA="arctus350-reported-inference-v1"
_TOKENIZER_SHA="4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd"
_HEX64=re.compile(r"[a-f0-9]{64}\Z")
_ALLOWED_KINDS=frozenset({"temperature","mass_known","mass_missing","compare"})
_ALLOWED_LANGS=frozenset({"de","en"})


class GroundingScoreError(ValueError):
    """Reject malformed, incomplete or misleading evaluation evidence."""


def _validate_manifest(manifest):
    if not isinstance(manifest,dict) or set(manifest)!={
        "schema","claimed_model","checkpoint_sha256","tokenizer_sha256","decoding"
    }:
        raise GroundingScoreError("Incomplete or unsupported run manifest")
    if manifest["schema"]!=_SCHEMA or manifest["claimed_model"]!="native-arctus350":
        raise GroundingScoreError("Only reported native-Arctus350 evidence supported")
    if not isinstance(manifest["checkpoint_sha256"],str) or not _HEX64.fullmatch(manifest["checkpoint_sha256"]):
        raise GroundingScoreError("Missing SHA-256 checkpoint fingerprint")
    if manifest["tokenizer_sha256"]!=_TOKENIZER_SHA:
        raise GroundingScoreError("Original locked native 16K tokenizer required")
    decoding=manifest["decoding"]
    if (not isinstance(decoding,dict) or set(decoding)!={"temperature","max_new_tokens"} or
        type(decoding["temperature"]) not in (int,float) or decoding["temperature"]!=0 or
        type(decoding["max_new_tokens"]) is not int or
        not 1<=decoding["max_new_tokens"]<=128):
        raise GroundingScoreError("Expected explicit deterministic decoding limits")


def _valid_observations(observations,expected):
    if not isinstance(observations,list) or len(observations)!=len(expected):
        raise GroundingScoreError("Exactly one observation for each frozen VAL case required")
    by_id={}
    for obs in observations:
        if not isinstance(obs,dict) or set(obs)!={"id","output"}:
            raise GroundingScoreError("Each observation needs only id and raw output")
        key=obs["id"]
        val=obs["output"]
        if not isinstance(key,str) or key not in expected or key in by_id:
            raise GroundingScoreError("Unknown, duplicate or non-VAL case ID")
        if (type(val) is not str or not 0<len(val)<=512 or
            any(ord(ch)<32 or ord(ch)==127 for ch in val) or not val.strip()):
            raise GroundingScoreError("Blank, control-character or oversized model output")
        by_id[key]=val
    if set(by_id)!=set(expected):
        raise GroundingScoreError("VAL coverage mismatch")
    return by_id


def _group(groups):
    return {k:{"total":v["total"],"correct":v["correct"],
               "incorrect":v["total"]-v["correct"],
               "accuracy":round(v["correct"]/v["total"],6)}
            for k,v in sorted(groups.items())}


def score_val(observations:list, manifest:dict) -> dict:
    """Score original, complete VAL examples; NEVER certify provenance of text."""
    _validate_manifest(manifest)
    rows=make_splits()["val"]
    if not rows or any(not verify_row(row) for row in rows):
        raise GroundingScoreError("Invalid reference corpus")
    expected={row["id"]:row for row in rows}
    if len(expected)!=len(rows):
        raise GroundingScoreError("Duplicate frozen reference IDs")
    outputs=_valid_observations(observations,expected)
    groups_lang={}
    groups_task={}
    groups_pair={}
    correct=0
    false_abstentions=0
    guessed_missing_mass=0
    for key,row in expected.items():
        lang=row["language"]
        kind=row["task_type"]
        if lang not in _ALLOWED_LANGS or kind not in _ALLOWED_KINDS:
            raise GroundingScoreError("Reference data outside supported categories")
        output=outputs[key].strip()
        gold=row["messages"][1]["content"]
        match=output==gold
        correct+=int(match)
        for groups,group_key in ((groups_lang,lang),(groups_task,kind),
                                  (groups_pair,lang+"/"+kind)):
            tally=groups.setdefault(group_key,{"total":0,"correct":0})
            tally["total"]+=1
            tally["correct"]+=int(match)
        if kind!="mass_missing" and output in ("nicht angegeben","not provided"):
            false_abstentions+=1
        # An incorrect reply is not necessarily a fabricated measurement.\n        # Count only numeric mass guesses (with optional g/kg units).\n        if kind=="mass_missing" and not match and re.fullmatch(\n            r"[+-]?\d+(?:[.,]\d+)?(?:\s*(?:kg|g))?", output, re.I\n        ):\n            guessed_missing_mass+=1
    total=len(rows)
    return {
        "schema":"arctus350-grounding-val-score-v1",
        "split":"VAL_ONLY",
        "status":"UNVERIFIED_REPORTED_OUTPUTS",
        "claimed_model":"native-arctus350",
        "checkpoint_sha256":manifest["checkpoint_sha256"],
        "tokenizer_sha256":manifest["tokenizer_sha256"],
        "generation_settings":dict(manifest["decoding"]),
        "total":total,
        "correct":correct,
        "incorrect":total-correct,
        "accuracy":round(correct/total,6),
        "false_abstentions":false_abstentions,
        "guessed_missing_mass":guessed_missing_mass,
        "by_language":_group(groups_lang),
        "by_task":_group(groups_task),
        "by_language_and_task":_group(groups_pair),
        "native_generation_verified":False,
        "release_eligible":False,
        "model_intelligence_improved":False,
        "test_set_evaluated":False,
        "note":"Scores submitted strings against synthetic VAL references only. "
               "Checkpoint identity, real inference and quality gains are NOT verified.",
    }


def _read_json(path,limit):
    path=Path(path)
    if not path.is_file() or not 0<path.stat().st_size<=limit:
        raise GroundingScoreError("Input JSON file missing, empty or oversized")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError,json.JSONDecodeError) as exc:
        raise GroundingScoreError("Invalid manifest JSON") from exc


def _read_jsonl(path):
    path=Path(path)
    if not path.is_file() or not 0<path.stat().st_size<=1024*1024:
        raise GroundingScoreError("Input JSONL file missing, empty or oversized")
    try:
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f]
    except (UnicodeError,json.JSONDecodeError) as exc:
        raise GroundingScoreError("Malformed prediction JSONL") from exc


def score_val_jsonl(predictions_path,manifest_path):
    return score_val(_read_jsonl(predictions_path),_read_json(manifest_path,4096))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_cli(*,predictions_path,manifest_path,output_path):
    dest=Path(output_path)
    if dest.exists():
        raise FileExistsError("Refuse to overwrite previous evidence")
    result=score_val_jsonl(predictions_path,manifest_path)
    result["predictions_sha256"]=_sha(predictions_path)
    result["manifest_sha256"]=_sha(manifest_path)
    # Exclusive create: no silent overwrite or partial overwrite of older reports.
    with dest.open("x",encoding="utf-8") as out:
        json.dump(result,out,sort_keys=True,ensure_ascii=False,indent=2)
        out.write("\n")
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-predictions",required=True,type=Path)
    parser.add_argument("--manifest",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    args=parser.parse_args()
    result=run_cli(predictions_path=args.val_predictions,manifest_path=args.manifest,output_path=args.out)
    print(json.dumps({"status":result["status"],"split":result["split"],
                      "total":result["total"],"accuracy":result["accuracy"],
                      "release_eligible":False},sort_keys=True))


if __name__=="__main__":
    main()
