"""Read-only Arctus350 SFT token-supervision audit.

Audits the *existing native message encoder*: target mask density,
final EOS labels and context losses per split/language/domain.
This program does not train, evaluate model weights or produce neural scores.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .chat350_contract import EXPECTED_SPECIALS, EXPECTED_TOKENIZER_SHA256
from .chat350_sft_format import EncodedChatExample


class _Counts:
    def __init__(self):
        self.rows=0
        self.eligible_rows=0
        self.ineligible_rows=0
        self.assistant_targets=0
        self.context_targets=0
        self.terminal_eos_targets=0
        self.sequence_tokens=0

    def add(self, result):
        self.eligible_rows+=1
        self.assistant_targets+=result["assistant_targets"]
        self.context_targets+=result["context_targets"]
        self.terminal_eos_targets+=result["eos_targets"]
        self.sequence_tokens+=result["sequence_length"]

    def report(self):
        total=self.assistant_targets+self.context_targets
        return {
            "rows":self.rows,"eligible_rows":self.eligible_rows,
            "ineligible_rows":self.ineligible_rows,
            "assistant_targets":self.assistant_targets,
            "context_targets":self.context_targets,
            "terminal_eos_targets":self.terminal_eos_targets,
            "sequence_tokens":self.sequence_tokens,
            "supervision_fraction":self.assistant_targets/total if total else 0.0,
            "context_fraction":self.context_targets/total if total else 0.0,
        }


def validate_encoded(example: EncodedChatExample, *, eos_id: int, pad_id: int) -> dict:
    """Fail closed on malformed assistant-only masks, EOS and padding leaks."""
    if (not isinstance(example,EncodedChatExample)
        or type(eos_id) is not int or type(pad_id) is not int
        or eos_id<0 or pad_id<0 or eos_id==pad_id
        or not isinstance(example.input_ids,list)
        or not isinstance(example.targets,list)
        or not example.input_ids
        or len(example.input_ids)!=len(example.targets)):
        raise ValueError("invalid SFT encoded example")
    x,y=example.input_ids,example.targets
    if (any(type(v) is not int or v<0 or v>=16_384 or v==pad_id for v in x)
        or any(type(v) is not int or (v!=-100 and (v<0 or v>=16_384 or v==pad_id)) for v in y)
        or x[0]!=EXPECTED_SPECIALS["<|bos|>"] or y[0]!=-100):
        raise ValueError("invalid native token or supervision mask")
    if y[-1]!=eos_id or sum(v==eos_id for v in y)!=1:
        raise ValueError("missing/multiple terminal EOS targets")
    supervised=sum(v!=-100 for v in y)
    if (type(example.assistant_tokens) is not int
        or example.assistant_tokens!=supervised
        or supervised<2 or supervised==len(y)):
        raise ValueError("assistant mask is inconsistent or lacks answer context")
    return {
        "assistant_targets":supervised,
        "context_targets":len(y)-supervised,
        "eos_targets":1,
        "sequence_length":len(x),
    }


def audit_supervision(splits: dict, *, encode, max_seq_len: int,
                      eos_id: int, pad_id: int) -> dict:
    """Only data aggregates; actual encoder callback required by caller."""
    if (not isinstance(splits,dict) or set(splits)!={"train","val","test"}
        or type(max_seq_len) is not int or not 1<=max_seq_len<=8192
        or not callable(encode)):
        raise ValueError("must supply 3 native splits, valid sequence length and encoder")
    output={}
    for split in ("train","val","test"):
        rows=splits[split]
        if not isinstance(rows,list) or not rows:
            raise ValueError("nonempty split rows required")
        overall=_Counts()
        languages=defaultdict(_Counts)
        domains=defaultdict(_Counts)
        for row in rows:
            if not isinstance(row,dict):
                raise ValueError("invalid NeralTalk row")
            for field in ("language","domain"):
                val=row.get(field)
                if not isinstance(val,str) or not val or len(val)>50 or not val.strip():
                    raise ValueError("invalid NeralTalk metadata")
            if not isinstance(row.get("messages"),list):
                raise ValueError("missing native chat messages")
            lang,domain=row["language"],row["domain"]
            counters=(overall,languages[lang],domains[domain])
            for counter in counters:counter.rows+=1
            try:
                encoded=encode(row,max_seq_len)
            except ValueError as exc:
                if str(exc).startswith("encoded example exceeds max_seq_len="):
                    for counter in counters:counter.ineligible_rows+=1
                    continue
                raise
            observation=validate_encoded(encoded,eos_id=eos_id,pad_id=pad_id)
            for counter in counters:counter.add(observation)
        output[split]={**overall.report(),
                       "languages":{k:v.report() for k,v in sorted(languages.items())},
                       "domains":{k:v.report() for k,v in sorted(domains.items())}}
    return {
        "schema":"arctus350-supervision-audit-v1",
        "status":"AUDIT_ONLY",
        "model_metrics":"NOT_MEASURED",
        "compute":"CPU, no model inference, no optimizer steps",
        "privacy":"aggregate token counts only; no text, IDs, raw tokens or prompts",
        "source":"native NeralTalk350 message encoder, assistant-only causal targets",
        "max_seq_len":max_seq_len,
        "splits":output,
        "interpretation":"Token target distribution is not proof of a good dataset, correct answers or higher model intelligence.",
    }


def _sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data",type=Path,required=True,help="Existing NeralTalk350 split root")
    ap.add_argument("--tokenizer",type=Path,required=True,help="Original locked 16K BPE file")
    ap.add_argument("--out",type=Path,required=True,help="NEW JSON evidence file (no overwrite)")
    ap.add_argument("--seq",type=int,default=256)
    args=ap.parse_args()
    if args.out.exists():
        raise FileExistsError("Refuse to overwrite previous experiment evidence")
    if not args.data.is_dir():
        raise FileNotFoundError("NeralTalk350 dataset root is not available")
    if not args.tokenizer.is_file() or _sha256(args.tokenizer)!=EXPECTED_TOKENIZER_SHA256:
        raise ValueError("Original locked Arctus350 tokenizer is missing or invalid")
    try:
        from tokenizers import Tokenizer
    except ImportError as exc:
        raise RuntimeError("Optional 'tokenizers' dependency required for original native tokenizer") from exc
    from .chat350_data import validate_dataset_root,validate_prompt_holdouts
    from .chat350_sft_format import encode_messages
    manifest,splits=validate_dataset_root(args.data)
    validate_prompt_holdouts(splits)
    tokenizer=Tokenizer.from_file(str(args.tokenizer))
    if (tokenizer.get_vocab_size()!=16_384
        or any(tokenizer.token_to_id(t)!=idx for t,idx in EXPECTED_SPECIALS.items())):
        raise ValueError("Original tokenizer shape or special IDs mismatch")
    report=audit_supervision(
        splits,
        encode=lambda row,max_seq_len:encode_messages(
            tokenizer,row["messages"],max_seq_len=max_seq_len),
        max_seq_len=args.seq,
        eos_id=EXPECTED_SPECIALS["<|eos|>"],
        pad_id=EXPECTED_SPECIALS["<|pad|>"],
    )
    report["data_manifest_sha256"]=manifest["manifest_sha256"]
    report["tokenizer_sha256"]=EXPECTED_TOKENIZER_SHA256
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(report,sort_keys=True,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    print(json.dumps(report,sort_keys=True,ensure_ascii=False))


if __name__=="__main__":
    main()
