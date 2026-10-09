from __future__ import annotations
import argparse,collections,hashlib,json,statistics
from pathlib import Path

try:
    from .chat350_contract import EXPECTED_TOKENIZER_SHA256
    from .chat350_data import validate_dataset_root
    from .chat350_sft_format import encode_messages
    from .chat350_preference_format import encode_final_response
except ImportError:
    from chat350_contract import EXPECTED_TOKENIZER_SHA256
    from chat350_data import validate_dataset_root
    from chat350_sft_format import encode_messages
    from chat350_preference_format import encode_final_response

SPLITS=("train","val","test")

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""):h.update(c)
    return h.hexdigest()

def _quantiles(xs):
    if not xs:return {}
    ys=sorted(xs)
    def q(p):return ys[min(len(ys)-1,max(0,round((len(ys)-1)*p)))]
    return {"min":ys[0],"p50":q(.5),"p90":q(.9),"p99":q(.99),"max":ys[-1],"mean":statistics.fmean(ys)}

def _read_jsonl(path: Path):
    out=[]
    for n,line in enumerate(Path(path).read_text(encoding="utf8").splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        if not isinstance(row,dict):raise ValueError(f"{path}:{n}: row must be object")
        out.append(row)
    if not out:raise ValueError(f"empty split: {path}")
    return out

def audit_chat(tokenizer, root: Path, *, max_seq: int=512, hard_cap: int=8192):
    if not 1<=max_seq<=hard_cap:raise ValueError("invalid chat sequence budget")
    _,splits=validate_dataset_root(Path(root))
    report={}
    total_bad=0
    for split in SPLITS:
        lengths=[];bad=[];supervised=0;by_language=collections.Counter();by_domain=collections.Counter();by_source=collections.Counter()
        for row in splits[split]:
            try:
                enc=encode_messages(tokenizer,row["messages"],max_seq_len=hard_cap)
                n=len(enc.input_ids)
                supervised+=enc.assistant_tokens
                by_language[row["language"]]+=enc.assistant_tokens
                by_domain[row["domain"]]+=enc.assistant_tokens
                by_source[row["source"]]+=enc.assistant_tokens
            except ValueError:
                n=hard_cap+1
            lengths.append(n)
            if n>max_seq:
                bad.append({"id":row["id"],"tokens":n})
        total_bad+=len(bad)
        report[split]={"rows":len(lengths),"token_lengths":_quantiles(lengths),"violations":len(bad),"violation_examples":bad[:20],"supervised_tokens":supervised,"supervised_tokens_by_language":dict(sorted(by_language.items())),"supervised_tokens_by_domain":dict(sorted(by_domain.items())),"supervised_tokens_by_source":dict(sorted(by_source.items())),"supervised_token_fraction_by_language":{k:v/max(1,supervised) for k,v in sorted(by_language.items())},"supervised_token_fraction_by_domain":{k:v/max(1,supervised) for k,v in sorted(by_domain.items())},"supervised_token_fraction_by_source":{k:v/max(1,supervised) for k,v in sorted(by_source.items())}}
    return {"kind":"chat","max_seq":max_seq,"splits":report,"violations":total_bad,"pass":total_bad==0}

def audit_preferences(tokenizer, root: Path, *, max_seq: int=768, hard_cap: int=8192):
    if not 1<=max_seq<=hard_cap:raise ValueError("invalid preference sequence budget")
    report={};total_bad=0
    for split in SPLITS:
        rs=_read_jsonl(Path(root)/f"{split}.jsonl");lengths=[];bad=[];chosen_tokens=[];rejected_tokens=[];by_language=collections.Counter()
        for r in rs:
            rid=str(r.get("id",""))
            try:
                c=encode_final_response(tokenizer,r["context"],r["chosen"],max_seq_len=hard_cap)
                j=encode_final_response(tokenizer,r["context"],r["rejected"],max_seq_len=hard_cap)
                n=max(len(c.input_ids),len(j.input_ids)); chosen_tokens.append(c.response_tokens); rejected_tokens.append(j.response_tokens); by_language[str(r.get("language","")).lower()]+=1
            except ValueError:
                n=hard_cap+1
            lengths.append(n)
            if n>max_seq:bad.append({"id":rid,"tokens":n})
        total_bad+=len(bad)
        report[split]={"pairs":len(lengths),"max_pair_sequence_lengths":_quantiles(lengths),"chosen_response_tokens":_quantiles(chosen_tokens),"rejected_response_tokens":_quantiles(rejected_tokens),"languages":dict(sorted(by_language.items())),"violations":len(bad),"violation_examples":bad[:20]}
    return {"kind":"preferences","max_seq":max_seq,"splits":report,"violations":total_bad,"pass":total_bad==0}

def load_locked_tokenizer(path: Path):
    from tokenizers import Tokenizer
    if sha256_file(path)!=EXPECTED_TOKENIZER_SHA256:raise ValueError("locked tokenizer SHA mismatch")
    tok=Tokenizer.from_file(str(path))
    return tok

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--tokenizer",required=True)
    ap.add_argument("--data")
    ap.add_argument("--preferences")
    ap.add_argument("--chat-max-seq",type=int,default=512)
    ap.add_argument("--preference-max-seq",type=int,default=768)
    ap.add_argument("--out",required=True)
    ap.add_argument("--strict",action="store_true")
    a=ap.parse_args()
    if not a.data and not a.preferences:raise ValueError("provide --data and/or --preferences")
    tok=load_locked_tokenizer(Path(a.tokenizer));r={"schema":1,"tokenizer_sha256":EXPECTED_TOKENIZER_SHA256}
    passed=True
    if a.data:
        r["chat"]=audit_chat(tok,Path(a.data),max_seq=a.chat_max_seq);passed=passed and r["chat"]["pass"]
    if a.preferences:
        r["preferences"]=audit_preferences(tok,Path(a.preferences),max_seq=a.preference_max_seq);passed=passed and r["preferences"]["pass"]
    r["pass"]=passed
    Path(a.out).write_text(json.dumps(r,indent=2,ensure_ascii=False)+"\n",encoding="utf8")
    print(json.dumps({"pass":r["pass"],"chat":r.get("chat",{}).get("violations"),"preferences":r.get("preferences",{}).get("violations")}))
    if a.strict and not r["pass"]:raise SystemExit(2)
if __name__=="__main__":main()
