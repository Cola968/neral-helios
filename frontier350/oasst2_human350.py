from __future__ import annotations
import gzip, hashlib, json, unicodedata
from pathlib import Path

ARCHIVE_NAME="2023-11-05_oasst2_ready.messages.jsonl.gz"
ARCHIVE_SHA256="a9f240c4c77aa1378364f70d37e753c07ba284e247b019d700e1947a0e5da751"
LICENSE="Apache-2.0"
SOURCE="OpenAssistant/oasst2"

def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for c in iter(lambda:f.read(1024*1024),b""): h.update(c)
    return h.hexdigest()

def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC",str(text)).casefold().split())

def reviewed_human(row: dict) -> bool:
    return row.get("deleted") is not True and row.get("synthetic") is False and row.get("review_result") is True and not row.get("model_name")

def load_archive(path: Path, *, verify_sha: bool=True) -> list[dict]:
    path=Path(path)
    if verify_sha and sha256_file(path)!=ARCHIVE_SHA256:
        raise ValueError("pinned OASST2 archive SHA-256 mismatch")
    rows=[]
    with gzip.open(path,"rt",encoding="utf8") as f:
        for n,line in enumerate(f,1):
            if not line.strip(): continue
            row=json.loads(line)
            if not isinstance(row,dict) or not isinstance(row.get("message_id"),str):
                raise ValueError(f"invalid OASST2 row at line {n}")
            rows.append(row)
    return rows

def indexes(rows: list[dict]):
    by_id={}; children={}
    for row in rows:
        mid=row["message_id"]
        if mid in by_id: raise ValueError(f"duplicate OASST2 message_id: {mid}")
        by_id[mid]=row
        pid=row.get("parent_id")
        if isinstance(pid,str): children.setdefault(pid,[]).append(row)
    return by_id,children

def path_to_root(mid: str, by_id: dict[str,dict], *, max_hops: int=64) -> list[dict]:
    out=[]; seen=set(); cur=mid
    while cur is not None:
        if cur in seen: raise ValueError("cycle in OASST2 message tree")
        seen.add(cur)
        row=by_id.get(cur)
        if row is None: raise ValueError(f"missing OASST2 parent/message: {cur}")
        out.append(row)
        cur=row.get("parent_id")
        if cur is not None and not isinstance(cur,str): raise ValueError("invalid parent_id")
        if len(out)>max_hops: raise ValueError("OASST2 path too deep")
    return list(reversed(out))

def valid_human_path(path: list[dict], *, end_role: str) -> bool:
    if not path or path[0].get("role")!="prompter" or path[-1].get("role")!=end_role:
        return False
    lang=path[0].get("lang")
    if lang not in {"de","en"}: return False
    expected="prompter"
    for row in path:
        if row.get("role")!=expected or row.get("lang")!=lang or not reviewed_human(row):
            return False
        text=row.get("text")
        if not isinstance(text,str) or not text.strip(): return False
        expected="assistant" if expected=="prompter" else "prompter"
    return True

def to_messages(path: list[dict]) -> list[dict[str,str]]:
    return [{"role":"user" if r["role"]=="prompter" else "assistant","content":r["text"].strip()} for r in path]

def split_for_root_prompt(text: str) -> str:
    n=normalize(text)
    h=hashlib.sha256(("SFT50:PROMPT_GROUP:v2:"+n).encode("utf8")).hexdigest()
    b=int(h[:8],16)%100
    return "holdout" if b<2 else "test" if b<5 else "val" if b<10 else "train"

def contaminated(texts: list[str], frozen: list[str]) -> bool:
    joined=normalize("\n".join(texts))
    return any(p in joined for p in frozen)
