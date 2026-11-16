"""Explicit opt-in, scope-separated SQLite memory; never automatically infers facts."""
from __future__ import annotations
from datetime import datetime, timezone
from pathlib import Path
import re
import sqlite3


def _utc(): return datetime.now(timezone.utc).isoformat()

class LocalMemory:
    def __init__(self, path: str | Path = "neral_memory.db"):
        self.db = sqlite3.connect(str(path))
        self.db.row_factory=sqlite3.Row
        self.db.execute("""CREATE TABLE IF NOT EXISTS memories(
          scope TEXT NOT NULL, key TEXT NOT NULL, value TEXT NOT NULL,
          kind TEXT NOT NULL, updated TEXT NOT NULL,
          PRIMARY KEY(scope,key))""")
        self.db.commit()

    def upsert(self, scope:str, key:str, value:str, *, kind:str="fact", consent:bool=False, timestamp:str|None=None)->bool:
        if not consent: raise PermissionError("explicit memory consent required")
        if kind not in ("fact","project","preference") or not all(isinstance(s,str) and 1<=len(s)<=2048 for s in (scope,key,value)):
            raise ValueError("invalid memory")
        if len(scope)>80 or len(key)>120: raise ValueError("key too long")
        stamp=timestamp or _utc()
        try:
            datetime.fromisoformat(stamp)
        except ValueError as exc: raise ValueError("invalid time") from exc
        with self.db:
            old=self.db.execute("SELECT updated FROM memories WHERE scope=? AND key=?",(scope,key)).fetchone()
            if old and old["updated"]>stamp: return False
            self.db.execute("""INSERT INTO memories(scope,key,value,kind,updated) VALUES(?,?,?,?,?)
              ON CONFLICT(scope,key) DO UPDATE SET value=excluded.value,kind=excluded.kind,updated=excluded.updated""",
              (scope,key,value,kind,stamp))
        return True

    def forget(self, scope:str, key:str)->bool:
        with self.db:
            result=self.db.execute("DELETE FROM memories WHERE scope=? AND key=?",(scope,key))
        return result.rowcount>0

    def list(self, scope:str="user")->list[dict]:
        return [dict(r) for r in self.db.execute("SELECT * FROM memories WHERE scope=? ORDER BY updated DESC",(scope,))]

    def context(self, query:str, *, scope:str="user", max_tokens:int=240)->str:
        if type(max_tokens) is not int or not 1<=max_tokens<=2000: raise ValueError("token budget")
        query_terms=set(re.findall(r"\w+",query.lower()))
        ranked=[]
        for item in self.list(scope):
            terms=set(re.findall(r"\w+",(item["key"]+" "+item["value"]).lower()))
            score=len(terms & query_terms)*5 + (2 if item["kind"]=="project" else 0)
            ranked.append((score,item["updated"],item))
        ranked.sort(key=lambda x:(x[0],x[1]), reverse=True)
        budget=max_tokens*3  # conservative approximation, not tokenizer guarantee
        lines=[]
        for _,_,item in ranked:
            line=f'- {item["kind"]}: {item["key"]} = {item["value"]}'
            if len(line)+1<=budget:
                lines.append(line);budget-=len(line)+1
        return "\n".join(lines)

    def close(self): self.db.close()
