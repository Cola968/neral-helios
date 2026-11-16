"""CPU Markdown/text/code chunking and persistent hybrid BM25 + embedding retrieval."""
from __future__ import annotations
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Callable, Sequence

_TOKEN=re.compile(r"\w+",re.UNICODE)

def chunk_text(text: str, *, size:int=700, overlap:int=100)->list[str]:
    if type(text) is not str or not 100<=size<=4000 or not 0<=overlap<size: raise ValueError("invalid chunks")
    if not text.strip(): return []
    chunks=[]
    start=0
    while start<len(text):
        end=min(len(text),start+size)
        if end<len(text):
            split=max(text.rfind("\n\n",start+size//2,end),text.rfind("\n",start+size//2,end))
            if split>start: end=split+1
        chunks.append(text[start:end])
        if end==len(text): break
        start=max(start+1,end-overlap)
    return chunks

def _terms(text): return [w.lower() for w in _TOKEN.findall(text)]
def _cosine(a,b):
    if not a or not b or len(a)!=len(b): return 0.0
    norm=math.sqrt(sum(x*x for x in a)*sum(y*y for y in b))
    return sum(x*y for x,y in zip(a,b))/norm if norm else 0.0

class LocalRAG:
    def __init__(self,path:str|Path="neral_rag.db", embedding_fn:Callable[[str],Sequence[float]]|None=None):
        self.db=sqlite3.connect(str(path))
        self.db.row_factory=sqlite3.Row
        self.embedding_fn=embedding_fn
        self.db.execute("""CREATE TABLE IF NOT EXISTS chunks(
          doc_id TEXT NOT NULL, chunk_id INTEGER NOT NULL, source TEXT NOT NULL,
          title TEXT NOT NULL, content TEXT NOT NULL, metadata TEXT NOT NULL,
          vector TEXT, PRIMARY KEY(doc_id,chunk_id))""")
        self.db.commit()

    def add(self, doc_id:str, text:str, *, source:str, title:str="", metadata:dict|None=None)->int:
        if not 1<=len(doc_id)<=100 or not 1<=len(source)<=1000: raise ValueError("document metadata")
        if not isinstance(metadata or {},dict): raise ValueError("metadata")
        chunks=chunk_text(text)
        with self.db:
            self.db.execute("DELETE FROM chunks WHERE doc_id=?",(doc_id,))
            for i,content in enumerate(chunks):
                vector=list(self.embedding_fn(content)) if self.embedding_fn else None
                self.db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?)",
                    (doc_id,i,source,title,content,json.dumps(metadata or {},ensure_ascii=False),json.dumps(vector) if vector is not None else None))
        return len(chunks)

    def search(self,query:str, limit:int=5, alpha:float=.75)->list[dict]:
        if not isinstance(query,str) or not 1<=limit<=20 or not 0<=alpha<=1: raise ValueError("invalid search")
        rows=self.db.execute("SELECT * FROM chunks").fetchall()
        if not rows: return []
        terms=_terms(query)
        lengths=[len(_terms(r["content"])) for r in rows]
        avg=max(1,sum(lengths)/len(rows))
        df={t:sum(t in set(_terms(r["content"])) for r in rows) for t in set(terms)}
        qv=list(self.embedding_fn(query)) if self.embedding_fn else None
        result=[]
        for r,length in zip(rows,lengths):
            words=_terms(r["content"])
            tf={t:words.count(t) for t in df}
            bm25=sum(math.log(1+(len(rows)-df[t]+.5)/(df[t]+.5))*(tf[t]*2.2)/(tf[t]+1.2*(.25+.75*length/avg)) for t in df if tf[t])
            vector=json.loads(r["vector"]) if r["vector"] else None
            vec=_cosine(qv,vector) if qv is not None and vector is not None else 0
            score=alpha*bm25 + (1-alpha)*max(0,vec)
            if score>0:
                result.append({"score":score,"doc_id":r["doc_id"],"chunk_id":r["chunk_id"],
                  "source":r["source"],"title":r["title"],"text":r["content"],
                  "metadata":json.loads(r["metadata"]),"citation":f'[{r["doc_id"]}#c{r["chunk_id"]}]'})
        return sorted(result,key=lambda v:v["score"],reverse=True)[:limit]

    def remove(self,doc_id:str):
        with self.db: self.db.execute("DELETE FROM chunks WHERE doc_id=?",(doc_id,))

    def close(self): self.db.close()
