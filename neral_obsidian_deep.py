"""CPU-only Obsidian coordination. No model/GPU imports or prompt storage."""
from contextvars import ContextVar
import os
import re
import threading
import time

_PROGRESS = ContextVar("obsidian_progress", default=None)
_SEARCH = ContextVar("obsidian_search", default=None)
_STAGES = {"plan":"Zerlege Aufgabe…", "wiki":"Suche im lokalen Wissen…",
           "web":"Prüfe Webquellen…", "tool":"Prüfe mit Werkzeug…",
           "critic":"Prüfe Antwort auf Vollständigkeit…", "proof":"Unabhängigen Beweis geprüft."}
_TOOLS = {"python_sandbox":"Python", "python_code_sandbox":"Python-Code", "z3_solver":"Z3", "direct_reasoning":"Quellen/Antwort"}

def enabled():
    return os.environ.get("NERAL_V42_OBSIDIAN_DEEP", "1") == "1"

_COMPLEX_INTENT = re.compile(
    r"(?iu)\b(?:berechn\w*|rechne|rechnung\w*|mathematik|logik|z3|"
    r"programmier\w*|python|code|funktion\w*|algorithmus|debug\w*|"
    r"schicht\w*|zuordn\w*|vergleich\w*|vergleiche|analysier\w*|"
    r"recherch\w*|quelle\w*|beleg\w*|zitat\w*|web\w*|wiki\w*|"
    r"aktuell\w*|heute|gestern|morgen|neueste\w*|jahr\w*|datum|"
    r"statistik\w*|beweis\w*|mehrstufig\w*|zuerst|danach|"
    r"mehrere\s+(?:schritte|quellen|artikel)|"
    r"wann|wer|wo|wieviel|wie\s+viel|wie\s+hoch)\b"
)
_SIMPLE_EXPLANATION = re.compile(
    r"(?iu)(?:^|[?!.]\s+)(?:warum|wieso|weshalb|erkläre|erklaere|"
    r"wie\s+funktioniert|was\s+bedeutet|was\s+ist)\b"
)


def is_simple_everyday_question(question):
    """Fail closed: only brief, timeless, non-numeric explanation requests."""
    if type(question) is not str or not 4 <= len(question.strip()) <= 280:
        return False
    return bool(_SIMPLE_EXPLANATION.search(question)
                and not _COMPLEX_INTENT.search(question)
                and not re.search(r"[0-9€$%]|[+×÷=<>]|https?://|[`{}]", question)
                and question.count("?") <= 1)


def active(tier):
    return enabled() and tier in ("Obsidian", "Helios", "helios")

def constrained_query(original, proposed):
    """Prevent model-derived or private new terms entering a web request."""
    if not isinstance(original,str):
        return ""
    if (not isinstance(proposed,str) or not 2 <= len(proposed.strip()) <= 160
            or not re.fullmatch(r"[\w\s.,?()\-]+",proposed)
            or not all(ch.isprintable() for ch in proposed)):
        return original
    words = lambda s: re.findall(r"\w+",s.casefold())
    return proposed.strip() if words(proposed) and set(words(proposed)) <= set(words(original)) else original

def emit(callback,stage,attempt=1,tool=None):
    if callable(callback):
        callback({"stage":stage,"attempt":attempt,"tool":tool})

class ProgressRegistry:
    """Bounded session-scoped ephemeral structured status; no raw prompts."""
    def __init__(self,clock=time.monotonic):
        self._clock=clock
        self._lock=threading.Lock()
        self._rows={}

    def _prune(self):
        for key in list(self._rows):
            if self._rows[key]["expires"] <= self._clock():
                del self._rows[key]

    def begin(self,owner,request_id):
        if not owner or not isinstance(request_id,str) or not re.fullmatch(r"[0-9a-f]{32}",request_id):
            return False
        with self._lock:
            self._prune()
            key=(owner,request_id)
            if key in self._rows or len(self._rows)>=32:
                return False
            self._rows[key]={"events":[],"done":False,"expires":self._clock()+600}
            return True

    def emit(self,owner,request_id,event):
        stage,attempt,tool=event.get("stage"),event.get("attempt"),event.get("tool")
        if stage not in _STAGES or type(attempt) is not int or not 1<=attempt<=3:
            return
        if tool not in _TOOLS:
            tool=None
        message=_STAGES[stage]
        if stage=="tool" and tool:
            message="Prüfe mit "+_TOOLS[tool]+"…"
        if stage!="proof":
            message+=f" (Versuch {attempt}/3)"
        with self._lock:
            row=self._rows.get((owner,request_id))
            if row is None or row["done"] or len(row["events"])>=32:
                return
            row["events"].append({"stage":stage,"attempt":attempt,"tool":tool,
                                  "message":message,"seq":len(row["events"])+1})

    def read(self,owner,request_id):
        with self._lock:
            self._prune()
            row=self._rows.get((owner,request_id))
            return None if row is None else {"events":[dict(e) for e in row["events"]],"done":row["done"]}

    def finish(self,owner,request_id):
        with self._lock:
            row=self._rows.get((owner,request_id))
            if row is not None:
                row["done"]=True
                row["expires"]=self._clock()+60
