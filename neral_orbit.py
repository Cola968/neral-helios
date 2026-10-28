"""NERAL Orbit v0 – local-only, human-controlled drafts for future agent work.

No timers, network, model inference, code execution, auto PRs or cloud workers.
Approval records the operator's intent but NEVER starts execution.
"""
from __future__ import annotations
import json
import re
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

CATEGORIES=("coding","research","monitor")
LIMIT=40
_MAX_GOAL=1200


class OrbitError(ValueError):
    """Invalid request, bad revision or unsafe status transition."""


class OrbitConflict(OrbitError):
    """Optimistic-lock mismatch."""


def _content(value,minimum,maximum,field):
    if not isinstance(value,str):
        raise OrbitError(f"{field}: Text erwartet")
    value=value.strip()
    if not minimum<=len(value)<=maximum or any(
        ord(char)<32 or ord(char)==127 for char in value):
        raise OrbitError(f"{field}: ungültige Länge oder Steuerzeichen")
    return value


def _plan(kind):
    plans={
        "coding":("Scope und Erfolgskriterien klären","Fehler mit echtem Test reproduzieren",
                  "Nur im isolierten Branch korrigieren und CI prüfen","Draft-PR zur menschlichen Review vorlegen"),
        "research":("Frage und vertrauenswürdige Quellen festlegen","Quellen prüfen und Belege sammeln",
                    "Behauptungen und Aktualität unabhängig prüfen","Bericht zur Review vorbereiten"),
        "monitor":("Beobachtetes Ereignis und Zeitfenster definieren","Erlaubte Datenquellen festlegen",
                   "Änderungserkennung mit Fehlalarmen testen","Manuelle Freigabe für laufenden Worker einholen"),
    }
    return [{"description":x,"status":"NOT_STARTED"} for x in plans[kind]]


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _render(row):
    return {
        "id":row["id"],"kind":row["kind"],"title":row["title"],
        "goal":row["goal"],"status":row["status"],"revision":row["revision"],
        "created_at":row["created_at"],"updated_at":row["updated_at"],
        "plan":_plan(row["kind"]),
        "execution_enabled":False,
        "executor":"none",
        "budget_eur":0,
        "message":"Dies ist ein geplanter Auftrag. Noch kein Cloud-Worker, keine automatische Ausführung.",
    }


class OrbitStore:
    """SQLite writes serialize under one thread lock and DB BEGIN IMMEDIATE."""
    def __init__(self,path):
        self.path=Path(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self._lock=threading.RLock()
        with self._open() as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS orbit_tasks (
                id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL,
                goal TEXT NOT NULL, status TEXT NOT NULL, revision INTEGER NOT NULL,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")

    def _open(self):
        conn=sqlite3.connect(self.path,timeout=8)
        conn.row_factory=sqlite3.Row
        return conn

    def list(self):
        with self._lock,self._open() as conn:
            return [_render(row) for row in conn.execute(
                "SELECT * FROM orbit_tasks ORDER BY created_at DESC,id DESC LIMIT ?",(LIMIT,))]

    def get(self,task_id):
        if not isinstance(task_id,str) or not re.fullmatch(r"[0-9a-f]{32}",task_id):
            raise OrbitError("Unbekannter Auftrag")
        with self._lock,self._open() as conn:
            row=conn.execute("SELECT * FROM orbit_tasks WHERE id=?",(task_id,)).fetchone()
        if row is None:raise OrbitError("Unbekannter Auftrag")
        return _render(row)

    def create(self,*,kind,title,goal):
        if kind not in CATEGORIES or type(kind) is not str:
            raise OrbitError("Unbekannter Auftragstyp")
        title=_content(title,3,120,"Titel")
        goal=_content(goal,8,_MAX_GOAL,"Auftrag")
        ident=uuid.uuid4().hex
        at=_now()
        with self._lock,self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT count(*) FROM orbit_tasks").fetchone()[0]>=LIMIT:
                raise OrbitError("Maximal 40 Aufträge; alte Aufgaben zuerst löschen (in v0 kein Löschen verfügbar)")
            conn.execute("INSERT INTO orbit_tasks VALUES (?,?,?,?,?,?,?,?)",
                         (ident,kind,title,goal,"DRAFT",1,at,at))
        return self.get(ident)

    def transition(self,task_id,action,*,revision):
        if type(revision) is not int or revision<1:
            raise OrbitError("Revision muss eine positive ganze Zahl sein")
        if action not in ("approve","cancel"):
            raise OrbitError("Nur explizites Freigeben und Abbrechen sind erlaubt")
        if not isinstance(task_id,str) or not re.fullmatch(r"[0-9a-f]{32}",task_id):
            raise OrbitError("Unbekannter Auftrag")
        with self._lock,self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row=conn.execute("SELECT * FROM orbit_tasks WHERE id=?",(task_id,)).fetchone()
            if row is None:raise OrbitError("Unbekannter Auftrag")
            if row["revision"]!=revision:raise OrbitConflict("Auftrag wurde bereits geändert. Bitte neu laden.")
            if action=="approve" and row["status"]=="DRAFT":
                status="APPROVED_NO_EXECUTOR"
            elif action=="cancel" and row["status"] in ("DRAFT","APPROVED_NO_EXECUTOR"):
                status="CANCELLED"
            else:raise OrbitError("Unzulässiger Statuswechsel")
            conn.execute("UPDATE orbit_tasks SET status=?, revision=revision+1, updated_at=? WHERE id=?",
                         (status,_now(),task_id))
        return self.get(task_id)
