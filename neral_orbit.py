"""NERAL Orbit v0 – local-only, human-controlled drafts for future agent work.

No timers, model inference, shell execution, auto PRs or cloud workers.
One bounded Wikipedia lookup can be manually started for an approved research task.
Approval alone NEVER launches the worker.
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
    state=row["status"]
    report=json.loads(row["report_json"]) if "report_json" in row.keys() and row["report_json"] else None
    error=row["error_text"] if "error_text" in row.keys() else None
    once=row["kind"]=="research" and state in ("RESEARCH_RUNNING","REVIEW_READY","RESEARCH_FAILED","REVIEWED")
    return {
        "id":row["id"],"kind":row["kind"],"title":row["title"],
        "goal":row["goal"],"status":state,"revision":row["revision"],
        "created_at":row["created_at"],"updated_at":row["updated_at"],
        "plan":_plan(row["kind"]), "report":report, "error":error,
        "execution_enabled":state=="RESEARCH_RUNNING",
        "executor":"wikipedia_research_once" if once else "none",
        "budget_eur":0,
        "message":("Manuell gestartete einmalige Recherche; keine Hintergrundüberwachung."
                   if once else "Freigabe allein startet keinen Worker."),
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
            conn.execute("""CREATE TABLE IF NOT EXISTS orbit_research_reports (
                task_id TEXT PRIMARY KEY, report_json TEXT, error_text TEXT,
                FOREIGN KEY(task_id) REFERENCES orbit_tasks(id)
            )""")

    def _open(self):
        conn=sqlite3.connect(self.path,timeout=8)
        conn.row_factory=sqlite3.Row
        return conn

    def list(self, *, archived=False):
        if type(archived) is not bool:
            raise OrbitError("Ungültiger Archiv-Filter")
        clause="status = 'ARCHIVED'" if archived else "status != 'ARCHIVED'"
        with self._lock,self._open() as conn:
            return [_render(row) for row in conn.execute(
                f"""SELECT t.*,r.report_json,r.error_text FROM orbit_tasks t
                    LEFT JOIN orbit_research_reports r ON r.task_id=t.id
                    WHERE t.{clause} ORDER BY t.created_at DESC,t.id DESC LIMIT ?""",(LIMIT,))]

    def get(self,task_id):
        if not isinstance(task_id,str) or not re.fullmatch(r"[0-9a-f]{32}",task_id):
            raise OrbitError("Unbekannter Auftrag")
        with self._lock,self._open() as conn:
            row=conn.execute("""SELECT t.*,r.report_json,r.error_text FROM orbit_tasks t
                                 LEFT JOIN orbit_research_reports r ON r.task_id=t.id
                                 WHERE t.id=?""",(task_id,)).fetchone()
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
            if conn.execute("SELECT count(*) FROM orbit_tasks WHERE status != 'ARCHIVED'").fetchone()[0]>=LIMIT:
                raise OrbitError("Maximal 40 aktive Aufträge; abgebrochene Aufgaben zuerst archivieren.")
            conn.execute("INSERT INTO orbit_tasks VALUES (?,?,?,?,?,?,?,?)",
                         (ident,kind,title,goal,"DRAFT",1,at,at))
        return self.get(ident)

    def transition(self,task_id,action,*,revision):
        if type(revision) is not int or revision<1:
            raise OrbitError("Revision muss eine positive ganze Zahl sein")
        if action not in ("approve","cancel","archive","review"):
            raise OrbitError("Nur explizites Freigeben, Abbrechen, Prüfen und Archivieren sind erlaubt")
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
            elif action=="review" and row["status"]=="REVIEW_READY":
                # Seen by operator, not independently fact checked.
                status="REVIEWED"
            elif action=="archive" and row["status"] in ("CANCELLED","REVIEWED","RESEARCH_FAILED"):
                # Preserve findings, never remove SQLite task history.
                status="ARCHIVED"
            else:raise OrbitError("Unzulässiger Statuswechsel")
            conn.execute("UPDATE orbit_tasks SET status=?, revision=revision+1, updated_at=? WHERE id=?",
                         (status,_now(),task_id))
        return self.get(task_id)

    def begin_research(self,task_id,*,revision):
        """Atomically reserve ONE approved research task; never start by approval alone."""
        if type(revision) is not int or revision<1:
            raise OrbitError("Revision muss eine positive ganze Zahl sein")
        if not isinstance(task_id,str) or not re.fullmatch(r"[0-9a-f]{32}",task_id):
            raise OrbitError("Unbekannter Auftrag")
        with self._lock,self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row=conn.execute("SELECT kind,status,revision FROM orbit_tasks WHERE id=?",(task_id,)).fetchone()
            if row is None:raise OrbitError("Unbekannter Auftrag")
            if row["revision"]!=revision:raise OrbitConflict("Auftrag wurde bereits geändert. Bitte neu laden.")
            if row["kind"]!="research" or row["status"]!="APPROVED_NO_EXECUTOR":
                raise OrbitError("Nur einmalig freigegebene Rechercheaufträge dürfen gestartet werden")
            conn.execute("""UPDATE orbit_tasks
                         SET status='RESEARCH_RUNNING',revision=revision+1,updated_at=?
                         WHERE id=?""",(_now(),task_id))
        return self.get(task_id)

    def finish_research(self,task_id,*,revision,report=None,error=None):
        """Save sourced results or bounded failure, only for a reserved run."""
        if type(revision) is not int or revision<1:
            raise OrbitError("Revision muss eine positive ganze Zahl sein")
        if (report is None)==(error is None):
            raise OrbitError("Genau ein Recherchebericht oder Fehler ist erforderlich")
        if report is not None:
            if (not isinstance(report,dict) or
                    report.get("provider")!="Wikipedia (de)" or
                    report.get("model_used") is not False or
                    report.get("requires_review") is not True or
                    not isinstance(report.get("sources"),list) or
                    not 1<=len(report["sources"])<=3 or
                    report.get("source_count")!=len(report["sources"])):
                raise OrbitError("Recherchebericht verletzt Quellen- und Größenvertrag")
            for item in report["sources"]:
                if (not isinstance(item,dict) or
                        not isinstance(item.get("url"),str) or
                        not re.fullmatch(r"https://de\.wikipedia\.org/\?curid=[1-9][0-9]{0,15}",item["url"]) or
                        not isinstance(item.get("title"),str) or
                        not 1<=len(item["title"])<=180 or
                        not isinstance(item.get("excerpt"),str) or
                        not 1<=len(item["excerpt"])<=700):
                    raise OrbitError("Unsichere oder unvollständige Recherchequelle")
            report_json=json.dumps(report,ensure_ascii=False)
            if len(report_json.encode("utf8"))>10000:
                raise OrbitError("Recherchebericht überschreitet das Speicherlimit")
            state="REVIEW_READY"
        else:
            error=_content(error,3,160,"Recherchefehler")
            report_json=None
            state="RESEARCH_FAILED"
        if not isinstance(task_id,str) or not re.fullmatch(r"[0-9a-f]{32}",task_id):
            raise OrbitError("Unbekannter Auftrag")
        with self._lock,self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row=conn.execute("SELECT status,revision FROM orbit_tasks WHERE id=?",(task_id,)).fetchone()
            if row is None:raise OrbitError("Unbekannter Auftrag")
            if row["revision"]!=revision:raise OrbitConflict("Auftrag wurde bereits geändert. Bitte neu laden.")
            if row["status"]!="RESEARCH_RUNNING":raise OrbitError("Kein laufender Rechercheauftrag")
            conn.execute("INSERT INTO orbit_research_reports(task_id,report_json,error_text) VALUES (?,?,?)",
                         (task_id,report_json,error))
            conn.execute("UPDATE orbit_tasks SET status=?,revision=revision+1,updated_at=? WHERE id=?",
                         (state,_now(),task_id))
        return self.get(task_id)
