"""NERAL Orbit v0 – local-only, human-controlled drafts for future agent work.

No timers, model inference, shell execution, auto PRs or cloud workers.
One bounded Wikipedia lookup can be manually started for an approved research task.
Approval alone NEVER launches the worker.
"""
from __future__ import annotations
import json
import re
from contextlib import contextmanager
import sqlite3
import threading
import uuid
from datetime import datetime, timezone, timedelta
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


def _plan(kind, state="DRAFT", *, has_report=False, has_error=False):
    plans={
        "coding":("Scope und Erfolgskriterien klären","Fehler mit echtem Test reproduzieren",
                  "Nur im isolierten Branch korrigieren und CI prüfen","Draft-PR zur menschlichen Review vorlegen"),
        "research":("Recherchefrage und öffentliche Quelle freigeben",
                    "Enzyklopädische Quellen abrufen",
                    "Inhalte und Aktualität unabhängig überprüfen",
                    "Quellenbericht zur Prüfung bereitstellen"),
        "monitor":("Beobachtetes Ereignis und Zeitfenster definieren","Erlaubte Datenquellen festlegen",
                   "Änderungserkennung mit Fehlalarmen testen","Manuelle Freigabe für laufenden Worker einholen"),
    }
    stages=["NOT_STARTED"]*4
    if kind=="research":
        if state=="ARCHIVED":
            # Preserve truthful history; archiving never silently passes review.
            state=("REVIEWED" if has_report else
                   "RESEARCH_FAILED" if has_error else "CANCELLED")
        if state in {"APPROVED_NO_EXECUTOR","RESEARCH_RUNNING",
                     "REVIEW_READY","REVIEWED","RESEARCH_FAILED"}:
            stages[0]="APPROVED"
        if state=="RESEARCH_RUNNING":
            stages[1]="IN_PROGRESS"
        elif state in {"REVIEW_READY","REVIEWED"}:
            stages[1]="COMPLETED"
            stages[2]="NEEDS_REVIEW"  # Reading is not independent fact-checking.
            stages[3]="COMPLETED"
        elif state=="RESEARCH_FAILED":
            stages[1]="FAILED"
    return [{"description":description,"status":stage}
            for description,stage in zip(plans[kind],stages)]


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
        "plan":_plan(row["kind"],state,has_report=report is not None,
                     has_error=error is not None), "report":report, "error":error,
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

    @contextmanager
    def _open(self):
        # sqlite3.Connection.__exit__ commits/rolls back but does not close.
        # On Windows a leaked connection locks orbit.sqlite3 and prevents
        # test cleanup, backup and future directory rotation.
        conn=sqlite3.connect(self.path,timeout=8)
        conn.row_factory=sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def list(self, *, archived=False, offset=0, limit=LIMIT):
        if type(archived) is not bool:
            raise OrbitError("Ungültiger Archiv-Filter")
        # Offset pagination is a read-only view. Values are always bound SQL
        # parameters; no arbitrary clauses or unbounded result windows.
        if type(offset) is not int or not 0 <= offset <= 100000:
            raise OrbitError("Ungültiger Archiv-Offset")
        if type(limit) is not int or not 1 <= limit <= LIMIT+1:
            raise OrbitError("Ungültige Seitengröße")
        if not archived and offset:
            raise OrbitError("Aktive Aufgaben erlauben keine Seitennavigation")
        clause="status = 'ARCHIVED'" if archived else "status != 'ARCHIVED'"
        with self._lock,self._open() as conn:
            return [_render(row) for row in conn.execute(
                f"""SELECT t.*,r.report_json,r.error_text FROM orbit_tasks t
                    LEFT JOIN orbit_research_reports r ON r.task_id=t.id
                    WHERE t.{clause} ORDER BY t.created_at DESC,t.id DESC LIMIT ? OFFSET ?""",
                (limit,offset))]

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
        # The current public research worker accepts only 200 characters.
        # Fail at creation, not after the user has approved the task.
        if kind=="research" and len(goal)>200:
            raise OrbitError("Wikipedia-Rechercheauftrag: maximal 200 Zeichen")
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
            seen_source_urls=set()
            for item in report["sources"]:
                if (not isinstance(item,dict) or
                        not isinstance(item.get("url"),str) or
                        not re.fullmatch(r"https://de\.wikipedia\.org/\?curid=[1-9][0-9]{0,15}",item["url"]) or
                        not isinstance(item.get("title"),str) or
                        not 1<=len(item["title"])<=180 or
                        not isinstance(item.get("excerpt"),str) or
                        not 1<=len(item["excerpt"])<=700):
                    raise OrbitError("Unsichere oder unvollständige Recherchequelle")
                if item["url"] in seen_source_urls:
                    raise OrbitError("Wikipedia-Quelle doppelt im Recherchebericht")
                seen_source_urls.add(item["url"])
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

    def reconcile_stale_research(self, *, min_age_seconds=60):
        """Mark orphaned one-shot searches as failed; never retry or fabricate.

        A lookup has a strict six-second network timeout. Use a generous
        grace period to avoid interfering with a concurrently active request.
        Intended for app startup; does not contact Wikimedia or schedule jobs.
        """
        if type(min_age_seconds) is not int or not 30 <= min_age_seconds <= 3600:
            raise OrbitError("Ungültige Wiederherstellungsfrist")
        cutoff=(datetime.now(timezone.utc)-timedelta(
            seconds=min_age_seconds)).isoformat(timespec="seconds")
        with self._lock,self._open() as conn:
            conn.execute("BEGIN IMMEDIATE")
            rows=conn.execute(
                """SELECT id FROM orbit_tasks
                   WHERE kind='research' AND status='RESEARCH_RUNNING'
                     AND updated_at<?""",(cutoff,)).fetchall()
            for row in rows:
                conn.execute(
                    """INSERT INTO orbit_research_reports(task_id,report_json,error_text)
                       VALUES(?,NULL,?)
                       ON CONFLICT(task_id) DO UPDATE SET
                           report_json=NULL, error_text=excluded.error_text""",
                    (row["id"],"Recherche unterbrochen. Kein Ergebnis; neuer Auftrag nötig."))
                conn.execute(
                    """UPDATE orbit_tasks SET status='RESEARCH_FAILED',
                       revision=revision+1,updated_at=? WHERE id=?""",
                    (_now(),row["id"]))
        return len(rows)
