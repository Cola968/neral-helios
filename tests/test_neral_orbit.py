"""NERAL Orbit v0 real persistent task drafts; NO unattended executor."""
import sqlite3
import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from neral_orbit import OrbitStore, OrbitError


class OrbitStorageTests(unittest.TestCase):
    def setUp(self):
        self.dir=TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.db=Path(self.dir.name)/"orbit.sqlite3"
        self.store=OrbitStore(self.db)

    def test_draft_is_persistent_and_has_a_readable_4_stage_plan(self):
        t=self.store.create(kind="coding",title="NERAL verbessern",
                            goal="Behebe den Fehler und erstelle einen Draft-PR.")
        self.assertEqual(t["status"],"DRAFT")
        self.assertEqual(t["revision"],1)
        self.assertEqual(t["execution_enabled"],False)
        self.assertEqual(len(t["plan"]),4)
        self.assertEqual(t["plan"][0]["status"],"NOT_STARTED")
        self.assertEqual(t["executor"],"none")
        self.assertEqual(OrbitStore(self.db).list(),[t])
        self.assertNotIn("model_reply",str(t))

    def test_approval_is_recorded_but_never_starts_a_worker(self):
        t=self.store.create(kind="research",title="BER recherchieren",
                            goal="Prüfe jede Woche neue Langstreckenverbindungen.")
        approved=self.store.transition(t["id"],"approve",revision=1)
        self.assertEqual(approved["status"],"APPROVED_NO_EXECUTOR")
        self.assertEqual(approved["execution_enabled"],False)
        self.assertEqual(approved["executor"],"none")
        self.assertEqual(approved["plan"][0]["status"],"NOT_STARTED")
        self.assertEqual(approved["revision"],2)
        with self.assertRaises(OrbitError):
            self.store.transition(t["id"],"approve",revision=2)
        cancelled=self.store.transition(t["id"],"cancel",revision=2)
        self.assertEqual(cancelled["status"],"CANCELLED")
        with self.assertRaises(OrbitError):
            self.store.transition(t["id"],"approve",revision=3)

    def test_fail_closed_revision_validation_and_no_external_execution(self):
        t=self.store.create(kind="monitor",title="Flüge beobachten",
                            goal="Überwache neue Strecken am Berliner Flughafen.")
        for action in ("run","execute","complete","merge","deploy","schedule"):
            with self.subTest(action=action),self.assertRaises(OrbitError):
                self.store.transition(t["id"],action,revision=1)
        for revision in (True,0,"1",2,None):
            with self.subTest(revision=revision),self.assertRaises(OrbitError):
                self.store.transition(t["id"],"approve",revision=revision)
        self.assertEqual(self.store.get(t["id"])["status"],"DRAFT")
        with self.assertRaises(OrbitError):
            self.store.transition("not-a-uuid","approve",revision=1)

    def test_constrained_untrusted_titles_types_and_count(self):
        for kind,title,goal in (
            ("terminal","Build","Do anything."),
            ("coding"," ","Fix this bug."),
            ("coding","OK","\x00rm -rf"),
            ("coding","X"*125,"Improve."),
            ("coding","Task","a"*1201),
        ):
            with self.subTest(kind=kind,title=title[:10]),self.assertRaises(OrbitError):
                self.store.create(kind=kind,title=title,goal=goal)
        with self.assertRaises(OrbitError):
            self.store.create(kind="coding",title="A",goal="Do a real task")
        for i in range(40):
            self.store.create(kind="coding",title=f"Task {i:02d}",goal="Analyse, test and create a reviewed PR.")
        with self.assertRaises(OrbitError):
            self.store.create(kind="coding",title="Extra",goal="Cannot create unlimited tasks.")
        self.assertEqual(len(self.store.list()),40)

    def test_no_new_tables_are_exec_or_scheduler_credentials(self):
        with sqlite3.connect(self.db) as conn:
            ddl="\n".join(row[0] for row in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table'"))
        self.assertNotIn("access_token",ddl)
        self.assertNotIn("cron",ddl)
        self.assertNotIn("exec_command",ddl)


class OrbitFlaskTests(unittest.TestCase):
    def setUp(self):
        import app
        self.app=app
        self.dir=TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.original=(app.orbit_store,app.ORBIT_ENABLED)
        self.addCleanup(self.restore)
        app.orbit_store=OrbitStore(Path(self.dir.name)/"orbit.sqlite3")
        app.ORBIT_ENABLED=True
        app._hits.clear()
        self.client=app.app.test_client()
        self.assertEqual(self.client.get("/").status_code,200)

    def restore(self):
        self.app.orbit_store,self.app.ORBIT_ENABLED=self.original

    def test_real_cookie_guard_create_list_approve_cancel(self):
        payload={"kind":"coding","title":"Neues NERAL Feature",
                 "goal":"Erstelle zuerst einen Test, dann eine sichere Änderung."}
        raw=self.app.app.test_client().post("/api/orbit/tasks",json=payload)
        self.assertEqual(raw.status_code,401)
        created=self.client.post("/api/orbit/tasks",json=payload)
        self.assertEqual(created.status_code,201)
        task=created.json["task"]
        self.assertEqual(task["status"],"DRAFT")
        self.assertEqual(self.client.get("/api/orbit/tasks").json["tasks"],[task])
        done=self.client.post(f'/api/orbit/tasks/{task["id"]}/transition',
                              json={"action":"approve","revision":1})
        self.assertEqual(done.status_code,200)
        self.assertEqual(done.json["task"]["status"],"APPROVED_NO_EXECUTOR")
        self.assertFalse(done.json["task"]["execution_enabled"])
        self.assertEqual(self.client.post(
            f'/api/orbit/tasks/{task["id"]}/transition',
            json={"action":"approve","revision":1}).status_code,409)
        cancelled=self.client.post(f'/api/orbit/tasks/{task["id"]}/transition',
                              json={"action":"cancel","revision":2})
        self.assertEqual(cancelled.json["task"]["status"],"CANCELLED")

    def test_no_execution_endpoint_and_disabled_feature_404(self):
        self.assertEqual(self.client.get("/orbit").status_code,200)
        self.assertIn(b"NERAL Orbit",self.client.get("/orbit").data)
        self.assertEqual(self.client.post("/api/orbit/execute",json={}).status_code,404)
        self.app.ORBIT_ENABLED=False
        self.assertEqual(self.client.get("/api/orbit/tasks").status_code,404)
        self.assertEqual(self.client.post("/api/orbit/tasks",json={}).status_code,404)
        self.assertEqual(self.client.get("/orbit").status_code,200)


if __name__=="__main__":
    unittest.main()
