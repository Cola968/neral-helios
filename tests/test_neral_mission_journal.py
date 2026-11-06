"""CPU-only regression tests for opt-in NERAL mission history."""
from pathlib import Path
import os
import sqlite3
import tempfile
import unittest
from neral_mission_journal import MissionJournal


def state(number, status="running", steps=0, **extra):
    return {"id": f"{number:024x}", "status": status, "steps": steps, **extra}


class MissionJournalTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.path = Path(self.dir.name) / "private" / "mission-history.sqlite3"

    def test_no_sensitive_content_is_stored(self):
        journal = MissionJournal(self.path)
        journal.record(state(1, "completed", 3, goal="SECRET-PROMPT",
                             answer="SECRET-ANSWER", history=[{"observation": "SECRET-FILE"}],
                             proposal={"token": "SECRET-APPROVAL"}))
        self.assertEqual(len(journal.list_recent()), 1)
        with sqlite3.connect(self.path) as conn:
            columns = {row[1] for row in conn.execute("PRAGMA table_info(missions)")}
        self.assertEqual(columns, {"id", "status", "steps", "updated_at_ms"})
        for secret in (b"SECRET-PROMPT", b"SECRET-ANSWER", b"SECRET-FILE", b"SECRET-APPROVAL"):
            self.assertNotIn(secret, self.path.read_bytes())
        if os.name == "posix":
            self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_survives_restart_and_marks_active_as_interrupted(self):
        journal = MissionJournal(self.path)
        journal.record(state(1, "running", 1))
        journal.record(state(2, "paused", 2))
        journal.record(state(3, "awaiting_approval", 3))
        journal.record(state(4, "completed", 4))
        reopened = MissionJournal(self.path)
        by_id = {x["id"]: x for x in reopened.list_recent()}
        for n in (1, 2, 3):
            self.assertEqual(by_id[f"{n:024x}"]["status"], "interrupted")
        self.assertEqual(by_id[f"{4:024x}"]["status"], "completed")

    def test_upsert_and_bounded_retention(self):
        journal = MissionJournal(self.path, max_entries=2)
        journal.record(state(1, "running", 0))
        journal.record(state(1, "completed", 2))
        self.assertEqual(len(journal.list_recent()), 1)
        journal.record(state(2, "cancelled", 0))
        journal.record(state(3, "failed", 1))
        self.assertEqual({x["id"] for x in journal.list_recent()},
                         {f"{2:024x}", f"{3:024x}"})

    def test_invalid_records_are_rejected(self):
        journal = MissionJournal(self.path)
        for invalid in (state(1, "invented"), state(1, "running", -1),
                        state(1, "running", True),
                        {"id": "../escape", "status": "running", "steps": 1}):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                journal.record(invalid)
        with self.assertRaises(ValueError):
            journal.list_recent(limit=0)

    def test_symlink_database_is_rejected(self):
        source = Path(self.dir.name) / "source.sqlite3"
        source.write_bytes(b"other")
        link = Path(self.dir.name) / "symlink.sqlite3"
        try:
            link.symlink_to(source)
        except (OSError, NotImplementedError):
            self.skipTest("Symlinks unsupported")
        with self.assertRaises(ValueError):
            MissionJournal(link)


if __name__ == "__main__":
    unittest.main()
