"""Opt-in local metadata-only mission history for NERAL.

No prompts, answers, observations, paths, patch contents or approval tokens
are ever written here. This is history, not a mission resumption engine.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import os
import re
import sqlite3
import time

STATES = frozenset({"running", "paused", "awaiting_approval", "completed", "failed",
                    "limit_reached", "cancelled", "interrupted"})
ACTIVE_STATES = ("running", "paused", "awaiting_approval")
ID_PATTERN = re.compile(r"[a-f0-9]{24}\Z")


class MissionJournal:
    """Single-owner history with a strict no-content database schema."""

    def __init__(self, path: str | Path, *, max_entries: int = 200):
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            raise ValueError("Mission history path must be absolute.")
        if self.path.is_symlink() or (self.path.exists() and not self.path.is_file()):
            raise ValueError("Mission history path must be a regular file.")
        if not 1 <= max_entries <= 1000:
            raise ValueError("Invalid mission history capacity.")
        self.max_entries = max_entries
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with closing(sqlite3.connect(self.path, timeout=5)) as conn:
            with conn:
                conn.execute("""CREATE TABLE IF NOT EXISTS missions (
                    id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    steps INTEGER NOT NULL, updated_at_ms INTEGER NOT NULL
                )""")
                # A prior process cannot safely be resumed.
                conn.execute("UPDATE missions SET status='interrupted' WHERE status IN (?,?,?)",
                             ACTIVE_STATES)
        if os.name == "posix":
            os.chmod(self.path, 0o600)

    def record(self, snapshot: dict) -> None:
        """Persist only server-generated scalar fields from an agent snapshot."""
        if not isinstance(snapshot, dict):
            raise ValueError("Invalid mission snapshot.")
        ident, status, steps = (snapshot.get(k) for k in ("id", "status", "steps"))
        if not isinstance(ident, str) or not ID_PATTERN.fullmatch(ident):
            raise ValueError("Invalid mission identifier.")
        if status not in STATES:
            raise ValueError("Invalid mission state.")
        if type(steps) is not int or not 0 <= steps <= 12:
            raise ValueError("Invalid mission step count.")
        with closing(sqlite3.connect(self.path, timeout=5)) as conn:
            with conn:
                conn.execute("""INSERT INTO missions (id,status,steps,updated_at_ms)
                    VALUES (?,?,?,?) ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status, steps=excluded.steps,
                    updated_at_ms=excluded.updated_at_ms""",
                             (ident, status, steps, int(time.time() * 1000)))
                conn.execute("""DELETE FROM missions WHERE id NOT IN (
                    SELECT id FROM missions ORDER BY updated_at_ms DESC, rowid DESC LIMIT ?
                )""", (self.max_entries,))

    def list_recent(self, *, limit: int = 50) -> list[dict]:
        if type(limit) is not int or not 1 <= limit <= 200:
            raise ValueError("Invalid mission history limit.")
        with closing(sqlite3.connect(self.path, timeout=5)) as conn:
            rows = conn.execute("""SELECT id,status,steps,updated_at_ms FROM missions
                ORDER BY updated_at_ms DESC, rowid DESC LIMIT ?""",
                (min(limit, self.max_entries),)).fetchall()
        return [dict(zip(("id", "status", "steps", "updated_at_ms"), row)) for row in rows]
