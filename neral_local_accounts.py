"""Local-only account and conversation storage for the NERAL 1.0 preview.

Credentials and conversations never enter the source ZIP or GitHub. Accounts
separate local data on a shared PC; this is NOT internet-ready authentication.
"""
from __future__ import annotations

import json
import hmac
import os
from contextlib import contextmanager
import re
import secrets
import sqlite3
import time
from pathlib import Path

from flask import jsonify, request, session
from werkzeug.security import check_password_hash, generate_password_hash

MAX_CHATS = 100
MAX_MESSAGES = 80
MAX_MESSAGE_BYTES = 4000
MAX_CHAT_BYTES = 120000


def default_account_db_path():
    """OS user data persists when the app ZIP is replaced."""
    if os.name == "nt":
        root = Path(os.environ.get("LOCALAPPDATA") or
                    Path.home() / "AppData" / "Local")
    else:
        root = Path(os.environ.get("XDG_DATA_HOME") or
                    Path.home() / ".local" / "share")
    return root / "NERAL" / "accounts.sqlite3"


def register_local_accounts(app):
    # The database location is injectable for tests, not user-controlled via HTTP.
    app.config.setdefault("NERAL_ACCOUNT_DB", str(default_account_db_path()))
    failures = {}

    @contextmanager
    def database():
        path = Path(app.config["NERAL_ACCOUNT_DB"])
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        db = sqlite3.connect(path, timeout=8)
        db.execute("PRAGMA busy_timeout=8000")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("""CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL, created INTEGER NOT NULL,
            auth_revision TEXT NOT NULL DEFAULT '')""")
        columns = {row[1] for row in db.execute("PRAGMA table_info(users)")}
        if "auth_revision" not in columns:
            db.execute("ALTER TABLE users ADD COLUMN auth_revision TEXT NOT NULL DEFAULT ''")
        db.execute("""CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            title TEXT NOT NULL, messages TEXT NOT NULL, updated INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)""")
        db.execute("CREATE INDEX IF NOT EXISTS chats_owner ON chats(user_id, updated DESC)")
        db.commit()
        try:
            with db:
                yield db
        finally:
            db.close()

    def user_id():
        value = session.get("neral_user")
        return value if isinstance(value, str) and len(value) == 32 else None

    def auth():
        uid = user_id()
        if not uid:
            return None
        with database() as db:
            row = db.execute("SELECT id,username,auth_revision FROM users WHERE id=?",
                             (uid,)).fetchone()
        revision = session.get("neral_auth_revision")
        if not row or not isinstance(revision, str):
            return None
        return row if hmac.compare_digest(revision, row[2]) else None

    def clean_username(value):
        return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{3,32}", value) is not None

    def payload():
        return request.get_json(silent=True) if request.is_json else None

    def rotate(user, revision=None):
        session.clear()
        session["csrf"] = secrets.token_urlsafe(32)
        if user:
            session["neral_user"] = user
            session["neral_auth_revision"] = revision

    @app.get("/api/neral/account/me")
    def account_me():
        row = auth()
        return jsonify(authenticated=bool(row),
                       username=row[1] if row else None)

    @app.post("/api/neral/account/register")
    def account_register():
        data = payload()
        if not isinstance(data, dict) or set(data) != {"username", "password"}:
            return jsonify(error="Benutzername und Passwort erforderlich."), 400
        username, password = data["username"], data["password"]
        if (not clean_username(username) or not isinstance(password, str)
                or not 12 <= len(password) <= 128 or len(password.encode("utf8")) > 512):
            return jsonify(error="Name: 3–32 Zeichen (Buchstaben, Ziffern, . _ -); Passwort: 12–128 Zeichen."), 400
        # scrypt: never save plaintext or export password hashes to build artifacts.
        hashed = generate_password_hash(password, method="scrypt")
        uid = secrets.token_hex(16)
        try:
            with database() as db:
                revision = secrets.token_hex(16)
                db.execute("INSERT INTO users (id,username,password_hash,created,auth_revision) "
                           "VALUES (?,?,?,?,?)",
                           (uid, username.lower(), hashed, int(time.time()), revision))
        except sqlite3.IntegrityError:
            return jsonify(error="Dieser Benutzername ist bereits vergeben."), 409
        rotate(uid, revision)
        return jsonify(authenticated=True, username=username.lower(),
                       csrf=session["csrf"]), 201

    @app.post("/api/neral/account/login")
    def account_login():
        data = payload()
        if not isinstance(data, dict) or set(data) != {"username", "password"}:
            return jsonify(error="Ungültige Anmeldung."), 400
        username, password = data["username"], data["password"]
        if (not clean_username(username) or not isinstance(password, str)
                or len(password) > 128):
            return jsonify(error="Ungültige Anmeldung."), 401
        key = username.lower()
        now = time.monotonic()
        attempts = failures.get(key, [])
        attempts = [t for t in attempts if now - t < 120]
        if len(attempts) >= 8:
            return jsonify(error="Zu viele Versuche. Bitte später erneut versuchen."), 429
        with database() as db:
            row = db.execute("SELECT id,password_hash,auth_revision FROM users WHERE username=?",
                             (key,)).fetchone()
        if row is None or not check_password_hash(row[1], password):
            attempts.append(now)
            failures[key] = attempts
            return jsonify(error="Ungültige Anmeldung."), 401
        failures.pop(key, None)
        revision = row[2]
        if not revision:
            revision = secrets.token_hex(16)
            with database() as db:
                db.execute("UPDATE users SET auth_revision=? WHERE id=?",
                           (revision, row[0]))
        rotate(row[0], revision)
        return jsonify(authenticated=True, username=key, csrf=session["csrf"])

    @app.post("/api/neral/account/logout")
    def account_logout():
        rotate(None)
        return jsonify(authenticated=False, csrf=session["csrf"])

    @app.get("/api/neral/account/chats")
    def list_chats():
        user = auth()
        if not user:
            return jsonify(error="Bitte anmelden."), 401
        with database() as db:
            rows = db.execute("SELECT id,title,updated FROM chats WHERE user_id=? "
                              "ORDER BY updated DESC,id DESC LIMIT ?",
                              (user[0], MAX_CHATS)).fetchall()
        return jsonify(chats=[{"id":r[0], "title":r[1], "updated":r[2]} for r in rows])

    @app.post("/api/neral/account/chats")
    def create_chat():
        user = auth()
        if not user:
            return jsonify(error="Bitte anmelden."), 401
        data = payload()
        if not isinstance(data, dict) or set(data) != {"title"} or not isinstance(data["title"], str):
            return jsonify(error="Ungültiger Chat-Titel."), 400
        title = data["title"].strip()[:80] or "Neuer Chat"
        cid = secrets.token_hex(16)
        with database() as db:
            count = db.execute("SELECT COUNT(*) FROM chats WHERE user_id=?",
                               (user[0],)).fetchone()[0]
            if count >= MAX_CHATS:
                return jsonify(error="Maximal 100 gespeicherte Chats."), 409
            db.execute("INSERT INTO chats VALUES (?,?,?,?,?)",
                       (cid, user[0], title, "[]", int(time.time())))
        return jsonify(id=cid, title=title, messages=[]), 201

    def owned_chat(cid):
        user = auth()
        if not user:
            return None, None
        if not re.fullmatch(r"[0-9a-f]{32}", cid):
            return user, None
        with database() as db:
            row = db.execute("SELECT id,title,messages FROM chats WHERE id=? AND user_id=?",
                             (cid, user[0])).fetchone()
        return user, row

    @app.get("/api/neral/account/chats/<cid>")
    def get_chat(cid):
        user, row = owned_chat(cid)
        if not user:
            return jsonify(error="Bitte anmelden."), 401
        if row is None:
            return jsonify(error="Chat nicht gefunden."), 404
        return jsonify(id=row[0], title=row[1], messages=json.loads(row[2]))

    @app.put("/api/neral/account/chats/<cid>")
    def save_chat(cid):
        user, row = owned_chat(cid)
        if not user:
            return jsonify(error="Bitte anmelden."), 401
        if row is None:
            return jsonify(error="Chat nicht gefunden."), 404
        data = payload()
        if not isinstance(data, dict) or set(data) != {"messages", "title"}:
            return jsonify(error="Ungültiger Chat."), 400
        title = data["title"]
        messages = data["messages"]
        if (not isinstance(title, str) or not 1 <= len(title.strip()) <= 80
                or not isinstance(messages, list) or len(messages) > MAX_MESSAGES):
            return jsonify(error="Ungültiger Chat-Titel oder Verlauf."), 400
        for item in messages:
            if (not isinstance(item, dict) or set(item) != {"role", "content"}
                    or item["role"] not in ("user", "assistant")
                    or not isinstance(item["content"], str)
                    or len(item["content"].encode("utf8")) > MAX_MESSAGE_BYTES):
                return jsonify(error="Ungültige Chat-Nachricht."), 400
        blob = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
        if len(blob.encode("utf8")) > MAX_CHAT_BYTES:
            return jsonify(error="Gespräch zu groß."), 413
        with database() as db:
            db.execute("UPDATE chats SET title=?,messages=?,updated=? WHERE id=? AND user_id=?",
                       (title.strip(), blob, int(time.time()), cid, user[0]))
        return jsonify(saved=True)

    @app.delete("/api/neral/account/chats/<cid>")
    def delete_chat(cid):
        user, row = owned_chat(cid)
        if not user:
            return jsonify(error="Bitte anmelden."), 401
        if row is None:
            return jsonify(error="Chat nicht gefunden."), 404
        with database() as db:
            db.execute("DELETE FROM chats WHERE id=? AND user_id=?", (cid, user[0]))
        return jsonify(deleted=True)

    @app.get("/api/neral/account/export")
    def export_account():
        user = auth()
        if not user:
            return jsonify(error="Bitte anmelden."), 401
        with database() as db:
            rows = db.execute(
                "SELECT title,messages,updated FROM chats WHERE user_id=? "
                "ORDER BY updated DESC,id DESC LIMIT ?", (user[0], MAX_CHATS)
            ).fetchall()
        response = jsonify(
            format="NERAL-local-account-export-v1",
            username=user[1],
            chats=[{"title":row[0], "messages":json.loads(row[1]),
                    "updated":row[2]} for row in rows]
        )
        response.headers["Content-Disposition"] = 'attachment; filename="neral-chats.json"'
        return response

    @app.post("/api/neral/account/password")
    def change_password():
        user = auth()
        if not user:
            return jsonify(error="Bitte anmelden."), 401
        data = payload()
        if not isinstance(data, dict) or set(data) != {"current_password", "new_password"}:
            return jsonify(error="Aktuelles und neues Passwort erforderlich."), 400
        current, new = data["current_password"], data["new_password"]
        if (not isinstance(current, str) or not isinstance(new, str)
                or len(current) > 128 or not 12 <= len(new) <= 128
                or len(new.encode("utf8")) > 512):
            return jsonify(error="Ungültige Passwörter."), 400
        with database() as db:
            row = db.execute("SELECT password_hash FROM users WHERE id=?",
                             (user[0],)).fetchone()
            if row is None or not check_password_hash(row[0], current):
                return jsonify(error="Aktuelles Passwort stimmt nicht."), 401
            db.execute("UPDATE users SET password_hash=?,auth_revision=? WHERE id=?",
                       (generate_password_hash(new, method="scrypt"),
                        secrets.token_hex(16), user[0]))
        rotate(None)
        return jsonify(changed=True, authenticated=False, csrf=session["csrf"])

    @app.delete("/api/neral/account")
    def delete_account():
        user = auth()
        if not user:
            return jsonify(error="Bitte anmelden."), 401
        data = payload()
        password = data.get("password") if isinstance(data, dict) and set(data) == {"password"} else None
        if not isinstance(password, str) or len(password) > 128:
            return jsonify(error="Passwort erforderlich."), 400
        with database() as db:
            row = db.execute("SELECT password_hash FROM users WHERE id=?",
                             (user[0],)).fetchone()
            if row is None or not check_password_hash(row[0], password):
                return jsonify(error="Passwort stimmt nicht."), 401
            db.execute("DELETE FROM users WHERE id=?", (user[0],))
        rotate(None)
        return jsonify(deleted=True, authenticated=False, csrf=session["csrf"])
