"""Local-only account and conversation storage for the NERAL 1.0 preview.

Credentials and conversations never enter the source ZIP or GitHub. Accounts
separate local data on a shared PC; this is NOT internet-ready authentication.
"""
from __future__ import annotations

import json
import re
import secrets
import sqlite3
import time
from pathlib import Path

from flask import jsonify, request, session, current_app
from werkzeug.security import check_password_hash, generate_password_hash

MAX_CHATS = 100
MAX_MESSAGES = 80
MAX_MESSAGE_BYTES = 4000
MAX_CHAT_BYTES = 120000


def register_local_accounts(app):
    # The database location is injectable for tests, not user-controlled via HTTP.
    app.config.setdefault("NERAL_ACCOUNT_DB", str(Path(__file__).resolve().parent /
                                                     "local_state" / "neral_accounts.sqlite3"))
    failures = {}

    def database():
        path = Path(app.config["NERAL_ACCOUNT_DB"])
        path.parent.mkdir(parents=True, exist_ok=True)
        db = sqlite3.connect(path, timeout=8)
        db.execute("PRAGMA busy_timeout=8000")
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("""CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY, username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL, created INTEGER NOT NULL)""")
        db.execute("""CREATE TABLE IF NOT EXISTS chats (
            id TEXT PRIMARY KEY, user_id TEXT NOT NULL,
            title TEXT NOT NULL, messages TEXT NOT NULL, updated INTEGER NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE)""")
        db.execute("CREATE INDEX IF NOT EXISTS chats_owner ON chats(user_id, updated DESC)")
        db.commit()
        return db

    def user_id():
        value = session.get("neral_user")
        return value if isinstance(value, str) and len(value) == 32 else None

    def auth():
        uid = user_id()
        if not uid:
            return None
        with database() as db:
            return db.execute("SELECT id,username FROM users WHERE id=?", (uid,)).fetchone()

    def clean_username(value):
        return isinstance(value, str) and re.fullmatch(r"[a-zA-Z0-9_.-]{3,32}", value) is not None

    def payload():
        return request.get_json(silent=True) if request.is_json else None

    def rotate(user):
        session.clear()
        session["csrf"] = secrets.token_urlsafe(32)
        if user:
            session["neral_user"] = user

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
                db.execute("INSERT INTO users VALUES (?,?,?,?)",
                           (uid, username.lower(), hashed, int(time.time())))
        except sqlite3.IntegrityError:
            return jsonify(error="Dieser Benutzername ist bereits vergeben."), 409
        rotate(uid)
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
            row = db.execute("SELECT id,password_hash FROM users WHERE username=?",
                             (key,)).fetchone()
        if row is None or not check_password_hash(row[1], password):
            attempts.append(now)
            failures[key] = attempts
            return jsonify(error="Ungültige Anmeldung."), 401
        failures.pop(key, None)
        rotate(row[0])
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
