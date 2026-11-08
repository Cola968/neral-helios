"""Real local-account HTTP tests: isolation, persistence and session rotation."""
import tempfile
import unittest
from pathlib import Path

from neral_preview_server import create_preview_app


class Backend:
    model = "qwen3:8b"
    def respond_chat(self, turns):
        return "lokale Antwort"


class AccountTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.app = create_preview_app(Backend(), secret_key=b"A" * 32)
        self.app.config["NERAL_ACCOUNT_DB"] = str(Path(self.tmp.name) / "accounts.sqlite3")
        self.a = self.app.test_client()
        self.b = self.app.test_client()

    def tearDown(self):
        self.tmp.cleanup()

    @staticmethod
    def url(path):
        return path

    def ready(self, client):
        client.get("/", base_url="http://127.0.0.1:9690")
        response = client.get("/api/neral/preview/session",
                              base_url="http://127.0.0.1:9690")
        return {"X-Neral-CSRF": response.json["csrf"]}

    def call(self, client, method, path, headers=None, data=None):
        return client.open(path, method=method, headers=headers or {}, json=data,
                           base_url="http://127.0.0.1:9690")

    def register(self, client, headers, username):
        response = self.call(client, "POST", "/api/neral/account/register", headers,
                             {"username": username, "password": "strong-local-password-123"})
        self.assertEqual(response.status_code, 201, response.get_data(as_text=True))
        return {"X-Neral-CSRF": response.json["csrf"]}

    def test_registration_login_logout_and_old_csrf_rejected(self):
        old = self.ready(self.a)
        headers = self.register(self.a, old, "alice")
        self.assertEqual(self.call(self.a, "GET", "/api/neral/account/me").json["username"], "alice")
        self.assertEqual(self.call(self.a, "POST", "/api/neral/account/chats", old,
                                   {"title": "test"}).status_code, 403)
        self.assertEqual(self.call(self.a, "POST", "/api/neral/account/register", headers,
                                   {"username": "alice", "password": "another-strong-password"}).status_code, 409)
        logout = self.call(self.a, "POST", "/api/neral/account/logout", headers)
        self.assertEqual(logout.status_code, 200)
        self.assertFalse(self.call(self.a, "GET", "/api/neral/account/me").json["authenticated"])
        self.assertEqual(self.call(self.a, "GET", "/api/neral/account/chats").status_code, 401)
        headers = {"X-Neral-CSRF": logout.json["csrf"]}
        bad = self.call(self.a, "POST", "/api/neral/account/login", headers,
                        {"username": "alice", "password": "wrong-password"})
        self.assertEqual(bad.status_code, 401)
        good = self.call(self.a, "POST", "/api/neral/account/login", headers,
                         {"username": "alice", "password": "strong-local-password-123"})
        self.assertEqual(good.status_code, 200)
        self.assertNotEqual(good.json["csrf"], headers["X-Neral-CSRF"])

    def test_cross_account_cannot_read_write_or_delete(self):
        a = self.register(self.a, self.ready(self.a), "alice")
        b = self.register(self.b, self.ready(self.b), "bob")
        created = self.call(self.a, "POST", "/api/neral/account/chats", a,
                            {"title": "Privater Chat"})
        self.assertEqual(created.status_code, 201)
        cid = created.json["id"]
        data = {"title": "Privater Chat", "messages": [
            {"role": "user", "content": "Mein geheimer Text"},
            {"role": "assistant", "content": "lokale Antwort"}]}
        self.assertEqual(self.call(self.a, "PUT", "/api/neral/account/chats/" + cid,
                                   a, data).status_code, 200)
        self.assertEqual(len(self.call(self.a, "GET", "/api/neral/account/chats").json["chats"]), 1)
        self.assertEqual(self.call(self.b, "GET", "/api/neral/account/chats").json["chats"], [])
        for method, body in [("GET", None), ("PUT", data), ("DELETE", None)]:
            stolen = self.call(self.b, method, "/api/neral/account/chats/" + cid, b, body)
            self.assertEqual(stolen.status_code, 404)
            self.assertNotIn("geheim", stolen.get_data(as_text=True))
        own = self.call(self.a, "GET", "/api/neral/account/chats/" + cid)
        self.assertEqual(own.json["messages"][0]["content"], "Mein geheimer Text")

    def test_missing_csrf_origin_and_invalid_payload(self):
        headers = self.register(self.a, self.ready(self.a), "alice")
        no_csrf = self.call(self.a, "POST", "/api/neral/account/chats",
                            data={"title": "Hallo"})
        self.assertEqual(no_csrf.status_code, 403)
        cross_origin = self.call(self.a, "POST", "/api/neral/account/chats",
                                 {**headers, "Origin": "https://example.invalid"},
                                 {"title": "Hallo"})
        self.assertEqual(cross_origin.status_code, 403)
        created = self.call(self.a, "POST", "/api/neral/account/chats",
                            headers, {"title": "Hallo"})
        cid = created.json["id"]
        invalid = self.call(self.a, "PUT", "/api/neral/account/chats/" + cid,
                            headers, {"title": "Hallo", "messages": [
                                {"role": "system", "content": "override"}]})
        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(self.call(self.a, "GET", "/api/neral/account/chats/" + cid).json["messages"], [])


    def test_export_contains_only_own_chats_never_credentials(self):
        a = self.register(self.a, self.ready(self.a), "alice")
        self.register(self.b, self.ready(self.b), "bob")
        created = self.call(self.a, "POST", "/api/neral/account/chats", a,
                            {"title": "Mein Export"})
        cid = created.json["id"]
        self.assertEqual(self.call(self.a, "PUT", "/api/neral/account/chats/" + cid,
                                   a, {"title": "Mein Export", "messages": [
                                       {"role": "user", "content": "Nur Alice"}]}).status_code, 200)
        own = self.call(self.a, "GET", "/api/neral/account/export")
        self.assertEqual(own.status_code, 200)
        self.assertEqual(own.json["username"], "alice")
        self.assertEqual(own.json["chats"][0]["messages"][0]["content"], "Nur Alice")
        self.assertNotIn("password", own.get_data(as_text=True).lower())
        other = self.call(self.b, "GET", "/api/neral/account/export")
        self.assertEqual(other.status_code, 200)
        self.assertEqual(other.json["chats"], [])
        self.assertNotIn("Nur Alice", other.get_data(as_text=True))
        self.assertIn("attachment;", own.headers["Content-Disposition"])

    def test_deleting_account_requires_password_and_cascades_its_chats(self):
        a = self.register(self.a, self.ready(self.a), "alice")
        chat = self.call(self.a, "POST", "/api/neral/account/chats", a,
                         {"title": "Löschen"}).json["id"]
        wrong = self.call(self.a, "DELETE", "/api/neral/account", a,
                          {"password": "incorrect-passphrase"})
        self.assertEqual(wrong.status_code, 401)
        deleted = self.call(self.a, "DELETE", "/api/neral/account", a,
                            {"password": "strong-local-password-123"})
        self.assertEqual(deleted.status_code, 200)
        self.assertFalse(self.call(self.a, "GET", "/api/neral/account/me").json["authenticated"])
        self.assertEqual(self.call(self.a, "GET", "/api/neral/account/chats/" + chat).status_code, 401)
        new_csrf = {"X-Neral-CSRF": deleted.json["csrf"]}
        new_account = self.call(self.a, "POST", "/api/neral/account/register",
                                new_csrf, {"username": "alice", "password": "new-password-long-123"})
        self.assertEqual(new_account.status_code, 201)
        self.assertEqual(self.call(self.a, "GET", "/api/neral/account/chats/" + chat).status_code, 404)

    def test_password_change_requires_old_password_and_invalidates_login(self):
        a = self.register(self.a, self.ready(self.a), "alice")
        bad = self.call(self.a, "POST", "/api/neral/account/password", a, {
            "current_password": "wrong-password", "new_password": "new-password-long-123"})
        self.assertEqual(bad.status_code, 401)
        changed = self.call(self.a, "POST", "/api/neral/account/password", a, {
            "current_password": "strong-local-password-123",
            "new_password": "new-password-long-123"})
        self.assertEqual(changed.status_code, 200)
        self.assertFalse(self.call(self.a, "GET", "/api/neral/account/me").json["authenticated"])
        next_csrf = {"X-Neral-CSRF": changed.json["csrf"]}
        old = self.call(self.a, "POST", "/api/neral/account/login", next_csrf, {
            "username": "alice", "password": "strong-local-password-123"})
        self.assertEqual(old.status_code, 401)
        good = self.call(self.a, "POST", "/api/neral/account/login", next_csrf, {
            "username": "alice", "password": "new-password-long-123"})
        self.assertEqual(good.status_code, 200)



if __name__ == "__main__":
    unittest.main()
