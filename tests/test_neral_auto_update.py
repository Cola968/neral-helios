"""Offline regression tests: update metadata must never execute unverified content."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import neral_auto_update as update

HASH_A = "a" * 40
HASH_B = "b" * 40
ZIP_NAME = "NERAL-1.0-Local-Preview.zip"


def fake_release(kind, sha, *, published="2026-11-08T17:00:00Z"):
    tag = f"neral-{kind}-{sha[:12]}"
    names = ([ZIP_NAME, "SHA256SUMS.txt", "SOURCE-COMMIT.txt"]
             if kind == "preview" else ["NERAL.exe", "NERAL.exe.sha256"])
    return {
        "tag_name": tag,
        "published_at": published,
        "prerelease": True,
        "draft": False,
        "assets": [{"name": name, "size": 1024,
                    "browser_download_url":
                    f"https://github.com/Cola968/neral-helios/releases/download/{tag}/{name}"}
                   for name in names],
    }


def fixture_zip(sha):
    prefix = "NERAL-1.0-Local/"
    files = {
        "app.py": b"print('safe')\n",
        "static/icon.svg": b"<svg></svg>",
    }
    manifest = {
        "schema": 1,
        "name": "NERAL 1.0 Local Preview",
        "model_weights_included": False,
        "files_sha256": {k: hashlib.sha256(v).hexdigest()
                         for k, v in files.items()},
    }
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(prefix + name, data)
        z.writestr(prefix + "RELEASE-MANIFEST.json", json.dumps(manifest))
    return out.getvalue(), manifest


class AutoUpdateTests(unittest.TestCase):
    def test_only_exact_repository_release_with_provenance_is_selected(self):
        old = fake_release("preview", HASH_A)
        newer = fake_release("preview", HASH_B,
                             published="2026-11-08T18:00:00Z")
        self.assertEqual(update.select_release([old, newer], "preview")["tag_name"],
                         newer["tag_name"])
        self.assertIsNone(update.select_release([
            {**newer, "tag_name": "../../latest"},
            {**newer, "assets": [
                {**newer["assets"][0], "browser_download_url":
                 "https://evil.example/NERAL.exe"}]},
        ], "preview"))
        self.assertIsNone(update.select_release([
            {**newer, "assets": [newer["assets"][0]]}], "preview"))
        self.assertIsNone(update.select_release([
            {**newer, "draft": True}], "preview"))

    def test_invalid_zip_digest_never_touches_files(self):
        data, _ = fixture_zip(HASH_A)
        with self.assertRaises(update.UpdateRejected):
            update.inspect_preview_zip(data, "0" * 64, HASH_A)

    def test_zip_checks_manifest_and_refuses_extra_untrusted_members(self):
        data, manifest = fixture_zip(HASH_A)
        entries = update.inspect_preview_zip(
            data, hashlib.sha256(data).hexdigest(), HASH_A)
        self.assertEqual(entries["app.py"], b"print('safe')\n")
        with io.BytesIO() as stream:
            with zipfile.ZipFile(stream, "w") as z:
                z.writestr("NERAL-1.0-Local/../local_state/stolen", "evil")
                z.writestr("NERAL-1.0-Local/RELEASE-MANIFEST.json",
                           json.dumps(manifest))
            invalid = stream.getvalue()
        with self.assertRaises(update.UpdateRejected):
            update.inspect_preview_zip(
                invalid, hashlib.sha256(invalid).hexdigest(), HASH_A)

    def test_source_update_preserves_local_state_and_rejects_user_changes(self):
        old_bytes, manifest = fixture_zip(HASH_A)
        entries = update.inspect_preview_zip(
            old_bytes, hashlib.sha256(old_bytes).hexdigest(), HASH_A)
        new_bytes = {**entries, "app.py": b"print('new')\n"}
        new_manifest = {**manifest, "files_sha256": {
            k: hashlib.sha256(v).hexdigest()
            for k, v in new_bytes.items() if k != "RELEASE-MANIFEST.json"}}
        new_bytes["RELEASE-MANIFEST.json"] = json.dumps(new_manifest).encode()
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            for name, content in entries.items():
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(content)
            state = root / "local_state"
            state.mkdir()
            (state / "important-chat.db").write_bytes(b"CHATS")
            update.apply_preview(root, new_bytes, HASH_B)
            self.assertEqual((root / "app.py").read_bytes(), b"print('new')\n")
            self.assertEqual((state / "important-chat.db").read_bytes(), b"CHATS")
            (root / "app.py").write_text("local changes", encoding="utf8")
            with self.assertRaises(update.UpdateRejected):
                update.apply_preview(root, entries, HASH_A)
            self.assertEqual((root / "app.py").read_text(), "local changes")

    def test_source_update_refuses_git_checkout_and_missing_manifest(self):
        data, _ = fixture_zip(HASH_A)
        entries = update.inspect_preview_zip(data, hashlib.sha256(data).hexdigest(), HASH_A)
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with self.assertRaises(update.UpdateRejected):
                update.apply_preview(root, entries, HASH_B)
            (root / ".git").mkdir()
            with self.assertRaises(update.UpdateRejected):
                update.apply_preview(root, entries, HASH_B)

    def test_exe_staging_requires_sha256_and_never_executes_download(self):
        data = b"MZ" + b"\0" * 1024
        checksum = hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            state = Path(td)
            stage = update.stage_windows_exe(state, data, HASH_B, checksum)
            self.assertTrue(stage.is_file())
            self.assertEqual(stage.read_bytes(), data)
            self.assertEqual(stage.name, f"NERAL-{HASH_B[:12]}.exe")
            with self.assertRaises(update.UpdateRejected):
                update.stage_windows_exe(state, data, HASH_A, "f" * 64)
            self.assertEqual(stage.read_bytes(), data)

    def test_installed_app_starts_offline_and_preserves_model_and_state(self):
        code = (Path(__file__).resolve().parents[1] / "neral_oneclick.py").read_text("utf8")
        self.assertIn("neral_auto_update", code)
        self.assertNotIn("shell=True", code)
        launcher = (Path(__file__).resolve().parents[1] /
                    "Start-Neral-1.0-Local.ps1").read_text("utf8")
        self.assertIn("-m neral_auto_update", launcher)
        self.assertIn("local_state", launcher)
        pack = (Path(__file__).resolve().parents[1] /
                "scripts/build_neral_1_0_preview.py").read_text("utf8")
        self.assertIn('"neral_auto_update.py"', pack)
        workflow = (Path(__file__).resolve().parents[1] /
                    ".github/workflows/neral-single-click-windows.yml").read_text("utf8")
        self.assertIn("Publish verified Windows release", workflow)
        self.assertIn("NERAL.exe.sha256", workflow)
        self.assertIn("windows-latest", workflow)


if __name__ == "__main__":
    unittest.main()
