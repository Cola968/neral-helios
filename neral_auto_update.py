"""Conservative, stdlib-only NERAL updater.

Only owner-published, checksum-verified GitHub prereleases are eligible.
No models, chats, account data or user-created project files are replaced.
The source preview updates BEFORE server startup; a running EXE is never
overwritten and only changes executable after a fresh process launch.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import urllib.parse
import urllib.request
import zipfile

REPO = "Cola968/neral-helios"
API = "https://api.github.com/repos/Cola968/neral-helios/releases?per_page=50"
SOURCE_ZIP = "NERAL-1.0-Local-Preview.zip"
PREFIX = "NERAL-1.0-Local/"
MAX_ZIP = 10_000_000
MAX_EXE = 150_000_000
SHA_RE = re.compile(r"[0-9a-f]{40}\Z")
DIGEST_RE = re.compile(r"[0-9a-f]{64}\Z")
_RELATIVE_RE = re.compile(r"[A-Za-z0-9_./ -]{1,160}\Z")


class UpdateRejected(RuntimeError):
    """Do not modify the local application or executable."""


def _relative(name: str) -> PurePosixPath:
    if not isinstance(name, str) or not _RELATIVE_RE.fullmatch(name):
        raise UpdateRejected("Ungültiger Dateiname im NERAL-Update.")
    rel = PurePosixPath(name)
    if rel.is_absolute() or any(x in (".", "..", "local_state") for x in rel.parts):
        raise UpdateRejected("Unsicherer Dateipfad im NERAL-Update.")
    return rel


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _tag(release: dict, kind: str) -> str | None:
    if kind not in ("preview", "windows"):
        raise ValueError("Unknown NERAL package")
    tag = release.get("tag_name")
    if not isinstance(tag, str) or not re.fullmatch(
        rf"neral-{kind}-[0-9a-f]{{12}}", tag):
        return None
    return tag


def _asset(release: dict, name: str) -> dict | None:
    tag = release["tag_name"]
    assets = release.get("assets")
    if not isinstance(assets, list):
        return None
    expected = f"https://github.com/{REPO}/releases/download/{tag}/{name}"
    matching = [asset for asset in assets if isinstance(asset, dict)
                and asset.get("name") == name
                and asset.get("browser_download_url") == expected
                and isinstance(asset.get("size"), int)
                and 0 < asset["size"] <= (MAX_EXE if name.endswith(".exe") else MAX_ZIP)]
    return matching[0] if len(matching) == 1 else None


def select_release(releases: object, kind: str) -> dict | None:
    """Only immutable-looking NERAL tags with all required repo-local assets."""
    if kind not in ("preview", "windows"):
        raise ValueError("Invalid update channel")
    if not isinstance(releases, list):
        return None
    required = ([SOURCE_ZIP, "SHA256SUMS.txt", "SOURCE-COMMIT.txt"] if kind == "preview"
                else ["NERAL.exe", "NERAL.exe.sha256", "SOURCE-COMMIT.txt"])
    good = []
    for release in releases:
        if not isinstance(release, dict) or release.get("draft") is not False:
            continue
        if release.get("prerelease") is not True or not _tag(release, kind):
            continue
        published = release.get("published_at")
        if not isinstance(published, str) or not re.fullmatch(
                r"20[0-9]{2}-[01][0-9]-[0-3][0-9]T[0-2][0-9]:[0-5][0-9]:[0-5][0-9]Z",
                published):
            continue
        if all(_asset(release, name) is not None for name in required):
            good.append(release)
    return max(good, key=lambda r: (r["published_at"], r["tag_name"])) if good else None


def _read_url(url: str, *, limit: int) -> bytes:
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in (
            "api.github.com", "github.com"):
        raise UpdateRejected("Unzulässige Update-Quelle.")
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json" if parsed.hostname == "api.github.com"
        else "application/octet-stream",
        "User-Agent": "NERAL-Local-Updater/1"})
    with urllib.request.urlopen(req, timeout=12) as response:
        final = urllib.parse.urlparse(response.geturl())
        if final.scheme != "https" or final.hostname not in (
                "api.github.com", "github.com", "release-assets.githubusercontent.com",
                "objects.githubusercontent.com"):
            raise UpdateRejected("Unerwartete Weiterleitung.")
        data = response.read(limit + 1)
    if not data or len(data) > limit:
        raise UpdateRejected("NERAL-Update überschreitet die Dateigrenze.")
    return data


def _releases() -> object:
    return json.loads(_read_url(API, limit=1_000_000))


def _download_asset(release: dict, name: str, limit: int) -> bytes:
    asset = _asset(release, name)
    if asset is None:
        raise UpdateRejected("Offizielles Update-Asset fehlt.")
    content = _read_url(asset["browser_download_url"], limit=limit)
    if len(content) != asset["size"]:
        raise UpdateRejected("Unerwartete Download-Größe.")
    return content


def _source_commit(release: dict) -> str:
    content = _download_asset(release, "SOURCE-COMMIT.txt", 200)
    sha = content.decode("ascii", errors="strict").strip()
    if not SHA_RE.fullmatch(sha) or not sha.startswith(release["tag_name"].rsplit("-", 1)[-1]):
        raise UpdateRejected("Release-Commit stimmt nicht zum Tag.")
    return sha


def _checksum_file(raw: bytes, filename: str) -> str:
    try:
        line = raw.decode("ascii").strip()
    except UnicodeError as exc:
        raise UpdateRejected("Ungültige SHA-256-Datei.") from exc
    match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *]" + re.escape(filename), line)
    if not match:
        # Windows build used to store a bare SHA-256 hex value.
        if filename == "NERAL.exe" and re.fullmatch(r"[0-9a-fA-F]{64}", line):
            return line.lower()
        raise UpdateRejected("Prüfsummenformat ungültig.")
    return match.group(1).lower()


def inspect_preview_zip(archive: bytes, digest: str, source_sha: str) -> dict[str, bytes]:
    """Parse everything in memory before any local file operation."""
    if not DIGEST_RE.fullmatch(digest) or _hash(archive) != digest:
        raise UpdateRejected("Archivprüfsumme stimmt nicht.")
    if not SHA_RE.fullmatch(source_sha) or not 0 < len(archive) <= MAX_ZIP:
        raise UpdateRejected("Release-Kennung oder Archivgröße ungültig.")
    content: dict[str, bytes] = {}
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            infos = z.infolist()
            if not 1 <= len(infos) <= 130:
                raise UpdateRejected("Unerwartete Anzahl von Release-Dateien.")
            for info in infos:
                if info.is_dir() or not info.filename.startswith(PREFIX):
                    raise UpdateRejected("Unerlaubter ZIP-Eintrag.")
                name = info.filename[len(PREFIX):]
                _relative(name)
                if name in content or (info.external_attr >> 16) & 0o170000 == 0o120000:
                    raise UpdateRejected("Doppelter Eintrag oder Symlink.")
                if info.file_size < 1 or info.file_size > 200_000:
                    raise UpdateRejected("Ungültige Dateigröße.")
                total += info.file_size
                if total > MAX_ZIP:
                    raise UpdateRejected("ZIP-Entpackung überschreitet Limit.")
                content[name] = z.read(info)
    except (zipfile.BadZipFile, OSError, UnicodeError) as exc:
        raise UpdateRejected("Beschädigtes NERAL-Archiv.") from exc
    if "RELEASE-MANIFEST.json" not in content:
        raise UpdateRejected("Release-Manifest fehlt.")
    try:
        manifest = json.loads(content["RELEASE-MANIFEST.json"])
    except (ValueError, UnicodeError) as exc:
        raise UpdateRejected("Release-Manifest beschädigt.") from exc
    hashes = manifest.get("files_sha256") if isinstance(manifest, dict) else None
    if (manifest.get("schema") != 1 or manifest.get("model_weights_included") is not False
            or not isinstance(hashes, dict)
            or set(hashes) != set(content) - {"RELEASE-MANIFEST.json"}):
        raise UpdateRejected("Das Manifest entspricht keinem zulässigen Quellpaket.")
    for name, expected in hashes.items():
        _relative(name)
        if not isinstance(expected, str) or not DIGEST_RE.fullmatch(expected):
            raise UpdateRejected("Ungültiger Dateihash.")
        if _hash(content[name]) != expected:
            raise UpdateRejected("Release-Dateien passen nicht zum Manifest.")
    return content


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise UpdateRejected("Dateilink statt regulärer Datei.")
    fd, temporary = tempfile.mkstemp(prefix=".neral-upgrade-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _checked_target(root: Path, name: str) -> Path:
    rel = _relative(name)
    target = root.joinpath(*rel.parts)
    current = root
    for part in rel.parts[:-1]:
        current = current / part
        if current.is_symlink():
            raise UpdateRejected("Symlink in NERAL-Anwendung.")
    if target.is_symlink():
        raise UpdateRejected("Symlink in NERAL-Anwendung.")
    return target


def apply_preview(root: Path, entries: dict[str, bytes], commit: str) -> None:
    """Replace only the managed allowlist; preserve state and roll back failures."""
    root = Path(root).resolve()
    if (root.joinpath(".git").exists() or not SHA_RE.fullmatch(commit)
            or not isinstance(entries, dict)):
        raise UpdateRejected("Kein sicheres NERAL-Paketverzeichnis.")
    installed = root / "RELEASE-MANIFEST.json"
    if not installed.is_file() or installed.is_symlink():
        raise UpdateRejected("Kein verwaltetes Release-Manifest vorhanden.")
    if "RELEASE-MANIFEST.json" not in entries:
        raise UpdateRejected("Neues Release-Manifest fehlt.")
    try:
        previous = json.loads(installed.read_text("utf8"))
        upcoming = json.loads(entries["RELEASE-MANIFEST.json"])
        old_hashes = previous["files_sha256"]
        new_hashes = upcoming["files_sha256"]
        if not isinstance(old_hashes, dict) or not isinstance(new_hashes, dict):
            raise ValueError()
        if set(entries) != set(new_hashes) | {"RELEASE-MANIFEST.json"}:
            raise ValueError()
    except (ValueError, KeyError, TypeError, UnicodeError) as exc:
        raise UpdateRejected("Manifest stimmt nicht mit Paket überein.") from exc
    # Never overwrite changes made by the user or unknown existing files.
    for name, expected in old_hashes.items():
        path = _checked_target(root, name)
        if not isinstance(expected, str) or not DIGEST_RE.fullmatch(expected):
            raise UpdateRejected("Ungültiges altes Manifest.")
        if not path.is_file() or _hash(path.read_bytes()) != expected:
            raise UpdateRejected("Lokal geänderte Datei: " + name)
    for name, expected in new_hashes.items():
        path = _checked_target(root, name)
        if _hash(entries[name]) != expected:
            raise UpdateRejected("Neue Datei nicht verifiziert: " + name)
        if path.exists() and name not in old_hashes:
            raise UpdateRejected("Update würde eine fremde Datei überschreiben: " + name)
    state = root / "local_state"
    if state.is_symlink():
        raise UpdateRejected("Lokales Zustandsverzeichnis ist ein Link.")
    state.mkdir(exist_ok=True)
    # Keep old sources for recovery. Never include local_state itself.
    backup = state / "previous-version" / commit[:12]
    if backup.exists():
        raise UpdateRejected("Backup-Ziel bereits vorhanden.")
    backup.mkdir(parents=True)
    previous_bytes = {}
    for name in [*old_hashes, "RELEASE-MANIFEST.json"]:
        path = _checked_target(root, name)
        previous_bytes[name] = path.read_bytes()
        destination = _checked_target(backup, name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(previous_bytes[name])
    changed = []
    try:
        for name, content in entries.items():
            target = _checked_target(root, name)
            _atomic_write(target, content)
            changed.append(name)
        _atomic_write(state / "neral-auto-update.json",
                      (json.dumps({"commit": commit}, sort_keys=True)+"\n").encode("utf8"))
    except Exception:
        for name in reversed(changed):
            try:
                if name in previous_bytes:
                    _atomic_write(_checked_target(root, name), previous_bytes[name])
                else:
                    _checked_target(root, name).unlink(missing_ok=True)
            except OSError:
                pass
        raise


def update_source_preview(root: Path) -> str:
    """Network failure keeps the existing version; invalid releases fail closed."""
    root = Path(root).resolve()
    if not (root / "RELEASE-MANIFEST.json").is_file() or (root / ".git").exists():
        return "Unverwaltete NERAL-Installation; kein automatisches Überschreiben."
    chosen = select_release(_releases(), "preview")
    if chosen is None:
        return "Kein geprüftes NERAL-Update veröffentlicht."
    sha = _source_commit(chosen)
    marker = root / "local_state" / "neral-auto-update.json"
    if marker.is_file() and not marker.is_symlink():
        try:
            if json.loads(marker.read_text("utf8")).get("commit") == sha:
                return "NERAL ist auf dem aktuellen freigegebenen Stand."
        except (OSError, ValueError):
            pass
    digest = _checksum_file(_download_asset(chosen, "SHA256SUMS.txt", 256), SOURCE_ZIP)
    archive = _download_asset(chosen, SOURCE_ZIP, MAX_ZIP)
    entries = inspect_preview_zip(archive, digest, sha)
    apply_preview(root, entries, sha)
    return "NERAL auf freigegebene Version " + sha[:12] + " aktualisiert."


def stage_windows_exe(state: Path, data: bytes, commit: str, digest: str,
                      *, published: str = "") -> Path:
    """Stage verified EXE; never replace/execute a running binary."""
    state = Path(state)
    if (state.is_symlink() or not SHA_RE.fullmatch(commit) or
            not DIGEST_RE.fullmatch(digest) or not 512 < len(data) <= MAX_EXE
            or data[:2] != b"MZ" or _hash(data) != digest):
        raise UpdateRejected("Windows-Datei oder SHA-256 ungültig.")
    state.mkdir(parents=True, exist_ok=True)
    folder = state / "versions"
    if folder.is_symlink():
        raise UpdateRejected("Ungültiges Update-Verzeichnis.")
    folder.mkdir(exist_ok=True)
    result = folder / ("NERAL-" + commit[:12] + ".exe")
    _atomic_write(result, data)
    receipt = {"commit": commit, "sha256": digest, "published": published}
    _atomic_write(state / "staged-update.json",
                  (json.dumps(receipt, sort_keys=True)+"\n").encode("utf8"))
    return result


def staged_windows_exe(state: Path, current_commit: str, built_utc: str) -> Path | None:
    """Return only a newer, locally checksum-verified executable."""
    state = Path(state)
    metadata = state / "staged-update.json"
    if not metadata.is_file() or metadata.is_symlink():
        return None
    try:
        receipt = json.loads(metadata.read_text("utf8"))
        sha, digest = receipt["commit"], receipt["sha256"]
        published = receipt.get("published", "")
        if not SHA_RE.fullmatch(sha) or not DIGEST_RE.fullmatch(digest):
            return None
        if sha == current_commit or (built_utc and published and published <= built_utc):
            return None
        binary = state / "versions" / ("NERAL-" + sha[:12] + ".exe")
        if binary.is_symlink() or not binary.is_file() or not 512 < binary.stat().st_size <= MAX_EXE:
            return None
        return binary if _hash(binary.read_bytes()) == digest else None
    except (OSError, TypeError, ValueError, KeyError):
        return None


def check_windows_release(state: Path, current_commit: str, built_utc: str) -> Path | None:
    """Download newer official Windows build while existing GUI keeps running."""
    chosen = select_release(_releases(), "windows")
    if chosen is None:
        return None
    sha = _source_commit(chosen)
    published = chosen["published_at"]
    if sha == current_commit or (built_utc and published <= built_utc):
        return None
    existing = staged_windows_exe(state, current_commit, built_utc)
    if existing is not None and existing.name == "NERAL-" + sha[:12] + ".exe":
        return existing
    digest = _checksum_file(_download_asset(chosen, "NERAL.exe.sha256", 256),
                            "NERAL.exe")
    data = _download_asset(chosen, "NERAL.exe", MAX_EXE)
    return stage_windows_exe(state, data, sha, digest, published=published)


def main() -> int:
    parser = argparse.ArgumentParser(description="Update only validated NERAL source releases.")
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("NERAL_AUTO_UPDATE") == "0":
        print("NERAL-Auto-Update deaktiviert (NERAL_AUTO_UPDATE=0).")
        return 0
    try:
        print(update_source_preview(args.root))
    except Exception as exc:
        # Offline must never prevent the local app from starting.
        print("NERAL-Update nicht angewendet: " + str(exc)[:220])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
