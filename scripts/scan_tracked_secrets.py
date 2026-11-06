"""High-signal Git secret scanner. Never print potential credential values.

Scans tracked text and optionally all reachable patch history. It does not
replace GitHub Secret Scanning, history-aware specialist scanners, or review.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

PATTERNS = {
    "stripe_live_key": re.compile(r"(?<![A-Za-z0-9_])(?:sk|rk)_live_[A-Za-z0-9]{16,}(?![A-Za-z0-9])"),
    "stripe_webhook_key": re.compile(r"(?<![A-Za-z0-9_])whsec_[A-Za-z0-9]{20,}(?![A-Za-z0-9])"),
    "github_classic_token": re.compile(r"(?<![A-Za-z0-9_])(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36}(?![A-Za-z0-9])"),
    "github_fine_grained_token": re.compile(r"(?<![A-Za-z0-9_])github_pat_[A-Za-z0-9_]{50,}(?![A-Za-z0-9_])"),
    "aws_access_key_id": re.compile(r"(?<![A-Za-z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Za-z0-9])"),
    "openai_project_key": re.compile(r"(?<![A-Za-z0-9_])sk-proj-[A-Za-z0-9_-]{32,}(?![A-Za-z0-9_-])"),
}
TEXT_SUFFIXES = {".py", ".js", ".cjs", ".mjs", ".ts", ".tsx", ".jsx",
                 ".json", ".jsonl", ".yaml", ".yml", ".toml", ".ini",
                 ".cfg", ".txt", ".md", ".ps1", ".cmd", ".bat", ".sh"}
SENSITIVE_FILENAMES = {".env", ".env.local", ".env.production", "id_rsa",
                       "id_ed25519", "credentials.json", "service-account.json"}
SENSITIVE_SUFFIXES = {".p12", ".pfx", ".key"}


def is_sensitive_path(path: str) -> bool:
    name = Path(path).name.lower()
    return name in SENSITIVE_FILENAMES or Path(name).suffix in SENSITIVE_SUFFIXES


def scan_text(text: str, path: str, commit: str = "") -> list[tuple[str, str, int, str]]:
    findings = []
    for line_number, line in enumerate(text.splitlines(), 1):
        for category, pattern in PATTERNS.items():
            if pattern.search(line):
                findings.append((commit, path, line_number, category))
    return findings


def git(repo: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(repo), *args], stderr=subprocess.DEVNULL)


def scan_tracked(repo: Path) -> list[tuple[str, str, int, str]]:
    findings = []
    for raw in git(repo, "ls-files", "-z").split(b"\x00"):
        if not raw:
            continue
        relative = raw.decode("utf-8", errors="replace")
        if is_sensitive_path(relative):
            findings.append(("", relative, 0, "tracked_sensitive_file"))
        file = repo / relative
        if not file.is_file() or file.suffix.lower() not in TEXT_SUFFIXES or file.stat().st_size > 2_000_000:
            continue
        findings.extend(scan_text(file.read_text(encoding="utf-8", errors="replace"), relative))
    return findings


def scan_history(repo: Path) -> list[tuple[str, str, int, str]]:
    """Inspect added patch lines from reachable commits without echoing values."""
    cmd = ["git", "-C", str(repo), "log", "--all", "--format=NERAL_COMMIT:%H",
           "--no-ext-diff", "--no-textconv", "--unified=0", "--patch"]
    findings = []
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL) as proc:
        commit, path, current_line = "", "", 0
        assert proc.stdout is not None
        for raw in proc.stdout:
            line = raw.decode("utf-8", errors="replace").rstrip("\n")
            if line.startswith("NERAL_COMMIT:"):
                commit = line.split(":", 1)[1][:12]
            elif line.startswith("+++ b/"):
                path = line[6:]
                if is_sensitive_path(path):
                    findings.append((commit, path, 0, "historical_sensitive_file"))
            elif line.startswith("@@"):
                match = re.search(r"\+(\d+)(?:,\d+)?\s@@", line)
                if match:
                    current_line = int(match.group(1))
            elif line.startswith("+") and not line.startswith("+++") and path:
                for _, _, _, category in scan_text(line[1:], path, commit=commit):
                    findings.append((commit, path, current_line, category))
                current_line += 1
        if proc.wait() != 0:
            raise RuntimeError("git history scan failed")
    return findings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--history", action="store_true")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    try:
        findings = scan_tracked(repo)
        if args.history:
            findings.extend(scan_history(repo))
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        print(f"Secret scan could not complete: {type(exc).__name__}", file=sys.stderr)
        return 2
    seen = set()
    for finding in findings:
        if finding in seen:
            continue
        seen.add(finding)
        commit, path, line, category = finding
        prefix = f"{commit}:" if commit else ""
        print(f"{prefix}{path}:{line}: {category}")
    if seen:
        print(f"Secret scan FAILED: {len(seen)} potential exposures; no values printed.")
        return 1
    print("Secret scan passed for supported high-signal patterns" + (" and Git history." if args.history else "."))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
