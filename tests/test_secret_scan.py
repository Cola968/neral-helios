"""Synthetic secret checks; values are never committed or printed."""
import subprocess

from scripts.scan_tracked_secrets import is_sensitive_path, scan_history, scan_text, scan_tracked


def test_secret_signatures_report_only_metadata():
    synthetic = "sk_live_" + "A" * 24
    findings = scan_text("SAFE\n" + synthetic + "\n", "mock.py")
    assert findings == [("", "mock.py", 2, "stripe_live_key")]
    assert synthetic not in repr(findings)
    assert not scan_text("sk_test_" + "B" * 24, "mock.py")


def test_sensitive_files_not_example_files():
    assert is_sensitive_path("billing/.env.production")
    assert is_sensitive_path("config/server.key")
    assert not is_sensitive_path("billing/.env.example")
    assert not is_sensitive_path("billing/stripe_sandbox_ids.json")


def test_git_history_detects_removed_key_without_printing_it(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    def run(*args):
        subprocess.check_call(["git", "-C", str(repo), *args], stdout=subprocess.DEVNULL)
    run("init", "-q")
    run("config", "user.email", "test@localhost.invalid")
    run("config", "user.name", "Security test")
    path = repo / "example.py"
    secret = "ghp_" + "B" * 36
    path.write_text("TOKEN = '" + secret + "'\n")
    run("add", ".")
    run("commit", "-qm", "fixture")
    path.write_text("TOKEN = None\n")
    run("commit", "-qam", "remove")
    assert not scan_tracked(repo)
    findings = scan_history(repo)
    assert any(f[1] == "example.py" and f[3] == "github_classic_token" for f in findings)
    assert secret not in repr(findings)
