"""Security regression tests for LAN pairing credentials (no GPU required)."""
import os
import stat

import pytest

from setup_phone import setup


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes; Windows ACLs require separate verification")
def test_phone_pairing_secrets_are_private_with_permissive_umask(tmp_path):
    target = tmp_path / "pairing"
    old_umask = os.umask(0o022)
    try:
        config = setup("192.168.8.27", target)
    finally:
        os.umask(old_umask)
    assert config["host"] == "192.168.8.27"
    assert stat.S_IMODE(target.stat().st_mode) == 0o700
    for filename in ("server.key", "config.json", "HANDY-ZUGANG.txt"):
        assert stat.S_IMODE((target / filename).stat().st_mode) == 0o600


def test_phone_pairing_does_not_overwrite_preexisting_files(tmp_path):
    target = tmp_path / "pairing"
    target.mkdir(mode=0o700)
    (target / "server.key").write_text("keep-this-existing-key", encoding="utf-8")
    with pytest.raises(FileExistsError):
        setup("192.168.8.27", target)
    assert (target / "server.key").read_text(encoding="utf-8") == "keep-this-existing-key"


@pytest.mark.parametrize("host", ["8.8.8.8", "127.0.0.1", "0.0.0.0", "169.254.2.3", "fd00::1", "not-an-ip"])
def test_phone_pairing_rejects_unsafe_host(host, tmp_path):
    with pytest.raises(ValueError):
        setup(host, tmp_path / "pairing")

@pytest.mark.skipif(os.name == "nt", reason="POSIX directory modes")
def test_existing_shared_directory_is_not_silently_chmodded(tmp_path):
    target = tmp_path / "shared"
    target.mkdir(mode=0o755)
    with pytest.raises(PermissionError):
        setup("192.168.8.27", target)
    assert stat.S_IMODE(target.stat().st_mode) == 0o755
