"""Guard opt-in three-tier launcher dependency isolation.

The frozen QLoRA venv is shared with training and cannot be mutated to
install web server packages. The launcher may *only* add a small independent
waitress package tree in local_state after explicit user consent.
"""
from pathlib import Path

LAUNCHER = Path(__file__).resolve().parents[1] / "Start-Neral-Three-Tier.ps1"


def test_waitress_dependency_is_isolated_and_consent_gated():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "local_state" in source
    assert "three-tier-web-deps" in source
    assert "Read-Host" in source
    assert "--target" in source
    assert "--no-deps" in source
    assert "--no-cache-dir" in source
    assert "waitress==3.0.2" in source
    assert "PYTHONPATH" in source
    assert "NERAL_THREE_TIER_INSTALL_CONSENT" in source


def test_launcher_never_installs_dependencies_into_model_environment():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "D:/neral-qlora/venv/Scripts/python.exe" in source
    assert "pip install waitress==3.0.2" not in source
    assert "pip install -r" not in source
    assert "pip install --upgrade" not in source
    assert "pip install torch" not in source
    assert "pip install peft" not in source


def test_check_only_must_never_prompt_or_install():
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "$CheckOnly" in source
    assert "if(-not $CheckOnly)" in source

def test_powershell_python_c_instructions_do_not_require_lossy_quoted_literals():
    """Windows PowerShell 5.1 strips the inner quote characters in -c argv."""
    source=LAUNCHER.read_text(encoding="utf-8")
    assert 'print("NERAL_THREE_TIER_LOCAL_PINNED_READY")' not in source
    assert 'find_spec("waitress")' not in source
    assert 'os.environ["NERAL_THREE_TIER_WEB_DEPS"]' not in source
    assert "print(1)" in source
    assert "Path(sys.path[1])" in source


def test_launcher_dependency_install_must_be_separate_from_windows_checkonly():
    source=LAUNCHER.read_text(encoding="utf-8")
    assert "pip list --format=freeze" in source
    assert "if($CheckOnly){" in source
    assert "--target $webDeps waitress==3.0.2" in source
