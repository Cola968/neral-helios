"""NERAL Windows preview must reject unsupported Python before any bootstrap side effects.

These static ordering guards complement the real extracted-Windows-smoke test;
they intentionally do not execute PowerShell on non-Windows runners.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_python_version_is_checked_before_consent_venv_and_pip():
    source = (ROOT / "Start-Neral-1.0-Local.ps1").read_text(encoding="utf-8")
    start = source.index("if(-not(Test-Path -LiteralPath $py -PathType Leaf))")
    finish = source.index("# Existing preview-venv", start)
    bootstrap = source[start:finish]

    version_check = bootstrap.find("sys.version_info >= (3,11)")
    consent_prompt = bootstrap.index("Read-Host")
    create_venv = bootstrap.index("& $base -B -m venv")
    pip_install = bootstrap.index("& $py -B -m pip install")

    assert 0 <= version_check < consent_prompt < create_venv < pip_install, (
        "Check Python 3.11+ before asking for consent, creating a venv or using pip."
    )
    assert "Python 3.11+" in bootstrap, "Failure must explain the required version."
