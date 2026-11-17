"""CPU-only contracts for NERAL v3 prep routing and Windows Docker discovery.

These tests must never start Docker, WSL, Windows services, or GPU jobs.
"""
from pathlib import Path
from neral_cpu_foundation.router import choose_route, generation_profile, escalate

ROOT=Path(__file__).resolve().parents[1]


def test_luna_default_is_unmodified_four_bounded_96_nonthinking():
    route=choose_route("Guten Morgen! Wie geht es?")
    profile=generation_profile(route)
    assert route.tier=="Luna"
    assert route.base=="qwen3.5:4b"
    assert route.adapter_id is None
    assert profile["think"] is False
    assert profile["options"]["num_predict"]==96
    assert profile["adapter_id"] is None


def test_luna_peft_requires_explicit_experiment_opt_in():
    route=choose_route("Hallo",experimental_luna_adapter=True)
    assert route.tier=="Luna"
    assert route.adapter_id=="luna-v1-4b-peft"
    assert choose_route("Hallo").adapter_id is None


def test_complex_task_escalates_and_not_using_luna_96():
    route=choose_route("Beweise mit Z3 und NetworkX, dass es eine Lösung gibt")
    assert route.tier=="Helios"
    assert route.adapter_id is None
    assert generation_profile(route)["options"]["num_predict"]==768
    assert escalate(choose_route("Hallo"),tool_failed=True).tier=="Helios"


def test_windows_script_checks_registry_user_and_d_drives():
    script=(ROOT/"scripts"/"neral_docker_desktop_preflight.ps1").read_text("utf-8")
    for marker in ["HKCU:", "LOCALAPPDATA", "ProgramFiles", "D:\\", "wsl.exe",
                   "docker-desktop", "Test-DockerReady", "90"]:
        assert marker in script


def test_windows_script_fails_closed_no_unreviewed_downloads():
    script=(ROOT/"scripts"/"neral_docker_desktop_preflight.ps1").read_text("utf-8")
    assert "image inspect" in script
    assert "image pull" not in script
    assert "Invoke-WebRequest" not in script
    assert "Set-ExecutionPolicy" not in script
