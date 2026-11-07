"""NERAL Cyber Guard v0: bounded, local, read-only Windows firewall status audit.

A deterministic evaluator; not an LLM, packet filter, IDS, or WLAN-wide
firewall. No network requests, shell evaluation, or system modifications.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


class GuardError(ValueError):
    """An unavailable or untrustworthy local snapshot."""


_ALLOWED_NAMES = frozenset(("Domain", "Private", "Public"))
_ALLOWED_NETWORKS = frozenset(("DomainAuthenticated", "Private", "Public"))
_ALLOWED_INBOUND = frozenset(("Allow", "Block", "NotConfigured"))
_MAX_SNAPSHOT_BYTES = 16_384


def _add(findings: list[dict], code: str, severity: str, explanation: str, action: str) -> None:
    findings.append({
        "id": code, "severity": severity,
        "explanation": explanation, "recommended_action": action,
    })


def analyze_snapshot(snapshot: object) -> dict:
    """Analyze only coarse OS protection state; never propagate extra fields."""
    if not isinstance(snapshot, dict) or type(snapshot.get("schema")) is not int or snapshot["schema"] != 1:
        raise GuardError("Ungültige Snapshot-Version.")
    if snapshot.get("source") != "windows-local-readonly":
        raise GuardError("Ungültige Herkunft des Snapshots.")

    raw_profiles = snapshot.get("firewall_profiles")
    raw_networks = snapshot.get("network_categories")
    raw_defender = snapshot.get("defender")
    if not isinstance(raw_profiles, list) or len(raw_profiles) != 3:
        raise GuardError("Unvollständige Windows-Firewall-Profile.")
    profiles: dict[str, dict] = {}
    for raw in raw_profiles:
        if not isinstance(raw, dict):
            raise GuardError("Ungültige Windows-Firewall-Einstellung.")
        name, enabled, inbound = raw.get("name"), raw.get("enabled"), raw.get("inbound")
        if (name not in _ALLOWED_NAMES or name in profiles
                or type(enabled) is not bool or inbound not in _ALLOWED_INBOUND):
            raise GuardError("Ungültiges oder mehrfaches Firewall-Profil.")
        profiles[name] = {"enabled": enabled, "inbound": inbound}
    if set(profiles) != _ALLOWED_NAMES:
        raise GuardError("Unvollständige Windows-Firewall-Profile.")

    if (not isinstance(raw_networks, list) or not 1 <= len(raw_networks) <= 8
            or any(type(cat) is not str or cat not in _ALLOWED_NETWORKS for cat in raw_networks)):
        raise GuardError("Aktives Netzwerkprofil unbekannt oder unvollständig.")
    active = sorted(set("Domain" if cat == "DomainAuthenticated" else cat
                        for cat in raw_networks))
    if raw_defender is not None:
        if (not isinstance(raw_defender, dict)
                or type(raw_defender.get("antivirus_enabled")) is not bool
                or type(raw_defender.get("realtime_enabled")) is not bool):
            raise GuardError("Ungültiger Virenschutzstatus.")

    findings: list[dict] = []
    for name in active:
        state = profiles[name]
        if not state["enabled"]:
            _add(findings, "active_firewall_disabled", "high",
                 "Die Windows-Firewall ist für ein aktives Netzwerkprofil deaktiviert.",
                 "Firewallstatus in Windows-Sicherheit prüfen und erst nach Bestätigung einschalten.")
        elif state["inbound"] == "Allow":
            _add(findings, "active_inbound_allow", "high",
                 "Das aktive Windows-Firewall-Profil erlaubt eingehende Verbindungen standardmäßig.",
                 "Eingangsregeln und Standardaktion in der Windows-Firewall prüfen.")
        elif state["inbound"] == "NotConfigured":
            _add(findings, "active_inbound_unknown", "unknown",
                 "Die Standardaktion für eingehende Verbindungen ist nicht eindeutig.",
                 "Die wirksamen Firewallrichtlinien manuell prüfen.")
    for name, state in profiles.items():
        if name not in active and not state["enabled"]:
            _add(findings, "inactive_firewall_disabled", "warning",
                 "Ein derzeit nicht aktives Windows-Firewall-Profil ist deaktiviert.",
                 "Das betroffene Profil vor dem nächsten Netzwerkwechsel prüfen.")

    if raw_defender is None:
        defender_state = "unknown"
        _add(findings, "antivirus_unavailable", "unknown",
             "Microsoft-Defender-Status konnte nicht gelesen werden; ein anderer Virenschutz ist möglich.",
             "Aktiven Virenschutz in Windows-Sicherheit oder beim Drittanbieter prüfen.")
    elif not raw_defender["antivirus_enabled"]:
        defender_state = "not_active"
        _add(findings, "antivirus_unavailable", "unknown",
             "Microsoft Defender Antivirus ist nicht aktiv; ob ein anderer Virenschutz schützt, ist unbekannt.",
             "Aktiven Virenschutz beim installierten Sicherheitsanbieter prüfen.")
    elif not raw_defender["realtime_enabled"]:
        defender_state = "realtime_disabled"
        _add(findings, "defender_realtime_disabled", "high",
             "Microsoft Defender meldet ausgeschalteten Echtzeitschutz.",
             "Echtzeitschutz in Windows-Sicherheit prüfen und bei Bedarf aktivieren.")
    else:
        defender_state = "realtime_enabled"

    if any(x["severity"] == "high" for x in findings):
        status = "action_required"
    elif any(x["severity"] == "unknown" for x in findings):
        status = "unknown"
    else:
        status = "no_critical_findings"

    return {
        "schema": 1,
        "product": "NERAL Cyber Guard",
        "status": status,
        "mode": "read_only",
        "local_only": True,
        "firewall_modified": False,
        "network_scanned": False,
        "network_wide_firewall": False,
        "quality_improvement_proven": False,
        "observed": {
            "active_network_categories": active,
            "firewall_profiles": [
                {"name": name, **profiles[name]}
                for name in ("Domain", "Private", "Public")
            ],
            "defender": defender_state,
        },
        "findings": findings,
    }


def collect_windows_snapshot() -> dict:
    """Execute only the packaged fixed read-only script, no dynamic commands."""
    if sys.platform != "win32":
        raise GuardError("Live-Prüfung ist nur auf Windows verfügbar.")
    script = Path(__file__).with_name("windows_snapshot.ps1").resolve()
    if not script.is_file():
        raise GuardError("Lokales Prüfskript fehlt.")
    try:
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive",
             "-ExecutionPolicy", "Bypass", "-File", str(script)],
            shell=False, capture_output=True, text=True, encoding="utf-8-sig",
            errors="replace", timeout=25, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardError("Windows-Sicherheitsdaten konnten nicht gelesen werden.") from None
    if completed.returncode != 0 or len(completed.stdout.encode("utf-8")) > _MAX_SNAPSHOT_BYTES:
        raise GuardError("Windows-Abfrage fehlgeschlagen; keine Schutzbewertung möglich.")
    try:
        value = json.loads(completed.stdout.strip())
    except (ValueError, TypeError):
        raise GuardError("Windows lieferte keinen gültigen Sicherheits-Snapshot.") from None
    if not isinstance(value, dict):
        raise GuardError("Ungültige Windows-Antwort.")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NERAL Cyber Guard: lokaler, schreibgeschützter Sicherheitsstatus.")
    parser.add_argument("--json", action="store_true", help="Nur einen datensparsamen JSON-Bericht ausgeben.")
    parser.add_argument("--snapshot", type=Path,
                        help="Nur eine lokale JSON-Testdatei auswerten (kein Live-Scan).")
    args = parser.parse_args(argv)
    try:
        if args.snapshot is not None:
            if args.snapshot.stat().st_size > _MAX_SNAPSHOT_BYTES:
                raise GuardError("Testdatei ist zu groß.")
            snapshot = json.loads(args.snapshot.read_text(encoding="utf-8"))
        else:
            snapshot = collect_windows_snapshot()
        report = analyze_snapshot(snapshot)
    except (OSError, ValueError, GuardError):
        print("NERAL Cyber: Prüfung nicht möglich. Keine Firewall-Einstellungen wurden geändert.",
              file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    else:
        print("NERAL CYBER GUARD | Lokale Prüfung | Kein WLAN-Gateway")
        print("Status:", report["status"])
        for finding in report["findings"]:
            print(f"[{finding['severity'].upper()}] {finding['explanation']}")
            print("  Maßnahme:", finding["recommended_action"])
        if not report["findings"]:
            print("Keine Auffälligkeit in den abgefragten Windows-Einstellungen.")
        print("Hinweis: Das ist keine Malware-Erkennung und schützt nicht das gesamte WLAN.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
