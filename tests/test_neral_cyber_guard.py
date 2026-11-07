"""NERAL Cyber Guard: offline read-only analysis contract (no live network required)."""
from __future__ import annotations
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from cyber.guard import GuardError, analyze_snapshot, collect_windows_snapshot


def snapshot(*, active="Private", enabled=True, inbound="Block", defender=True):
    return {
        "schema": 1,
        "source": "windows-local-readonly",
        "firewall_profiles": [
            {"name": "Domain", "enabled": True, "inbound": "Block"},
            {"name": "Private", "enabled": enabled, "inbound": inbound},
            {"name": "Public", "enabled": True, "inbound": "Block"},
        ],
        "network_categories": [active],
        "defender": {"antivirus_enabled": True, "realtime_enabled": defender},
    }


class LocalCyberGuardTests(unittest.TestCase):
    def test_active_firewall_off_is_high_severity(self):
        result = analyze_snapshot(snapshot(enabled=False))
        self.assertEqual(result["status"], "action_required")
        self.assertIn("active_firewall_disabled", [x["id"] for x in result["findings"]])
        self.assertFalse(result["firewall_modified"])
        self.assertFalse(result["network_scanned"])

    def test_active_inbound_allow_is_high_severity(self):
        result = analyze_snapshot(snapshot(inbound="Allow"))
        ids = [x["id"] for x in result["findings"]]
        self.assertIn("active_inbound_allow", ids)
        self.assertEqual(result["status"], "action_required")

    def test_healthy_profiles_do_not_claim_network_wide_protection(self):
        result = analyze_snapshot(snapshot())
        self.assertEqual(result["status"], "no_critical_findings")
        self.assertFalse(result["network_wide_firewall"])
        self.assertTrue(result["local_only"])
        self.assertFalse(result["quality_improvement_proven"])

    def test_defender_unknown_is_not_mislabelled_as_disabled(self):
        data = snapshot()
        data["defender"] = None
        result = analyze_snapshot(data)
        self.assertEqual(result["status"], "unknown")
        self.assertIn("antivirus_unavailable", [x["id"] for x in result["findings"]])
        self.assertNotIn("defender_realtime_disabled", [x["id"] for x in result["findings"]])

    def test_realtime_disabled_is_not_ignored(self):
        result = analyze_snapshot(snapshot(defender=False))
        self.assertIn("defender_realtime_disabled", [x["id"] for x in result["findings"]])
        self.assertEqual(result["status"], "action_required")

    def test_public_profile_is_not_inherently_insecure(self):
        result = analyze_snapshot(snapshot(active="Public"))
        self.assertEqual(result["status"], "no_critical_findings")

    def test_invalid_missing_or_corrupt_inputs_fail_closed(self):
        for data in (
            {}, [], {"schema": 2}, {"schema": 1},
            {**snapshot(), "firewall_profiles": []},
            {**snapshot(), "network_categories": ["???"]},
            {**snapshot(), "firewall_profiles": [
                {"name": "Private", "enabled": "yes", "inbound": "Block"}]},
        ):
            with self.subTest(data=str(data)[:100]):
                with self.assertRaises(GuardError):
                    analyze_snapshot(data)

    def test_report_never_echoes_addresses_ssid_or_private_paths(self):
        data = snapshot()
        data["wifi_password"] = "SUPERSECRET-WIFI-PASSWORD"
        data["ssid"] = "Our-Router-Name"
        data["ip"] = "192.168.42.123"
        data["device_path"] = "C:/Users/SecretPerson/private"
        result = analyze_snapshot(data)
        as_text = json.dumps(result)
        for secret in ("SUPERSECRET", "Our-Router", "192.168.42", "SecretPerson"):
            self.assertNotIn(secret, as_text)
        self.assertLess(len(as_text), 6000)

    def test_collection_refuses_unsupported_system_without_invoking_process(self):
        with patch("cyber.guard.sys.platform", "linux"), patch("cyber.guard.subprocess.run") as call:
            with self.assertRaises(GuardError):
                collect_windows_snapshot()
            call.assert_not_called()

    def test_collection_starts_only_fixed_local_powershell_script(self):
        valid = json.dumps(snapshot())
        with patch("cyber.guard.sys.platform", "win32"), patch("cyber.guard.subprocess.run") as call:
            call.return_value.stdout = valid
            call.return_value.returncode = 0
            result = collect_windows_snapshot()
        self.assertEqual(result["schema"], 1)
        args, kwargs = call.call_args
        self.assertEqual(args[0][:6], [
            "powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File"
        ])
        self.assertTrue(args[0][-1].endswith("windows_snapshot.ps1"))
        self.assertFalse(kwargs["shell"])
        self.assertTrue(kwargs["capture_output"])
        self.assertLessEqual(kwargs["timeout"], 30)

    def test_fixed_powershell_collector_has_no_mutations_or_network_requests(self):
        script = (Path(__file__).resolve().parents[1]/"cyber"/"windows_snapshot.ps1").read_text(encoding="utf-8")
        self.assertIn("Get-NetFirewallProfile", script)
        self.assertIn("Get-NetConnectionProfile", script)
        self.assertIn("Get-MpComputerStatus", script)
        for forbidden in (
            "New-NetFirewallRule", "Set-NetFirewall", "Remove-NetFirewall",
            "Disable-NetFirewall", "Invoke-WebRequest", "Invoke-RestMethod",
            "Start-Process", "Set-MpPreference", "Remove-Item", "Export-Clixml",
            "Get-NetTCPConnection", "Get-WmiObject", "0.0.0.0",
        ):
            self.assertNotIn(forbidden, script)

    def test_script_and_starter_are_shipped(self):
        root = Path(__file__).resolve().parents[1]
        self.assertTrue((root / "Start NERAL Cyber.cmd").is_file())
        self.assertTrue((root / "cyber" / "__init__.py").is_file())


if __name__ == "__main__":
    unittest.main()
