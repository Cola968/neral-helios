"""Pure/CPU tests for opt-in Ollama daemon recovery; no real process mutations."""
import unittest
import sys
from scripts.neral_ollama_cuda_recover import (
    safe_log_signals, recovery_eligible, validate_native_listener,
    environment_signals, RecoveryRefused,
)


class RecoveryContract(unittest.TestCase):
    def test_only_zero_vram_and_enough_free_memory_and_exact_consent(self):
        kw = dict(size_vram=0, free_mib=11022, consent="NERAL_RESTART_OLLAMA_ON_GPU_FALLBACK")
        self.assertTrue(recovery_eligible(**kw))
        for wrong in ({"size_vram": 200}, {"free_mib": 6000}, {"consent": "yes"}):
            args = {**kw, **wrong}
            with self.subTest(args=args):
                self.assertFalse(recovery_eligible(**args))
        self.assertFalse(recovery_eligible(size_vram=None,free_mib=11022,
                                           consent=kw["consent"]))

    def test_log_filter_is_limited_to_last_60_lines_and_never_returns_raw_lines(self):
        log = "\n".join(["unrelated secret prompt 123"]*80 + [
            'level=INFO msg="inference compute" library=cpu prompt=SECRET',
            "cuda error while loading C:\\secret\\model",
            'msg="gpu memory" vram=0',
            "request without gpu information",
        ])
        report = safe_log_signals(log)
        self.assertEqual(report["scanned_lines"], 60)
        self.assertGreater(report["technical_matching_lines"], 0)
        self.assertIn("cuda", report["signal_types"])
        self.assertIn("library_cpu", report["signal_types"])
        self.assertNotIn("SECRET", str(report))
        self.assertNotIn("123", str(report))

    def test_environment_snapshot_has_only_allowlisted_fields(self):
        report = environment_signals({
            "CUDA_VISIBLE_DEVICES": "-1",
            "OLLAMA_LLM_LIBRARY": "cpu",
            "TOKEN_SECRET": "never-expose",
        })
        self.assertTrue(report["cuda_visible_devices_set"])
        self.assertEqual(report["cuda_device_policy"], "disabled")
        self.assertEqual(report["llm_library_policy"], "cpu")
        self.assertNotIn("TOKEN_SECRET", str(report))
        self.assertNotIn("never-expose", str(report))

    def test_owner_validation_rejects_docker_foreign_service_and_ambiguous_listeners(self):
        root = "C:\\Users\\runner\\AppData\\Local\\Programs\\Ollama"
        correct = {
            "processes": [
                {"pid": 442, "name": "ollama.exe",
                 "path": root + "\\ollama.exe", "user": "runner"},
            ],
            "listener_pids": [442],
            "established_clients": 0,
            "listener_addresses": ["127.0.0.1"],
            "all_clients_loopback": True,
            "service_pids": [],
        }
        self.assertEqual(validate_native_listener(correct, root, "runner"), [442])
        bads = [
            {**correct, "listener_pids": [200]},
            {**correct, "listener_pids": []},
            {**correct, "service_pids": [442]},
            {**correct, "processes": [{**correct["processes"][0], "user": "other"}]},
            {**correct, "processes": [{**correct["processes"][0], "name": "docker.exe"}]},
            {**correct, "processes": [{**correct["processes"][0], "path": "C:\\malware\\ollama.exe"}]},
        ]
        for case in bads:
            with self.subTest(case=case), self.assertRaises(RecoveryRefused):
                validate_native_listener(case, root, "runner")

    def test_explicitly_authorized_restart_accepts_small_localhost_client_count(self):
        # The manual restart consent warns that local Ollama chats can be interrupted.
        root = "C:\\Users\\runner\\AppData\\Local\\Programs\\Ollama"
        native = {"pid": 442, "name": "ollama.exe",
                  "path": root + "\\ollama.exe", "user": "runner"}
        row = {"processes": [native], "listener_pids": [442],
               "listener_addresses": ["127.0.0.1"],
               "all_clients_loopback": True, "established_clients": 2,
               "service_pids": []}
        self.assertEqual(validate_native_listener(row, root, "runner"), [442])

    def test_rejects_nonlocal_or_unbounded_clients_with_distinct_reason(self):
        root = "C:\\Users\\runner\\AppData\\Local\\Programs\\Ollama"
        native = {"pid": 442, "name": "ollama.exe",
                  "path": root + "\\ollama.exe", "user": "runner"}
        good = {"processes": [native], "listener_pids": [442],
                "listener_addresses": ["127.0.0.1"],
                "all_clients_loopback": True, "established_clients": 0,
                "service_pids": []}
        cases = [
            ({**good, "listener_addresses": ["0.0.0.0"]}, "ollama_listener_not_loopback"),
            ({**good, "all_clients_loopback": False,
              "established_clients": 1}, "nonlocal_ollama_client"),
            ({**good, "established_clients": 9}, "too_many_connected_ollama_clients"),
            ({**good, "listener_pids": []}, "ambiguous_ollama_listener"),
            ({**good, "service_pids": [442]}, "foreign_or_service_listener"),
        ]
        for row, expected in cases:
            with self.subTest(expected=expected), self.assertRaisesRegex(
                    RecoveryRefused, expected):
                validate_native_listener(row, root, "runner")

    def test_netstat_fallback_recovers_native_loopback_listener(self):
        from scripts.neral_ollama_cuda_recover import parse_native_netstat_inventory
        netstat = """Active Connections
  Proto  Local Address          Foreign Address        State           PID
  TCP    127.0.0.1:11434        0.0.0.0:0              LISTENING       442
  TCP    127.0.0.1:11434        127.0.0.1:59101        ESTABLISHED     442
  TCP    127.0.0.1:59101        127.0.0.1:11434        ESTABLISHED     551
  TCP    [::1]:11434            [::1]:58102            ESTABLISHED     442
"""
        row = parse_native_netstat_inventory(netstat)
        self.assertEqual(row["listener_pids"], [442])
        self.assertEqual(row["listener_addresses"], ["127.0.0.1"])
        self.assertEqual(row["established_clients"], 2)
        self.assertTrue(row["all_clients_loopback"])

    def test_netstat_fallback_marks_externally_exposed_or_remote_listener(self):
        from scripts.neral_ollama_cuda_recover import parse_native_netstat_inventory
        netstat = """  TCP    0.0.0.0:11434           0.0.0.0:0              LISTENING       442
  TCP    192.168.1.4:11434      192.168.1.9:51001     ESTABLISHED     442
"""
        row = parse_native_netstat_inventory(netstat)
        self.assertEqual(row["listener_addresses"], ["0.0.0.0"])
        self.assertFalse(row["all_clients_loopback"])
        root = "C:\\Users\\runner\\AppData\\Local\\Programs\\Ollama"
        row["processes"] = [{"pid": 442, "name": "ollama.exe",
                             "path": root + "\\ollama.exe", "user": "runner"}]
        row["service_pids"] = []
        with self.assertRaisesRegex(RecoveryRefused, "ollama_listener_not_loopback"):
            validate_native_listener(row, root, "runner")

    @unittest.skipUnless(sys.platform == "win32", "Requires native Windows")
    def test_windows_powershell_inventory_returns_strict_metadata(self):
        from scripts.neral_ollama_cuda_recover import _windows_inventory
        row = _windows_inventory()
        self.assertIsInstance(row["processes"], list)
        self.assertIsInstance(row["listener_pids"], list)
        self.assertIsInstance(row["service_pids"], list)
        self.assertIsInstance(row["established_clients"], int)
        self.assertIsInstance(row["listener_addresses"], list)
        self.assertIsInstance(row["all_clients_loopback"], bool)


if __name__ == "__main__":
    unittest.main()
