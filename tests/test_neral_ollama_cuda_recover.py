"""Pure/CPU tests for opt-in Ollama daemon recovery; no real process mutations."""
import unittest
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
            "service_pids": [],
        }
        self.assertEqual(validate_native_listener(correct, root, "runner"), [442])
        bads = [
            {**correct, "listener_pids": [200]},
            {**correct, "listener_pids": []},
            {**correct, "established_clients": 1},
            {**correct, "service_pids": [442]},
            {**correct, "processes": [{**correct["processes"][0], "user": "other"}]},
            {**correct, "processes": [{**correct["processes"][0], "name": "docker.exe"}]},
            {**correct, "processes": [{**correct["processes"][0], "path": "C:\\malware\\ollama.exe"}]},
        ]
        for case in bads:
            with self.subTest(case=case), self.assertRaises(RecoveryRefused):
                validate_native_listener(case, root, "runner")


if __name__ == "__main__":
    unittest.main()
