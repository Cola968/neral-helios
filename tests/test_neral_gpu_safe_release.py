"""Guard a user-authorized local GPU reclaim operation from touching unrelated tasks."""
import unittest
from unittest.mock import Mock
from scripts.neral_gpu_safe_release import (
    parse_gpu_csv, parse_gpu_process_rows, select_ollama_targets,
    unload_known_models, assert_idle,
)

class TestSafeGpuRelease(unittest.TestCase):
    def test_genuine_capacity_and_gating(self):
        gpu=parse_gpu_csv("274, 11738, 12282, 2")
        self.assertEqual(gpu["free_mib"],274)
        self.assertEqual(gpu["used_mib"],11738)
        self.assertEqual(gpu["total_mib"],12282)
        self.assertEqual(gpu["util_percent"],2)
        assert_idle(gpu)
        with self.assertRaises(ValueError): assert_idle(parse_gpu_csv("500, 11750, 12282, 25"))
        for bad in ("274, 11738, 12282", "-1, 2, 3, 2", "1, 1, 3, 101", "a, b, c, d"):
            with self.subTest(raw=bad), self.assertRaises(ValueError):
                parse_gpu_csv(bad)

    def test_only_exact_pinned_qwen_models_are_eligible(self):
        models=[
            {"name":"qwen3:8b","size_vram":5_500_000_000},
            {"name":"qwen3.5:9b","size_vram":7_100_000_000},
            {"name":"other-persons-model:latest","size_vram":1_000_000},
        ]
        self.assertEqual(set(select_ollama_targets(models)),{"qwen3:8b","qwen3.5:9b"})
        self.assertEqual(select_ollama_targets([{"name":"other-persons-model:latest"}]),[])
        with self.assertRaises(ValueError): select_ollama_targets([{"name":"qwen3:8b"},{"name":"qwen3:8b"}])
        with self.assertRaises(ValueError): select_ollama_targets([{"name":"qwen3:8b\nfoo"}])

    def test_stop_does_not_touch_other_models_and_is_transactionally_scoped(self):
        calls=[]
        def post(name):
            calls.append(name)
            return {"done":True,"done_reason":"unload"}
        result=unload_known_models([
            {"name":"unknown:latest","size_vram":10},
            {"name":"qwen3:8b","size_vram":100}
        ],post,parse_gpu_csv("274, 11738, 12282, 2"))
        self.assertEqual(calls,["qwen3:8b"])
        self.assertEqual(result["unloaded_count"],1)
        self.assertFalse(result["other_programs_modified"])

    def test_busy_gpu_never_triggers_unload(self):
        p=Mock(return_value={"done":True})
        with self.assertRaises(ValueError):
            unload_known_models([{"name":"qwen3:8b"}],p,parse_gpu_csv("274,11738,12282,27"))
        p.assert_not_called()

    def test_no_unexpected_api_response_means_no_false_success(self):
        with self.assertRaises(ValueError):
            unload_known_models([{"name":"qwen3:8b"}],lambda _:{"done":False},parse_gpu_csv("274,11738,12282,2"))

    def test_sanitize_pid_names_from_user_paths(self):
        raw=("456, C:\\Users\\someuser\\Local\\Programs\\Ollama\\ollama.exe, N/A\n"
             "789, C:\\Private\\personal-name\\python.exe, 4700 MiB\n"
             "790, C:\\Windows\\DWM.EXE, N/A")
        result=parse_gpu_process_rows(raw)
        self.assertEqual(result["ollama_processes"],1)
        self.assertEqual(result["python_processes"],1)
        self.assertEqual(result["other_processes"],1)
        self.assertNotIn("Private",str(result))
        self.assertNotIn("someuser",str(result))
        self.assertNotIn("456",str(result))

if __name__=="__main__":
    unittest.main()
