"""Regression contracts: aggregate Windows GPU memory without exposing process identity."""
from __future__ import annotations
import json
import unittest

from scripts.neral_gpu_windows_memory import (
    ProbeError, classify_process, parse_windows_gpu_snapshot,
    _WINDOWS_CIM_COMMAND,
)


def sample():
    return {
        "memory": [
            {"Name":"pid_101_luid_0x00000000_0x0000aff0_phys_0",
             "DedicatedUsage":4_194_304_000},
            {"Name":"pid_101_luid_0x00000000_0x0000aff0_phys_1",
             "DedicatedUsage":3_145_728_000},
            {"Name":"pid_202_luid_0x00000000_0x0000afe0_phys_0",
             "DedicatedUsage":2_097_152_000},
            {"Name":"pid_303_luid_0x00000000_0x0000afff_phys_0",
             "DedicatedUsage":104_857_600},
        ],
        "processes": [
            {"ProcessId":101,"Name":"python.exe"},
            {"ProcessId":202,"Name":"ollama_llama_server.exe"},
            {"ProcessId":303,"Name":"explorer.exe"},
        ],
    }


class WindowsGPUAttributionContracts(unittest.TestCase):
    def test_deduplicates_contexts_and_returns_only_aggregate_categories(self):
        result = parse_windows_gpu_snapshot(json.dumps(sample()))
        self.assertEqual(result["status"], "measured_not_exact_vram_total")
        self.assertEqual(result["python_dedicated_mib"], 4000)
        self.assertEqual(result["ollama_dedicated_mib"], 2000)
        self.assertEqual(result["other_dedicated_mib"], 100)
        self.assertEqual(result["mapped_processes"], 3)
        self.assertNotIn("101", json.dumps(result))
        self.assertNotIn("python.exe", json.dumps(result))
        self.assertNotIn("explorer.exe", json.dumps(result))

    def test_unknown_process_is_not_invented_as_python(self):
        snapshot=sample()
        snapshot["processes"]=[{"ProcessId":101, "Name":"something_else.exe"}]
        result=parse_windows_gpu_snapshot(json.dumps(snapshot))
        self.assertEqual(result["python_dedicated_mib"], 0)
        self.assertEqual(result["other_dedicated_mib"], 4000)
        self.assertGreaterEqual(result["unmapped_processes"],2)

    def test_missing_windows_counter_refuses_quality_claim(self):
        result=parse_windows_gpu_snapshot('{"error":"gpu_counters_unavailable"}')
        self.assertEqual(result["status"],"counters_unavailable")
        self.assertNotIn("python_dedicated_mib",result)
        blank=parse_windows_gpu_snapshot('{"memory":[],"processes":[]}')
        self.assertEqual(blank["status"],"no_gpu_process_counters")
        self.assertNotIn("other_dedicated_mib",blank)

    def test_rejects_large_or_malformed_input_without_leaking_names(self):
        for invalid in (
            '{"memory":"secret_path","processes":[]}',
            '{"memory":[{"Name":"pid_101_luid_x","DedicatedUsage":-1}],"processes":[]}',
            '{"memory":[{"Name":"pid_101_luid_x","DedicatedUsage":"500"}],"processes":[]}',
            '{"memory":[{"Name":"pid_101_luid_x","DedicatedUsage":9000000000000}],"processes":[]}',
            '{"memory":[],"processes":[{"ProcessId":3,"Name":123}]}',
            "invalid JSON",
            "x"*1_000_002,
        ):
            with self.assertRaises(ProbeError) as err:
                parse_windows_gpu_snapshot(invalid)
            self.assertNotIn("secret_path",str(err.exception))

    def test_process_classification_and_command_are_readonly(self):
        self.assertEqual(classify_process("pythonw.exe"),"python")
        self.assertEqual(classify_process("ollama.exe"),"ollama")
        self.assertEqual(classify_process("chrome.exe"),"other")
        self.assertEqual(classify_process("C:\\Private\\Python.EXE"),"python")
        self.assertIn("Get-CimInstance",_WINDOWS_CIM_COMMAND)
        self.assertIn("ConvertTo-Json",_WINDOWS_CIM_COMMAND)
        for forbidden in ("Stop-Process","Remove-Item","taskkill","nvidia-smi -r",
                          "Invoke-WebRequest","Invoke-RestMethod","Get-Content"):
            self.assertNotIn(forbidden,_WINDOWS_CIM_COMMAND)


if __name__=="__main__":
    unittest.main()
