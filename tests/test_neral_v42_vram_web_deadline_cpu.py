"""V4.2 memory envelope, deterministic web source and tool-only deadline CPU gates."""
import gc
import json
import threading
import unittest
from unittest.mock import Mock, patch
from flask import Flask

from neral_v42_deadline import deadline_seconds_for_request, MODE_BUDGETS, remaining_seconds
from neral_three_tier_runtime import (
    MAX_TOTAL_GPU_USED_MIB, configure_cuda_allocator_limit,
    assert_cuda_total_used_below_cap, release_idle_cuda_cache,
    register_three_tier_api,
)
from neral_evidence_agent import run_evidence_agent
from scripts.neral_v42_telemetry_probe import DIAGNOSTIC_CASES
from test_neral_three_tier_api import make_runtime

QUESTION=DIAGNOSTIC_CASES[2][1]
QUOTE="Apollo 11 landed the first humans on the Moon in 1969."
PAGE={"url":"https://example.org/read-apollo", "page_read":True,"page_text":QUOTE}


class GPUEnvelopeCPU(unittest.TestCase):
    def test_allocator_limit_is_10500_gib_mib_and_does_not_discard_live_weights(self):
        torch=Mock()
        torch.cuda.mem_get_info.return_value=(10_000*2**20,12_000*2**20)
        limit=configure_cuda_allocator_limit(torch)
        self.assertEqual(MAX_TOTAL_GPU_USED_MIB,10752)
        self.assertEqual(limit,10752)
        torch.cuda.set_per_process_memory_fraction.assert_called_once()
        args,kw=torch.cuda.set_per_process_memory_fraction.call_args
        self.assertAlmostEqual(args[0],10752/12000)
        self.assertEqual(kw["device"],0)
    def test_driver_total_used_above_cap_fails_closed(self):
        torch=Mock()
        torch.cuda.mem_get_info.return_value=(2000*2**20,12000*2**20)
        with self.assertRaisesRegex(RuntimeError,"GPU memory cap"):
            assert_cuda_total_used_below_cap(torch)
        torch.cuda.mem_get_info.return_value=(1500*2**20,12000*2**20)
        self.assertLessEqual(assert_cuda_total_used_below_cap(torch),10752)
    def test_release_idle_cache_collects_and_never_calls_unload(self):
        mocktorch=Mock()
        mocktorch.cuda.is_available.return_value=True
        with patch.dict("sys.modules", {"torch":mocktorch}), \
             patch("neral_three_tier_runtime.gc.collect") as collect:
            release_idle_cuda_cache()
        collect.assert_called_once()
        mocktorch.cuda.empty_cache.assert_called_once()
    def test_stopped_owner_processes_must_be_scoped_and_baseline_measured(self):
        from pathlib import Path
        owner=Path("scripts/neral_v41_owner.ps1").read_text(encoding="utf-8")
        self.assertIn("NERAL_GPU_BASELINE_MIB",owner)
        self.assertIn("Get-CimInstance Win32_Process",owner)
        self.assertIn("NERAL_OWNER_PYTHON_CHILDREN_REAPED",owner)
        self.assertIn("--experimental-three-tier-hf",owner)
        self.assertNotIn("taskkill /im python.exe",owner.casefold())


class ToolOnlyDeadlinesCPU(unittest.TestCase):
    def test_envelope_only_for_tool_requests(self):
        self.assertEqual(MODE_BUDGETS["Quartz"],40)
        self.assertEqual(MODE_BUDGETS["Obsidian"],70)
        self.assertEqual(deadline_seconds_for_request("Quartz",tool_active=True),55)
        self.assertEqual(deadline_seconds_for_request("Obsidian",tool_active=True),85)
        self.assertEqual(deadline_seconds_for_request("Quartz",tool_active=False),40)
        self.assertEqual(deadline_seconds_for_request("Obsidian",tool_active=False),70)
        self.assertEqual(deadline_seconds_for_request("Amber",tool_active=True),25)
    def test_real_api_binds_tool_budget_without_extending_everyday(self):
        runtime, low, high, _=make_runtime()
        runtime.evidence_agent=True
        app=Flask("v42-deadline-selective")
        observed={}
        def answer(turns,tier,**kw):
            observed["everyday"]=remaining_seconds()
            return {"text":"ok","verified":False,"safe_to_promote":False}
        def with_tool(turns,tier,**kw):
            observed["web"]=remaining_seconds()
            return {"text":"ok","verified":False,"safe_to_promote":False}
        with patch.object(runtime,"answer",side_effect=answer), \
             patch.object(runtime,"answer_with_evidence",side_effect=with_tool):
            register_three_tier_api(app,runtime,busy=threading.Lock(),search_service=Mock())
            client=app.test_client()
            daily=client.post("/api/neral/chat/three-tier",
                              json={"tier":"Quartz","prompt":"Hallo!","history":[]})
            web=client.post("/api/neral/chat/three-tier",
                           json={"tier":"Quartz","prompt":QUESTION,"history":[],
                                 "web_query":"Apollo 11 first moon landing NASA"})
        self.assertEqual(daily.status_code,200,daily.get_json())
        self.assertEqual(web.status_code,200,web.get_json())
        self.assertGreater(observed["everyday"],39)
        self.assertLessEqual(observed["everyday"],40)
        self.assertGreater(observed["web"],54)
        self.assertLessEqual(observed["web"],55)


class LiteralWebFastPathCPU(unittest.TestCase):
    def _run(self, pages, question=QUESTION):
        calls=[]
        def respond(*_):
            calls.append(1)
            return json.dumps({"tool":"direct_reasoning","parts":["Lies die Originalseite"],
                               "search_query":""})
        class Search:
            def search(self,_):
                return {"results":[{"url":"https://example.org/read-apollo","snippet":"1969"}]}
        result=run_evidence_agent(question,respond,Mock(),
            web_query="Apollo 11 erste Mondlandung NASA", search=Search(),
            page_reader=lambda hits,q:pages,max_attempts=1)
        return result,len(calls)
    def test_single_unambiguous_read_https_quote_finishes_with_one_generation(self):
        result,calls=self._run([PAGE])
        self.assertEqual(calls,1)
        self.assertTrue(result["agent_checked"],result)
        self.assertEqual(result.get("source_year_match"),True)
        self.assertFalse(result["verified"])
        self.assertIn("1969",result["text"])
        self.assertIn("[web1]",result["text"])
        self.assertIsNone(result["error_code"])
    def test_unread_and_conflicting_years_are_not_certified(self):
        pages=[
            PAGE,
            {"url":"https://example.net/second","page_read":True,
             "page_text":"Apollo 11 landed the first humans on the Moon in 1970."},
        ]
        result,calls=self._run(pages)
        self.assertFalse(result.get("agent_checked",False),result)
        self.assertNotEqual(result.get("source_year_match"),True)
        only_unread=[{**PAGE,"page_read":False}]
        result,calls=self._run(only_unread)
        self.assertFalse(result.get("agent_checked",False),result)
    def test_planned_or_negative_event_does_not_shortcut(self):
        for quote in ["Apollo 11 planned to land the first humans on the Moon in 1969.",
                      "Apollo 11 did not land the first humans on the Moon in 1969."]:
            with self.subTest(quote=quote):
                result,calls=self._run([{**PAGE,"page_text":quote}])
                self.assertFalse(result.get("agent_checked",False),result)


if __name__=="__main__":
    unittest.main()
