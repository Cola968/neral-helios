"""RED contracts: per-request deadline must not leave GPU lock; 409 is transport backpressure."""
import threading
import unittest
from unittest.mock import patch
from flask import Flask

from neral_v42_deadline import (
    DeadlineExceeded, MODE_BUDGETS, request_deadline, check_deadline,
    should_stop_generation, remaining_seconds,
)
from scripts.neral_v42_resumable import (
    ServerBusyError, wait_until_available, evaluate_with_busy_retries, smoke_sample,
)
from test_neral_three_tier_api import make_runtime
from neral_three_tier_runtime import register_three_tier_api

class ControlledClock:
    def __init__(self):
        self.now=100.0
        self.sleeps=[]
    def monotonic(self):
        return self.now
    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now+=seconds

class DeadlineAndBackoffCPU(unittest.TestCase):
    def test_deadline_scoped_and_25_40_70(self):
        self.assertEqual(MODE_BUDGETS, {"Amber":25, "Quartz":40, "Obsidian":70})
        clock=ControlledClock()
        with patch("neral_v42_deadline.time.monotonic",clock.monotonic):
            self.assertIsNone(remaining_seconds())
            with request_deadline("Quartz"):
                self.assertAlmostEqual(remaining_seconds(),40)
                self.assertFalse(should_stop_generation())
                clock.now+=40.1
                self.assertTrue(should_stop_generation())
                with self.assertRaises(DeadlineExceeded):
                    check_deadline()
            self.assertIsNone(remaining_seconds())
            with request_deadline("Obsidian"):
                self.assertAlmostEqual(remaining_seconds(),70)

    def test_wait_until_available_polls_2_to_5_seconds_and_stops_at_90(self):
        clock=ControlledClock()
        class Client:
            def __init__(self):
                self.responses=[{"busy":True},{"busy":True},{"busy":False}]
                self.reads=0
            def get(self,path):
                self.reads+=1
                self.path=path
                return self.responses.pop(0)
        client=Client()
        with patch("scripts.neral_v42_resumable.time.monotonic",clock.monotonic), patch(
            "scripts.neral_v42_resumable.time.sleep",clock.sleep
        ):
            self.assertTrue(wait_until_available(client,max_wait_seconds=90))
        self.assertEqual(client.path,"/api/neral/chat/three-tier/availability")
        self.assertEqual(client.reads,3)
        self.assertTrue(all(2<=v<=5 for v in clock.sleeps))
        busy=type("Client",(),{"get":lambda self,path: {"busy":True}})()
        clock.now=100.0
        clock.sleeps=[]
        with patch("scripts.neral_v42_resumable.time.monotonic",clock.monotonic), patch(
            "scripts.neral_v42_resumable.time.sleep",clock.sleep
        ):
            self.assertFalse(wait_until_available(busy,max_wait_seconds=9))
        self.assertLessEqual(sum(clock.sleeps),9.001)

    def test_409_retries_without_scoring_or_losing_case(self):
        case={"id":"x","category":"everyday","prompt":"Hello"}
        attempts=[]
        def evaluate(tier,item):
            attempts.append(item["id"])
            if len(attempts)==1:
                raise ServerBusyError()
            return {"id":item["id"],"grade":"pass","score":1,"latency_ms":200,
                    "http_409_retries":0,"busy_wait_ms":0}
        with patch("scripts.neral_v42_resumable.wait_until_available",return_value=True), patch(
            "scripts.neral_v42_resumable.time.sleep",return_value=None
        ):
            row=evaluate_with_busy_retries("Quartz",case,client=object(),evaluator=evaluate)
        self.assertEqual(attempts,["x","x"])
        self.assertEqual(row["grade"],"pass")
        self.assertEqual(row["http_409_retries"],1)
        with patch("scripts.neral_v42_resumable.wait_until_available",return_value=False):
            with self.assertRaises(ServerBusyError):
                evaluate_with_busy_retries("Quartz",case,client=object(),evaluator=evaluate,
                                           max_wait_seconds=0)
        self.assertEqual(attempts,["x","x"])  # not a graded extra attempt

    def test_smoke_has_exact_five_distinct_categories(self):
        cases=[{"id":str(i),"prompt":"x","category":cat}
               for i,cat in enumerate(("everyday","multiturn","live_web",
                                        "math_logic","programming","writing","uncertainty"))]
        chosen=smoke_sample(cases)
        self.assertEqual(len(chosen),5)
        self.assertEqual(len(set(c["id"] for c in chosen)),5)
        self.assertEqual(set(c["category"] for c in chosen),
                         {"everyday","multiturn","live_web","math_logic","programming"})

    def test_authenticated_availability_probe_uses_owner_cookie_session(self):
        from scripts.neral_v42_owner_availability import verify_owner_available, STATUS, AVAILABILITY
        class Owner:
            def __init__(self):
                self.connected=False
                self.calls=[]
                self.busy=False
            def connect(self):
                self.connected=True
            def get(self,path):
                self.calls.append(path)
                if path==STATUS:
                    return {"enabled":True,"public_tiers":["Amber","Quartz","Obsidian"]}
                if path==AVAILABILITY:
                    return {"busy":self.busy,
                            "deadline_seconds":{"Amber":25,"Quartz":40,"Obsidian":70}}
                raise AssertionError("Unknown URL")
        client=Owner()
        self.assertFalse(verify_owner_available(client)["busy"])
        self.assertTrue(client.connected)
        self.assertEqual(client.calls,[STATUS,AVAILABILITY])
        client.busy=True
        with self.assertRaises(RuntimeError):
            verify_owner_available(client)

    def test_availability_endpoint_tracks_lock_without_revealing_prompts(self):
        runtime,*_=make_runtime()
        app=Flask(__name__)
        app.secret_key="test-secret"
        busy=threading.Lock()
        register_three_tier_api(app,runtime,busy=busy)
        client=app.test_client()
        path="/api/neral/chat/three-tier/availability"
        ready=client.get(path)
        self.assertEqual(ready.status_code,200)
        self.assertIs(ready.get_json()["busy"],False)
        self.assertNotIn("prompt",ready.get_json())
        busy.acquire()
        try:
            self.assertIs(client.get(path).get_json()["busy"],True)
        finally:
            busy.release()

    def test_obsidian_keeps_deep_and_quartz_stays_single_pass(self):
        runtime,*_=make_runtime()
        runtime.evidence_agent=True
        app=Flask(__name__)
        app.secret_key="test-secret"
        register_three_tier_api(app,runtime,busy=threading.Lock())
        client=app.test_client()
        observed=[]
        def mock_agent(*args,**kwargs):
            observed.append(kwargs)
            return {"text":"Test","verified":False,"safe_to_promote":False,
                    "agent_attempts":1,"agent_checked":True,"error_code":None}
        with patch("neral_evidence_agent.run_evidence_agent",side_effect=mock_agent):
            ob=client.post("/api/neral/chat/three-tier",
                json={"tier":"Obsidian","prompt":"Ordne vier Schichten mit Z3 und prüfe alle Bedingungen.",
                      "benchmark_isolated":True})
            self.assertEqual(ob.status_code,200)
            self.assertTrue(ob.get_json().get("obsidian_deep"))
            self.assertTrue(ob.get_json()["benchmark_isolated"])
        self.assertEqual(len(observed),1)
        self.assertTrue(observed[0]["deep"])

if __name__=="__main__":
    unittest.main()
