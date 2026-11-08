"""NERAL BahnConnections widget: bound external lookup and truthful travel results."""
from datetime import datetime
from unittest import TestCase
from unittest.mock import patch
from neral_bahnconnections import plan_journeys, normalize_journey, station_for_name, provider_base

def leg():
    return {
        "name":"ICE 599","category":"fern",
        "from":{"name":"Berlin Hbf","lat":52.52499,"lon":13.36918,"track":"5"},
        "to":{"name":"Stuttgart Hbf","lat":48.783,"lon":9.1816,"track":"8"},
        "startTime":"2026-11-09T09:00:00+01:00",
        "endTime":"2026-11-09T14:30:00+01:00",
        "scheduledStartTime":"2026-11-09T08:55:00+01:00",
        "scheduledEndTime":"2026-11-09T14:30:00+01:00",
        "realtime":True,"cancelled":False,
        "points":[[52.52499,13.36918],[52.52,13.36],[52.51,13.35]],
        "stops":[{"name":"Berlin Hbf"},{"name":"Stuttgart Hbf"}],
        "alerts":[]
    }
def journey():
    x=leg()
    return {"id":"j1","startTime":x["startTime"],"endTime":x["endTime"],
            "scheduledStartTime":x["scheduledStartTime"],
            "scheduledEndTime":x["scheduledEndTime"],
            "durationSeconds":19800,"transfers":0,"cancelled":False,
            "realtimeStatus":"live","legs":[x],"warnings":[],"sourceLabel":"Transitous / MOTIS"}

class BahnJourneyTests(TestCase):
    def test_curated_station_is_a_grounded_station(self):
        x=station_for_name("Berlin Hauptbahnhof")
        self.assertEqual(x["name"],"Berlin Hbf")
        self.assertGreater(x["lat"],50)
        self.assertEqual(station_for_name("Stuttgart")["name"],"Stuttgart Hbf")
        self.assertIsNone(station_for_name("Bahnhof erfunden"))

    def test_normalize_preserves_actual_times_and_real_rail_geometry(self):
        x=normalize_journey(journey())
        self.assertEqual(x["legs"][0]["name"],"ICE 599")
        self.assertEqual(x["legs"][0]["points"][1],[52.52,13.36])
        self.assertEqual(x["scheduledStartTime"],"2026-11-09T08:55:00+01:00")
        self.assertEqual(x["transfers"],0)

    def test_reject_invented_times_or_geometry(self):
        bad=journey()
        bad["endTime"]="nonsense"
        self.assertIsNone(normalize_journey(bad))
        bad=journey()
        bad["legs"][0]["points"]=[[12,13],["alert(1)",90],[91,20]]
        clean=normalize_journey(bad)
        self.assertEqual(clean["legs"][0]["points"],[[12,13]])

    def test_provider_url_is_fixed_https_or_explicit_safe_configuration(self):
        self.assertEqual(provider_base({}),"https://bahnconnections-de.a-stad.chatgpt.site")
        for value in ["http://127.0.0.1:4000","https://127.0.0.1","https://user:pass@example.org","file:///tmp/x","http://localhost","https://example.org/path"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    provider_base({"NERAL_BAHNCONNECTIONS_URL":value})

    def test_live_search_uses_bahnconnections_and_returns_provenance_without_guessing(self):
        received=[]
        def fake_fetch(url, body=None, timeout=0):
            received.append((url,body))
            return {"journeys":[journey()],"source":"Transitous / MOTIS","updatedAt":"2026-11-09T08:58:00+01:00","realtimeStatus":"live","warnings":[]}
        out=plan_journeys({"from":"Berlin Hauptbahnhof","to":"Stuttgart"},fetch_json=fake_fetch)
        self.assertEqual(len(received),1)
        self.assertTrue(received[0][0].endswith("/api/journeys"))
        self.assertEqual(received[0][1]["from"]["name"],"Berlin Hbf")
        self.assertEqual(out["status"],"ok")
        self.assertEqual(len(out["journeys"]),1)

    def test_provider_errors_never_create_fake_connections(self):
        def fail(*args,**kwargs):raise RuntimeError("Nicht erreichbar")
        with self.assertRaisesRegex(RuntimeError,"Nicht erreichbar"):
            plan_journeys({"from":"Berlin Hauptbahnhof","to":"Stuttgart"},fetch_json=fail)

    def test_large_or_unsafe_request_is_rejected(self):
        for payload in [
            {"from":"Berlin Hbf","to":"Stuttgart Hbf","evil":"x"},
            {"from":"Berlin Hbf","to":"Berlin Hbf"},
            {"from":"Berlin Hbf","to":"https://example.org"},
            {"from":"a"*81,"to":"Stuttgart Hbf"},
        ]:
            with self.subTest(payload=payload):
                with self.assertRaises(ValueError):
                    plan_journeys(payload,fetch_json=lambda *x,**y:None)
