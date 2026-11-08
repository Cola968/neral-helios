"""Transitous fallback follows BahnConnections' own MOTIS journey schema."""
import unittest
from neral_transitous_fallback import decode_polyline, transitous_leg, transitous_itinerary, find_transitous_stop

ENCODED="_p~iF~ps|U_ulLnnqC_mqNvxq"+chr(96)+"@"

class TransitousAdapterTests(unittest.TestCase):
    def test_encoded_geometry_is_decoded_without_inventing_points(self):
        output=decode_polyline(ENCODED,5)
        self.assertEqual(len(output),3)
        self.assertAlmostEqual(output[0][0],38.5)
        self.assertAlmostEqual(output[0][1],-120.2)
        self.assertEqual(decode_polyline("garbage\x00",6),[])

    def test_leg_mapping_keeps_actual_times_and_tracks(self):
        raw={
          "mode":"HIGHSPEED_RAIL","displayName":"ICE 1007",
          "from":{"name":"S+U Berlin Hauptbahnhof","lat":52.525,"lon":13.369,
                  "track":"4","departure":"2026-11-09T09:37:00Z"},
          "to":{"name":"Erfurt Hbf","lat":50.9726,"lon":11.0379,
                "track":"12","arrival":"2026-11-09T11:10:00Z"},
          "startTime":"2026-11-09T09:37:00Z","endTime":"2026-11-09T11:10:00Z",
          "scheduledStartTime":"2026-11-09T09:35:00Z",
          "scheduledEndTime":"2026-11-09T11:10:00Z",
          "realTime":True,"cancelled":False,
          "legGeometry":{"points":ENCODED,"precision":5},
        }
        leg=transitous_leg(raw)
        self.assertEqual(leg["name"],"ICE 1007")
        self.assertEqual(leg["category"],"fern")
        self.assertEqual(leg["from"]["track"],"4")
        self.assertEqual(leg["scheduledStartTime"],"2026-11-09T09:35:00Z")
        self.assertEqual(len(leg["points"]),3)

    def test_missing_times_reject_itinerary(self):
        self.assertIsNone(transitous_itinerary({"legs":[{"mode":"WALK"}]}))

    def test_geocode_rejects_ambiguous_stations(self):
        candidates=[
          {"type":"STOP","name":"Berlin Hbf","id":"berlin-a","lat":52.525,"lon":13.369},
          {"type":"STOP","name":"Berlin Hbf","id":"berlin-b","lat":52.525,"lon":13.369},
        ]
        with self.assertRaises(ValueError):
            find_transitous_stop(candidates,"Berlin Hbf")
