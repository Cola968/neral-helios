"""Read-only Transitous/MOTIS fallback using BahnConnections' existing source schema.

Enabled only if the deployed BahnConnections endpoint is unreachable. Queries
contain only two stations, departure time and rail-mode filters, never chats.
Source and degraded provenance are explicitly shown to users.
"""
from __future__ import annotations

from datetime import datetime
import json
import math
from urllib.parse import urlencode
from urllib.request import Request, build_opener
from urllib.error import HTTPError, URLError
from zoneinfo import ZoneInfo

from neral_bahnconnections import NoRedirect, normalize_name, normalize_journey

BASE = "https://api.transitous.org"
MODES = ",".join(("HIGHSPEED_RAIL", "LONG_DISTANCE", "NIGHT_RAIL",
                  "REGIONAL_FAST_RAIL", "REGIONAL_RAIL", "SUBURBAN", "SUBWAY", "TRAM"))
CATEGORIES = {
    "HIGHSPEED_RAIL": "fern", "LONG_DISTANCE": "fern", "NIGHT_RAIL": "fern",
    "REGIONAL_FAST_RAIL": "regional", "REGIONAL_RAIL": "regional",
    "SUBURBAN": "sbahn", "SUBWAY": "ubahn", "TRAM": "tram",
}


def decode_polyline(encoded: object, precision: int = 6) -> list[list[float]]:
    if not isinstance(encoded, str) or len(encoded) > 120_000 or precision not in (5, 6):
        return []
    coordinates=[]
    latitude=longitude=0
    index=0
    try:
        while index<len(encoded):
            output=[]
            for _ in range(2):
                shift=value=0
                while True:
                    if index>=len(encoded) or shift>35:
                        raise ValueError("Invalid coordinate")
                    char=ord(encoded[index])-63
                    index+=1
                    if not 0<=char<=63:
                        raise ValueError("Invalid encoding")
                    value|=(char&31)<<shift
                    shift+=5
                    if char<32:
                        break
                output.append((value>>1)^(-(value&1)))
            latitude+=output[0]
            longitude+=output[1]
            a,b=latitude/(10**precision),longitude/(10**precision)
            if not (math.isfinite(a) and math.isfinite(b) and -90<=a<=90 and -180<=b<=180):
                raise ValueError("Invalid coordinates")
            coordinates.append([a,b])
            if len(coordinates)>40_000:
                raise ValueError("Too many points")
    except (ValueError, OverflowError):
        return []
    if len(coordinates)>1700:
        step=math.ceil((len(coordinates)-1)/1699)
        reduced=coordinates[::step]
        if reduced[-1]!=coordinates[-1]:
            reduced.append(coordinates[-1])
        return reduced
    return coordinates


def transitous_leg(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    frm=raw.get("from") or {}
    to=raw.get("to") or {}
    if not isinstance(frm, dict) or not isinstance(to, dict):
        return None
    start=raw.get("startTime")
    end=raw.get("endTime")
    if not isinstance(start, str) or not isinstance(end, str):
        return None

    def stop(raw: dict) -> dict:
        result={"name":raw.get("name"),"lat":raw.get("lat"),"lon":raw.get("lon")}
        if raw.get("track") is not None:
            result["track"]=str(raw["track"])[:20]
        if raw.get("scheduledTrack") is not None:
            result["scheduledTrack"]=str(raw["scheduledTrack"])[:20]
        return result

    geom=raw.get("legGeometry")
    points=decode_polyline(geom.get("points"),geom.get("precision",6)) if isinstance(geom, dict) else []
    mode=raw.get("mode") if isinstance(raw.get("mode"),str) else "WALK"
    name=raw.get("displayName") or raw.get("tripShortName") or raw.get("routeShortName") or (
        "Fußweg" if mode=="WALK" else "Bahn")
    if not isinstance(name, str):
        return None
    return {
        "name": name[:80], "category": CATEGORIES.get(mode,"walk"),
        "from": stop(frm), "to": stop(to),
        "startTime":start, "endTime":end,
        "scheduledStartTime":raw.get("scheduledStartTime") or start,
        "scheduledEndTime":raw.get("scheduledEndTime") or end,
        "realtime":raw.get("realTime") is True,
        "cancelled":raw.get("cancelled") is True,
        "points":points,
    }


def transitous_itinerary(raw: object) -> dict | None:
    if not isinstance(raw, dict) or not isinstance(raw.get("legs"),list):
        return None
    legs=[]
    for section in raw["legs"][:10]:
        leg=transitous_leg(section)
        if not leg:
            return None
        legs.append(leg)
    if not legs:
        return None
    number=sum(l["category"]!="walk" for l in legs)-1
    if number<0:
        return None
    return normalize_journey({
        "startTime": raw.get("startTime"),
        "endTime": raw.get("endTime"),
        "transfers": number,
        "scheduledStartTime":legs[0]["scheduledStartTime"],
        "scheduledEndTime":legs[-1]["scheduledEndTime"],
        "cancelled":any(x["cancelled"] for x in legs),
        "realtimeStatus":"live" if any(x["realtime"] for x in legs) else "schedule",
        "legs":legs,"sourceLabel":"Transitous / MOTIS","warnings":[],
    })


def _get_json(path: str, params: dict) -> object:
    if path not in ("/api/v1/geocode","/api/v6/plan"):
        raise ValueError("Transitous endpoint not approved")
    url=BASE+path+"?"+urlencode(params)
    opener=build_opener(NoRedirect)
    try:
        req=Request(url,headers={"Accept":"application/json",
            "User-Agent":"NERAL-Local-User-Directed-Journey/1.0 (+https://github.com/Cola968/neral-helios)"})
        with opener.open(req,timeout=16 if path.endswith("/plan") else 9) as response:
            if not str(response.headers.get("Content-Type","")).lower().startswith("application/json"):
                raise RuntimeError("Unexpected Transitous content-type")
            raw=response.read(3_000_001)
    except (HTTPError, URLError, OSError) as exc:
        raise RuntimeError("Transitous derzeit nicht erreichbar.") from exc
    if len(raw)>3_000_000:
        raise RuntimeError("Transitous-Antwort zu groß.")
    try:
        return json.loads(raw)
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("Transitous liefert kein gültiges JSON.") from exc


def find_transitous_stop(candidates: object, name: str) -> dict:
    if not isinstance(candidates, list):
        raise RuntimeError("Transitous-Bahnhofsuche ungültig.")
    match=[s for s in candidates[:12] if isinstance(s,dict) and s.get("type")=="STOP"
           and s.get("id") and isinstance(s.get("name"),str) and
           normalize_name(s["name"])==normalize_name(name) and
           isinstance(s.get("lat"),(int,float)) and isinstance(s.get("lon"),(int,float))]
    if len(match)!=1:
        raise ValueError("Bahnhof nicht eindeutig gefunden. Bitte den vollständigen Namen angeben.")
    return match[0]


def plan_transitous_direct(from_name: str, to_name: str, departure: str,
                          *, request_json=None) -> dict:
    fetch=request_json or _get_json
    def station(name):
        candidates=fetch("/api/v1/geocode",{
            "text":name,"type":"STOP","mode":MODES,"language":"de","numResults":"10"})
        return find_transitous_stop(candidates,name)
    start=station(from_name)
    target=station(to_name)
    if start["id"]==target["id"]:
        raise ValueError("Start und Ziel müssen verschieden sein.")
    params={
        "fromPlace":start["id"],"toPlace":target["id"],
        "time":departure,"arriveBy":"false","maxTransfers":"4",
        "minTransferTime":"0","transitModes":MODES,
        "maxDirectTime":"0","numItineraries":"6","maxItineraries":"8",
        "searchWindow":"10800","timetableView":"false",
        "detailedLegs":"true","detailedTransfers":"false",
        "withScheduledSkippedStops":"true","withAlerts":"true",
        "realtimeMode":"REALTIME","language":"de",
    }
    result=fetch("/api/v6/plan",params)
    if not isinstance(result,dict) or not isinstance(result.get("itineraries"),list):
        raise RuntimeError("Transitous-Fahrplandaten ungültig.")
    journeys=[]
    for raw in result["itineraries"][:10]:
        converted=transitous_itinerary(raw)
        if converted:
            journeys.append(converted)
    updated=datetime.now(ZoneInfo("Europe/Berlin")).isoformat()
    return {
        "status":"ok" if journeys else "empty",
        "from":start["name"],"to":target["name"],
        "departure":departure,"journeys":journeys[:5],
        "source":"BahnConnections-Datenbasis · Transitous/MOTIS direkt",
        "updatedAt":updated,
        "realtimeStatus":"live" if any(j["realtimeStatus"]=="live" for j in journeys) else "schedule",
        "warnings":["BahnConnections war nicht erreichbar. Die Verbindungen wurden direkt aus Transitous/MOTIS abgerufen."],
        "providerUrl":"https://bahnconnections-de.a-stad.chatgpt.site",
    }
