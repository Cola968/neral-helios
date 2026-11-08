"""Strict, on-demand BahnConnections rail connection bridge for NERAL local chat.

No timetable guesses. Outbound traffic contains only route stations, time and
filters after the user explicitly asks for a journey. Never sends chat history.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from ipaddress import ip_address
import json
import os
import re
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener
from zoneinfo import ZoneInfo

BASE = "https://bahnconnections-de.a-stad.chatgpt.site"
DE = ZoneInfo("Europe/Berlin")
CATEGORIES = ["fern", "regional", "sbahn", "ubahn", "tram"]
# Coordinates are the curated station points in Cola968/BahnConnections/app/network-data.ts.
KNOWN = {
    "berlin hbf": ("Berlin Hbf", 52.5251, 13.3694),
    "stuttgart hbf": ("Stuttgart Hbf", 48.7830, 9.1816),
    "hamburg hbf": ("Hamburg Hbf", 53.5528, 10.0064),
    "munchen hbf": ("München Hbf", 48.140, 11.561),
    "koln hbf": ("Köln Hbf", 50.9430, 6.9587),
    "frankfurt main hbf": ("Frankfurt (Main) Hbf", 50.1071, 8.6638),
    "leipzig hbf": ("Leipzig Hbf", 51.3451, 12.3822),
    "hannover hbf": ("Hannover Hbf", 52.3772, 9.7410),
}
CITY_ALIASES = {
    "berlin": "berlin hbf", "stuttgart": "stuttgart hbf",
    "hamburg": "hamburg hbf", "munchen": "munchen hbf",
    "koln": "koln hbf", "leipzig": "leipzig hbf",
    "hannover": "hannover hbf", "frankfurt": "frankfurt main hbf",
    "frankfurt main": "frankfurt main hbf",
}
MAX_UPSTREAM = 2_000_000


def normalize_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name.lower())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.replace("ß", "ss")
    text = re.sub(r"\b(hauptbahnhof|hauptbf|hbf\.)\b", "hbf", text)
    text = re.sub(r"^s\+u\s+", "", text)
    return re.sub(r"[^a-z0-9 ]+", " ", text).strip()


def valid_name(value: object) -> bool:
    return (isinstance(value, str) and 2 <= len(value.strip()) <= 80 and
            not any(char in value for char in "/\\<>{}\r\n") and
            not re.search(r"(?:https?:|www\.)", value, re.I))


def station_for_name(name: str) -> dict | None:
    key = normalize_name(name)
    key = CITY_ALIASES.get(key, key)
    if key not in KNOWN:
        return None
    label, lat, lon = KNOWN[key]
    return {"name": label, "lat": lat, "lon": lon, "country": "DE"}


def provider_base(env: dict | None = None) -> str:
    environment = os.environ if env is None else env
    base = environment.get("NERAL_BAHNCONNECTIONS_URL", BASE)
    url = urlsplit(base)
    if (url.scheme != "https" or not url.hostname or not url.hostname.count(".") or
            url.username or url.password or url.path not in ("", "/") or
            url.query or url.fragment or url.port):
        raise ValueError("BahnConnections-Quelle muss eine HTTPS-Domain ohne Pfad sein.")
    try:
        ip_address(url.hostname)
    except ValueError:
        if url.hostname.endswith((".local", ".internal")) or url.hostname == "localhost":
            raise ValueError("Keine privaten Bahndaten-Proxyhosts erlaubt.")
    else:
        raise ValueError("IP-Adressen als Fahrplanquelle sind nicht erlaubt.")
    return base.rstrip("/")


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RuntimeError("BahnConnections-Weiterleitung abgelehnt.")


def fetch_provider_json(url: str, body: dict | None = None, timeout: int = 12) -> dict:
    base = provider_base()
    if not url.startswith(base + "/api/"):
        raise ValueError("Externes Ziel nicht erlaubt.")
    request = Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "NERAL-Local-User-Directed-Journey/1.0",
    })
    if body is not None:
        request = Request(url, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Accept": "application/json", "Content-Type": "application/json",
                     "User-Agent": "NERAL-Local-User-Directed-Journey/1.0"},
            method="POST")
    opener = build_opener(NoRedirect)
    try:
        with opener.open(request, timeout=timeout) as response:
            if not str(response.headers.get("Content-Type", "")).lower().startswith("application/json"):
                raise RuntimeError("BahnConnections lieferte kein JSON.")
            content = response.read(MAX_UPSTREAM + 1)
    except (HTTPError, URLError, OSError) as exc:
        raise RuntimeError("BahnConnections ist aktuell nicht erreichbar.") from exc
    if len(content) > MAX_UPSTREAM:
        raise RuntimeError("Die Fahrplandatenantwort ist zu groß.")
    try:
        data = json.loads(content)
    except (ValueError, UnicodeError) as exc:
        raise RuntimeError("BahnConnections lieferte ungültige Daten.") from exc
    if not isinstance(data, dict):
        raise RuntimeError("BahnConnections-Datenformat unerwartet.")
    return data


def _resolve_station(name: str, fetch_json, base: str) -> dict:
    known = station_for_name(name)
    if known is not None:
        return known
    response = fetch_json(base + "/api/stations/search?" + urlencode({"q": name}), timeout=10)
    stations = response.get("stations", [])
    if not isinstance(stations, list):
        raise RuntimeError("Bahnhofsuche lieferte keine verwendbaren Ergebnisse.")
    key = normalize_name(name)
    exact = [s for s in stations[:16] if isinstance(s, dict) and
             isinstance(s.get("name"), str) and normalize_name(s["name"]) == key and
             isinstance(s.get("lat"), (float, int)) and
             isinstance(s.get("lon"), (float, int))]
    if len(exact) != 1:
        raise ValueError("Bahnhof nicht eindeutig gefunden. Bitte den vollständigen Bahnhofsnamen angeben.")
    point = exact[0]
    return {"name": point["name"], "lat": point["lat"], "lon": point["lon"],
            "id": point.get("id", ""), "transitousId": point.get("transitousId", ""),
            "country": point.get("country", "DE")}


def _iso(value: object) -> bool:
    if not isinstance(value, str) or len(value) > 40:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed.tzinfo is not None and parsed.utcoffset() is not None
    except ValueError:
        return False


def _point(p: object) -> list | None:
    if (not isinstance(p, (list, tuple)) or len(p) != 2 or
            not all(isinstance(x, (int, float)) and not isinstance(x, bool) for x in p)):
        return None
    lat, lon = p
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return [float(lat), float(lon)]


def _stop(value: object) -> dict | None:
    if not isinstance(value, dict) or not valid_name(value.get("name")):
        return None
    stop = {"name": value["name"][:80]}
    for field in ("lat", "lon"):
        x = value.get(field)
        stop[field] = (x if isinstance(x, (float, int)) and not isinstance(x, bool)
                       and abs(x) <= (90 if field == "lat" else 180) else None)
    for field in ("track", "scheduledTrack"):
        x = value.get(field)
        if isinstance(x, str) and len(x) <= 20:
            stop[field] = x
    return stop


def normalize_journey(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    start, end = raw.get("startTime"), raw.get("endTime")
    if not _iso(start) or not _iso(end):
        return None
    if datetime.fromisoformat(end.replace("Z", "+00:00")) <= datetime.fromisoformat(start.replace("Z", "+00:00")):
        return None
    legs = []
    for piece in raw.get("legs", [])[:12]:
        if not isinstance(piece, dict) or not _iso(piece.get("startTime")) or not _iso(piece.get("endTime")):
            return None
        from_stop = _stop(piece.get("from"))
        to_stop = _stop(piece.get("to"))
        if not from_stop or not to_stop:
            return None
        pts = [_point(x) for x in piece.get("points", [])[:1800] if isinstance(x, (list, tuple))]
        points = [x for x in pts if x]
        name = piece.get("name")
        category = piece.get("category")
        if not isinstance(name, str) or not name.strip():
            return None
        legs.append({
            "name": name[:55], "category": category if category in CATEGORIES else "walk",
            "from": from_stop, "to": to_stop,
            "startTime": piece["startTime"], "endTime": piece["endTime"],
            "scheduledStartTime": piece.get("scheduledStartTime") if _iso(piece.get("scheduledStartTime")) else piece["startTime"],
            "scheduledEndTime": piece.get("scheduledEndTime") if _iso(piece.get("scheduledEndTime")) else piece["endTime"],
            "realtime": piece.get("realtime") is True, "cancelled": piece.get("cancelled") is True,
            "points": points,
        })
    if not legs or len(legs) > 9:
        return None
    transfers = raw.get("transfers")
    if not isinstance(transfers, int) or isinstance(transfers, bool) or not 0 <= transfers <= 8:
        return None
    warnings = raw.get("warnings") if isinstance(raw.get("warnings"), list) else []
    label = raw.get("sourceLabel")
    return {
        "startTime": start, "endTime": end,
        "scheduledStartTime": raw.get("scheduledStartTime") if _iso(raw.get("scheduledStartTime")) else start,
        "scheduledEndTime": raw.get("scheduledEndTime") if _iso(raw.get("scheduledEndTime")) else end,
        "durationSeconds": int((datetime.fromisoformat(end.replace("Z", "+00:00")) -
                                datetime.fromisoformat(start.replace("Z", "+00:00"))).total_seconds()),
        "transfers": transfers, "cancelled": raw.get("cancelled") is True,
        "realtimeStatus": raw.get("realtimeStatus") if raw.get("realtimeStatus") in ("live", "schedule", "partial") else "partial",
        "sourceLabel": label[:70] if isinstance(label, str) else "BahnConnections",
        "legs": legs,
        "warnings": [s[:180] for s in warnings[:4] if isinstance(s, str)],
    }


def _departure(payload: dict) -> str:
    value = payload.get("departure")
    if value is not None:
        if not _iso(value):
            raise ValueError("Ungültiges Abfahrtsdatum; ISO mit Zeitzone verwenden.")
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
        now = datetime.now(DE)
        if not now - timedelta(hours=2) <= when <= now + timedelta(days=21):
            raise ValueError("Reisedatum außerhalb des zulässigen Zeitfensters.")
        return when.isoformat()
    when = datetime.now(DE)
    day = payload.get("when")
    if day is not None:
        if day not in ("heute", "morgen", "übermorgen"):
            raise ValueError("Unbekannter Reisetag.")
        days = {"heute": 0, "morgen": 1, "übermorgen": 2}[day]
        when += timedelta(days=days)
    hour = payload.get("time")
    if hour is not None:
        if not isinstance(hour, str) or not re.fullmatch(r"(?:[01]\d|2[0-3]):[0-5]\d", hour):
            raise ValueError("Uhrzeit muss HH:MM sein.")
        h, m = map(int, hour.split(":"))
        when = when.replace(hour=h, minute=m, second=0, microsecond=0)
        if day is None and when < datetime.now(DE):
            # "ab 8 Uhr" at 18:00 means the upcoming 08:00, not earlier today.
            when += timedelta(days=1)
        elif day == "heute" and when < datetime.now(DE):
            raise ValueError("Die angegebene Abfahrtszeit liegt bereits in der Vergangenheit.")
    return when.isoformat()


def plan_journeys(payload: object, *, fetch_json=None) -> dict:
    if not isinstance(payload, dict) or not {"from", "to"} <= payload.keys() or not payload.keys() <= {"from", "to", "departure", "when", "time"}:
        raise ValueError("Nur Abfahrt, Ziel und gewünschte Reisezeit erlaubt.")
    start, target = payload["from"], payload["to"]
    if not valid_name(start) or not valid_name(target) or normalize_name(start) == normalize_name(target):
        raise ValueError("Zwei unterschiedliche, gültige Bahnhöfe erforderlich.")
    base = provider_base()
    departure = _departure(payload)
    allow_fallback = fetch_json is None
    fetch_json = fetch_json or fetch_provider_json
    # Pure station lookup; no full conversation, model data, files or browser tokens sent.
    try:
        from_station = _resolve_station(start, fetch_json, base)
        to_station = _resolve_station(target, fetch_json, base)
        if from_station["name"] == to_station["name"]:
            raise ValueError("Start und Ziel müssen verschieden sein.")
        raw = fetch_json(base + "/api/journeys", body={
            "from": from_station, "to": to_station, "departure": departure,
            "arriveBy": False, "bike": False, "wheelchair": False,
            "maxTransfers": 4, "minTransferMinutes": 0, "categories": CATEGORIES,
        }, timeout=12)
    except RuntimeError:
        if not allow_fallback:
            raise
        # Real timetable from BahnConnections' own primary provider.
        # Never fabricate a route when either upstream is unavailable.
        from neral_transitous_fallback import plan_transitous_direct
        from_label = station_for_name(start)
        to_label = station_for_name(target)
        return plan_transitous_direct(
            from_label["name"] if from_label else start,
            to_label["name"] if to_label else target,
            departure,
        )
    journeys_raw = raw.get("journeys", [])
    if not isinstance(journeys_raw, list):
        raise RuntimeError("BahnConnections lieferte kein Reiseergebnis.")
    journeys = [normalized for j in journeys_raw[:12] if (normalized := normalize_journey(j))]
    updated = raw.get("updatedAt")
    realtime = raw.get("realtimeStatus")
    return {
        "status": "ok" if journeys else "empty",
        "from": from_station["name"], "to": to_station["name"],
        "departure": departure,
        "journeys": journeys[:5],
        "source": "BahnConnections · " + str(raw.get("source", "Transitous / MOTIS"))[:70],
        "updatedAt": updated if _iso(updated) else None,
        "realtimeStatus": realtime if realtime in ("live", "schedule", "partial") else "partial",
        "warnings": [x[:200] for x in raw.get("warnings", [])[:4] if isinstance(x, str)]
        if isinstance(raw.get("warnings", []), list) else [],
        "providerUrl": base,
    }
