"""Opt-in, local-only NERAL desktop interaction.

Screenshots stay inside the process/browser session; no model or external API
receives frames. Every action is validated, previewed and individually
approved with a short-lived one-use capability. No background autonomy.
"""
from __future__ import annotations

import base64
import math
import re
import secrets
import sys
import threading
import time
from collections import deque
from io import BytesIO


class ComputerUseError(ValueError):
    """Public safe error without credentials or typed text."""


class WindowsDesktop:
    """Optional local primary-monitor backend. Must never start in CI."""

    def __init__(self):
        if sys.platform != "win32":
            raise ComputerUseError("Computer Use benötigt Windows.")
        try:
            import pyautogui
        except (ImportError, OSError) as exc:
            raise ComputerUseError("Optionales Paket pyautogui fehlt.") from exc
        self.gui = pyautogui
        self.gui.FAILSAFE = True
        self.gui.PAUSE = 0.15

    def screenshot(self) -> dict:
        # Image exists in memory only; never write a screenshot to disk.
        screenshot = self.gui.screenshot()
        actual_width, actual_height = screenshot.size
        if not 1 <= actual_width <= 16384 or not 1 <= actual_height <= 16384:
            raise ComputerUseError("Ungültige Bildschirmauflösung.")
        display = screenshot.copy()
        display.thumbnail((1280, 800))
        stream = BytesIO()
        display.convert("RGB").save(stream, format="JPEG", quality=65, optimize=True)
        return {
            "image_base64": base64.b64encode(stream.getvalue()).decode("ascii"),
            "media_type": "image/jpeg",
            "width": actual_width,
            "height": actual_height,
            "preview_width": display.width,
            "preview_height": display.height,
        }

    def execute(self, action: dict) -> None:
        kind = action["action"]
        gui = self.gui
        if kind == "click":
            gui.click(action["x"], action["y"])
        elif kind == "move":
            gui.moveTo(action["x"], action["y"], duration=0.15)
        elif kind == "scroll":
            gui.scroll(action["amount"])
        elif kind == "type":
            gui.write(action["text"], interval=0.02)
        elif kind == "press":
            gui.press(action["key"])
        else:
            raise ComputerUseError("Nicht erlaubte Computeraktion.")


class ComputerUse:
    MAX_ACTIONS = 30
    MAX_SCREEN_BASE64 = 1_500_000
    SESSION_SECONDS = 300
    APPROVAL_SECONDS = 20
    ALLOWED_KEYS = frozenset({
        "Escape", "Tab", "Enter", "Up", "Down", "Left",
        "Right", "Home", "End", "Backspace", "Delete",
    })

    def __init__(self, backend, *, clock=None):
        self._backend = backend
        self._clock = clock or time.monotonic
        self._lock = threading.RLock()
        self._enabled = False
        self._deadline = 0.
        self._pending = None
        self._dimensions = (0, 0)
        self._calls = deque()

    def _require_enabled(self):
        if not self._enabled or self._clock() >= self._deadline:
            self._enabled = False
            self._pending = None
            raise ComputerUseError("Computer Use ist nicht freigegeben oder abgelaufen.")

    def status(self) -> dict:
        with self._lock:
            live = self._enabled and self._clock() < self._deadline
            if not live:
                self._pending = None
                self._enabled = False
            return {
                "enabled": live,
                "computer_use": True,
                "mode": "local_explicit_one_action_at_a_time",
                "expires_in_seconds": max(0, int(self._deadline - self._clock())) if live else 0,
                "cloud_screenshots": False,
                "autonomous_actions": False,
            }

    def _validate_frame(self, frame: dict) -> None:
        if not isinstance(frame, dict):
            raise ComputerUseError("Bildschirmdaten fehlen.")
        width, height = frame.get("width"), frame.get("height")
        if type(width) is not int or type(height) is not int or not 1 <= width <= 16384 or not 1 <= height <= 16384:
            raise ComputerUseError("Ungültige Bildschirmabmessungen.")
        data = frame.get("image_base64")
        if not isinstance(data, str) or len(data) > self.MAX_SCREEN_BASE64:
            raise ComputerUseError("Bildschirmbild zu groß.")
        try:
            raw = base64.b64decode(data, validate=True)
        except ValueError as exc:
            raise ComputerUseError("Ungültige Bilddaten.") from exc
        if len(raw) > 1_200_000 or frame.get("media_type") not in {"image/png", "image/jpeg"}:
            raise ComputerUseError("Ungültiges Bildformat.")
        self._dimensions = (width, height)

    def start(self) -> dict:
        with self._lock:
            # Called only in response to the user's explicit UI activation.
            frame = self._backend.screenshot()
            self._validate_frame(frame)
            self._enabled = True
            self._deadline = self._clock() + self.SESSION_SECONDS
            self._pending = None
            self._calls.clear()
            return self.status()

    def stop(self) -> dict:
        with self._lock:
            self._enabled = False
            self._pending = None
            self._deadline = 0.
            return self.status()

    def screenshot(self) -> dict:
        with self._lock:
            self._require_enabled()
            frame = self._backend.screenshot()
            self._validate_frame(frame)
            return frame

    def _validate_action(self, payload: dict) -> dict:
        if not isinstance(payload, dict):
            raise ComputerUseError("Aktion muss ein JSON-Objekt sein.")
        kind = payload.get("action")
        if kind in {"click", "move"}:
            if set(payload) != {"action", "x", "y"}:
                raise ComputerUseError("Unzulässige Mausparameter.")
            x, y = payload["x"], payload["y"]
            w, h = self._dimensions
            if type(x) is not int or type(y) is not int or not 0 <= x < w or not 0 <= y < h:
                raise ComputerUseError("Mausposition außerhalb des Bildschirms.")
            return {"action": kind, "x": x, "y": y}
        if kind == "scroll":
            amount = payload.get("amount")
            if set(payload) != {"action", "amount"} or type(amount) is not int or not -3 <= amount <= 3 or amount == 0:
                raise ComputerUseError("Unzulässige Scrollbewegung.")
            return {"action": kind, "amount": amount}
        if kind == "press":
            key = payload.get("key")
            if set(payload) != {"action", "key"} or key not in self.ALLOWED_KEYS:
                raise ComputerUseError("Tastenkombination nicht erlaubt.")
            return {"action": kind, "key": key}
        if kind == "type":
            value = payload.get("text")
            if (set(payload) != {"action", "text"} or not isinstance(value, str) or
                    not 1 <= len(value) <= 200 or any(ord(char) < 32 or ord(char) == 127 for char in value)):
                raise ComputerUseError("Nur kurzer Text ohne Steuerzeichen ist zulässig.")
            return {"action": kind, "text": value}
        raise ComputerUseError("Diese Computeraktion ist nicht erlaubt.")

    def preview(self, payload: dict) -> dict:
        with self._lock:
            self._require_enabled()
            recent = self._clock() - 60
            while self._calls and self._calls[0] < recent:
                self._calls.popleft()
            if len(self._calls) >= self.MAX_ACTIONS:
                raise ComputerUseError("Computer-Use-Aktionslimit erreicht.")
            action = self._validate_action(payload)
            token = secrets.token_urlsafe(24)
            self._pending = (token, action, self._clock() + self.APPROVAL_SECONDS)
            return {
                "approval_token": token,
                "action": action["action"],
                "summary": "Tippen von Text (Inhalt bleibt lokal)" if action["action"] == "type"
                    else "Freizugebende Einzelaktion: " + action["action"],
                "approval_expires_in_seconds": self.APPROVAL_SECONDS,
            }

    def execute(self, token: str) -> dict:
        with self._lock:
            self._require_enabled()
            pending = self._pending
            if (not pending or not isinstance(token, str) or
                    not secrets.compare_digest(token, pending[0]) or self._clock() >= pending[2]):
                raise ComputerUseError("Freigabe fehlt oder ist abgelaufen.")
            self._pending = None  # Consume even if backend errors.
            _, action, _ = pending
            try:
                self._backend.execute(action)
            except Exception as exc:
                # pyautogui FailSafeException must terminate the active session.
                self.stop()
                raise ComputerUseError("Computeraktion abgebrochen oder fehlgeschlagen.") from exc
            self._calls.append(self._clock())
            return {"executed": True, "action": action["action"]}
