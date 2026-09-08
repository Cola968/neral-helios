"""Local-first training telemetry with optional Hugging Face Trackio sync.

Telemetry is always written locally as JSONL. Remote Trackio sync is opt-in via
NERAL_TRACKIO_ENABLED=1. No prompts, corpus rows, model weights, checkpoints, or local paths
are sent to Trackio; only numeric training metrics and a small explicit config are logged.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

TRUTHY = {"1", "true", "yes", "on"}


def _enabled(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in TRUTHY


class TrainingObserver:
    def __init__(self, out_dir: Path, run_name: str, config: dict[str, Any], group: str = "training") -> None:
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.out_dir / "telemetry.jsonl"
        self.run_name = run_name
        self.config = dict(config)
        self.group = group
        self._trackio = None
        self._remote = False
        self._write({"event": "observer_start", "run_name": run_name, "config": self.config, "group": group})

        if not _enabled("NERAL_TRACKIO_ENABLED"):
            return
        try:
            import trackio  # type: ignore

            kwargs: dict[str, Any] = {
                "project": os.environ.get("NERAL_TRACKIO_PROJECT", "neral-helios"),
                "name": run_name,
                "group": group,
                "config": self.config,
            }
            space_id = os.environ.get("NERAL_TRACKIO_SPACE_ID", "").strip()
            if space_id:
                kwargs["space_id"] = space_id
            trackio.init(**kwargs)
            self._trackio = trackio
            self._remote = bool(space_id)
            self._write({"event": "trackio_enabled", "remote_space_sync": self._remote})
        except Exception as exc:  # telemetry must never kill training
            self._write({"event": "trackio_init_failed", "error": type(exc).__name__, "message": str(exc)[:300]})

    @property
    def remote_enabled(self) -> bool:
        return self._remote

    def _write(self, payload: dict[str, Any]) -> None:
        row = {"time_unix": time.time(), **payload}
        with self.path.open("a", encoding="utf8") as f:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def log(self, step: int, metrics: dict[str, Any]) -> None:
        clean: dict[str, float | int] = {"step": int(step)}
        for key, value in metrics.items():
            if isinstance(value, bool):
                clean[key] = int(value)
            elif isinstance(value, (int, float)):
                clean[key] = value
        self._write({"event": "metrics", **clean})
        if self._trackio is not None:
            try:
                self._trackio.log(clean)
            except Exception as exc:
                self._write({"event": "trackio_log_failed", "step": int(step), "error": type(exc).__name__})

    def alert(self, title: str, text: str, level: str = "WARN", step: int | None = None) -> None:
        payload = {"event": "alert", "title": title, "text": text[:500], "level": level.upper()}
        if step is not None:
            payload["step"] = int(step)
        self._write(payload)
        if self._trackio is None or not hasattr(self._trackio, "alert"):
            return
        try:
            level_value = level.upper()
            enum = getattr(self._trackio, "AlertLevel", None)
            if enum is not None and hasattr(enum, level_value):
                level_value = getattr(enum, level_value)
            self._trackio.alert(title=title, text=text[:500], level=level_value)
        except Exception as exc:
            self._write({"event": "trackio_alert_failed", "error": type(exc).__name__})

    def finish(self, status: str = "completed") -> None:
        self._write({"event": "observer_finish", "status": status})
        if self._trackio is not None:
            try:
                self._trackio.finish()
            except Exception as exc:
                self._write({"event": "trackio_finish_failed", "error": type(exc).__name__})
