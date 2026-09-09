from __future__ import annotations

"""Immutable Hugging Face source lock for Neral 350M Frontier production data.

Pilot discovery may inspect live Hub metadata, but any release-candidate corpus must call this
module and use exactly the revisions, configs, splits and licenses committed in source_lock.json.
"""

import json
import re
from pathlib import Path

LOCK_PATH = Path(__file__).with_name("source_lock.json")
HEX40 = re.compile(r"^[0-9a-f]{40}$")


def load_lock() -> dict:
    obj = json.loads(LOCK_PATH.read_text(encoding="utf8"))
    if obj.get("schema") != 1 or obj.get("architecture") != "frontier350-v1":
        raise RuntimeError("invalid Frontier source lock schema")
    sources = obj.get("sources")
    if not isinstance(sources, dict) or not sources:
        raise RuntimeError("Frontier source lock has no sources")
    for name, item in sources.items():
        rev = str(item.get("revision", "")).lower()
        if not HEX40.fullmatch(rev):
            raise RuntimeError(f"source {name} lacks immutable 40-hex revision")
        for key in ("repo", "split", "license", "role"):
            if not str(item.get(key, "")).strip():
                raise RuntimeError(f"source {name} missing {key}")
    return obj


def pinned_revision(*, name: str, repo: str, config: str | None, split: str, license: str) -> str:
    item = load_lock()["sources"].get(name)
    if item is None:
        raise RuntimeError(f"source not present in immutable lock: {name}")
    expected = {
        "repo": repo,
        "config": config,
        "split": split,
        "license": license,
    }
    actual = {key: item.get(key) for key in expected}
    if actual != expected:
        raise RuntimeError(f"source contract drift for {name}: expected={expected}, lock={actual}")
    return str(item["revision"]).lower()


def locked_revisions(specs) -> dict[str, str]:
    return {
        spec.name: pinned_revision(
            name=spec.name,
            repo=spec.repo,
            config=spec.config,
            split=spec.split,
            license=spec.license,
        )
        for spec in specs
    }
