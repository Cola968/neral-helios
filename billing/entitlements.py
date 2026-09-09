"""Plan and entitlement helpers for Helios billing.

This module deliberately does not call Stripe. Payment providers are adapters;
Helios features are granted from normalized internal plan keys.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


_PLAN_FILE = Path(__file__).with_name("plans.json")
_VALID_PLANS = {"free", "plus", "pro"}


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    with _PLAN_FILE.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    plans = data.get("plans", {})
    missing = _VALID_PLANS.difference(plans)
    if missing:
        raise RuntimeError(f"Missing billing plans: {sorted(missing)}")
    return data


def normalize_plan(plan: str | None) -> str:
    """Return a safe internal plan key, defaulting unknown values to free."""
    if not isinstance(plan, str):
        return "free"
    value = plan.strip().lower()
    return value if value in _VALID_PLANS else "free"


def entitlements_for(plan: str | None) -> dict[str, Any]:
    """Return a copy of the configured feature entitlements for a plan."""
    key = normalize_plan(plan)
    return dict(_catalog()["plans"][key]["entitlements"])


def has_entitlement(plan: str | None, feature: str) -> bool:
    """Return True only for explicitly truthy boolean entitlements.

    Multi-level string entitlements (for example ``limited`` or ``full``)
    should be inspected with ``entitlement_value`` instead of being coerced.
    """
    return entitlements_for(plan).get(feature) is True


def entitlement_value(plan: str | None, feature: str, default: Any = None) -> Any:
    return entitlements_for(plan).get(feature, default)
