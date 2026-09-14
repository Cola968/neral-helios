from __future__ import annotations

"""Pure policy helpers for Scale V6 StreamGuard.

This module intentionally contains no training-loop wiring. It is safe to import and unit
test independently and does not affect Scale V5. The policy is conservative: fixed
physical execution profiles per resident segment, bounded memory lifetime, and hysteretic
retention pressure.
"""

from dataclasses import dataclass

MIB = 1024 * 1024


@dataclass(frozen=True)
class ExecutionProfile:
    name: str
    chunks: tuple[int, ...]
    min_free_mib: int
    target_reserved_mib: int


PROFILES = (
    ExecutionProfile("P48", (48,), 2200, 8704),
    ExecutionProfile("P32", (32, 16), 1600, 8192),
    ExecutionProfile("P24", (24, 24), 1200, 7680),
)


@dataclass
class MemoryWatchdog:
    baseline_reserved_mib: int
    max_growth_mib: int = 768
    min_free_mib: int = 1536
    max_segment_tokens: int = 50_000_000
    oom_retries: int = 0

    def needs_restart(self, *, reserved_mib: int, free_mib: int, segment_tokens: int) -> tuple[bool, str | None]:
        if segment_tokens >= self.max_segment_tokens:
            return True, "segment_token_limit"
        if reserved_mib - self.baseline_reserved_mib >= self.max_growth_mib:
            return True, "reserved_vram_growth"
        if free_mib < self.min_free_mib:
            return True, "free_vram_floor"
        if self.oom_retries >= 2:
            return True, "repeated_oom_retry"
        return False, None


@dataclass
class RetentionController:
    """Hysteretic pressure controller for already-produced 5M health evidence."""

    pressure: int = 0
    good_streak: int = 0
    max_pressure: int = 5

    def observe(
        self,
        *,
        healthy: bool,
        validation_delta: float,
        regressions: int,
        gains: int,
        margin_delta: float,
        rejected: bool = False,
    ) -> int:
        danger = (
            rejected
            or not healthy
            or validation_delta > 0.045
            or (regressions - gains) > 0
            or margin_delta < -0.015
        )
        clearly_good = (
            healthy
            and validation_delta <= 0.02
            and regressions <= gains
            and margin_delta >= -0.005
            and not rejected
        )

        if danger:
            self.pressure = min(self.max_pressure, self.pressure + 1)
            self.good_streak = 0
        elif clearly_good:
            self.good_streak += 1
            if self.good_streak >= 3 and self.pressure > 0:
                self.pressure -= 1
                self.good_streak = 0
        else:
            self.good_streak = 0
        return self.pressure


def choose_profile(*, free_mib: int, reserved_mib: int, previous: str | None = None) -> ExecutionProfile:
    """Choose once per resident segment; never silently upgrade after an OOM downgrade."""

    by_name = {p.name: p for p in PROFILES}
    if previous is not None:
        prev = by_name[previous]
        start = PROFILES.index(prev)
        allowed = PROFILES[start:]
    else:
        allowed = PROFILES

    for profile in allowed:
        if free_mib >= profile.min_free_mib and reserved_mib <= profile.target_reserved_mib:
            return profile
    return PROFILES[-1]


def downgrade_profile(name: str) -> ExecutionProfile:
    by_name = {p.name: i for i, p in enumerate(PROFILES)}
    idx = by_name[name]
    return PROFILES[min(idx + 1, len(PROFILES) - 1)]


def scaled_recipe(base_lr: float, base_lr_end: float, base_replay_boost: float, pressure: int) -> dict[str, float]:
    """Same conservative pressure semantics as V5, exposed for V6 tests."""

    pressure = min(5, max(0, int(pressure)))
    return {
        "lr": base_lr * (0.90 ** pressure),
        "lr_end": base_lr_end * (0.90 ** pressure),
        "replay_boost": min(2.55, base_replay_boost + 0.08 * pressure),
    }


def effective_tps(committed_tokens: int, wall_seconds: float) -> float:
    if committed_tokens < 0 or wall_seconds <= 0:
        raise ValueError("positive wall time and non-negative committed tokens required")
    return committed_tokens / wall_seconds
