"""Obsidian Codex test-first controller contract (CPU-only preparation).

No host execution, no subprocesses, no filesystem writes, no GPU calls.
The sandbox callback MUST be independently isolated, network-disabled and
CPU-only, and audited before integration. A mock report is NOT an attestation.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class TestBundle:
    """Model-proposed tests: only a trusted isolated sandbox may execute them."""
    pytest_module: str
    hypothesis_module: str

    def __post_init__(self):
        for source in (self.pytest_module, self.hypothesis_module):
            if type(source) is not str or not 1 <= len(source.strip()) <= 32768:
                raise ValueError("Missing or oversized test source")


@dataclass(frozen=True)
class SandboxReport:
    """From an independently isolated CPU executor; never an LLM self-report."""
    pytest_passed: bool
    hypothesis_passed: bool
    collected_pytest: int
    collected_hypothesis: int
    isolated: bool
    network_disabled: bool
    gpu_visible: bool
    stacktrace: str = ""

    def __post_init__(self):
        if any(type(getattr(self, name)) is not bool for name in (
                "pytest_passed", "hypothesis_passed", "isolated",
                "network_disabled", "gpu_visible")):
            raise ValueError("Invalid sandbox flags")
        if any(type(getattr(self, name)) is not int or not 0 <= getattr(self, name) <= 10000
               for name in ("collected_pytest", "collected_hypothesis")):
            raise ValueError("Invalid test collection counts")
        if type(self.stacktrace) is not str or len(self.stacktrace) > 16384:
            raise ValueError("Unbounded stacktrace")


@dataclass(frozen=True)
class CodexResult:
    status: str
    attempts: int
    repairs: int


def run_test_first(
    build_tests: Callable[[], TestBundle],
    sandbox: Callable[[TestBundle, str | None], SandboxReport],
    propose_patch: Callable[[str, int], str],
    *, max_repairs: int = 2,
) -> CodexResult:
    """CPU-only controller; execution and patch application belong to sandbox.

    Generates tests first; red result supplies real traceback to bounded patch
    proposals. Both pytest and Hypothesis must actually collect and pass.
    This stage does not supply, launch or certify an executor.
    """
    if type(max_repairs) is not int or not 0 <= max_repairs <= 2:
        raise ValueError("Max two repair cycles")
    bundle = build_tests()
    if not isinstance(bundle, TestBundle):
        raise ValueError("Test-first bundle required")
    patch = None
    for attempt in range(max_repairs + 1):
        report = sandbox(bundle, patch)
        if not isinstance(report, SandboxReport):
            raise ValueError("Trusted sandbox report required")
        if not report.isolated or not report.network_disabled or report.gpu_visible:
            raise ValueError("CPU-only networkless sandbox attestation missing")
        if report.collected_pytest < 1 or report.collected_hypothesis < 1:
            raise ValueError("Both pytest and Hypothesis must collect real tests")
        if report.pytest_passed and report.hypothesis_passed:
            return CodexResult("already_green" if attempt == 0 else "passed",
                               attempts=attempt + 1, repairs=attempt)
        if attempt >= max_repairs or not report.stacktrace.strip():
            return CodexResult("unverified", attempts=attempt + 1, repairs=attempt)
        patch = propose_patch(report.stacktrace, attempt + 1)
        if type(patch) is not str or not 0 < len(patch.strip()) <= 262144:
            return CodexResult("unverified", attempts=attempt + 1, repairs=attempt)
    raise AssertionError("Bounded repair loop unreachable")
