"""Opt-in numeric-only local NERAL telemetry, never captures questions or replies.

Scopes belong to one HTTP request. No files, persistence, threads, or network.
Production behavior and GPU memory are unchanged when diagnostics are off.
"""
from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from collections import Counter
import time

_CURRENT = ContextVar("neral_v42_telemetry", default=None)
_ALLOWED_TOOLS = frozenset(("kiwix", "web_v4", "python_z3"))


@dataclass
class RequestTelemetry:
    started: float = field(default_factory=time.monotonic)
    model_load_ms: float = 0.0
    model_load_cold: int = 0
    model_load_warm: int = 0
    model_switch_ms: float = 0.0
    tool_times: dict = field(default_factory=dict)
    generation_calls: list = field(default_factory=list)
    placement: dict = field(default_factory=dict)
    peak_cuda_allocated_mib: float | None = None
    peak_cuda_reserved_mib: float | None = None

    def load(self, was_loaded, milliseconds, model):
        self.model_load_ms += max(0.0, milliseconds)
        if was_loaded:
            self.model_load_warm += 1
        else:
            self.model_load_cold += 1
        if model.loaded:
            self.placement = summarize_placement(model)

    def generation(self, *, input_tokens, system_tokens, max_new_tokens,
                   first_token_at, started_at, finished_at, count):
        if len(self.generation_calls) >= 12:
            return
        ms = max(0., (finished_at - started_at) * 1000.)
        ttft = (round((first_token_at - started_at)*1000., 2)
                if first_token_at is not None else None)
        # Every token including EOS counts toward total generated tokens. The
        # first token's prefill cost must not inflate steady-state tokens/sec.
        remaining_s = finished_at - first_token_at if first_token_at is not None else 0
        tps = (round((count-1)/remaining_s, 2)
               if count is not None and count > 1 and remaining_s > 0 else None)
        self.generation_calls.append({
            "input_tokens": input_tokens, "system_prompt_tokens": system_tokens,
            "max_new_tokens": max_new_tokens, "ttft_ms": ttft,
            "gen_tokens": count, "generation_ms": round(ms, 2),
            "tokens_per_sec": tps,
        })

    def tool(self, name, duration_ms):
        if name not in _ALLOWED_TOOLS:
            raise ValueError("Unknown diagnostic tool classification")
        self.tool_times[name] = round(self.tool_times.get(name, 0.) + max(0., duration_ms), 2)

    def cuda(self, torch):
        if torch.cuda.is_available():
            self.peak_cuda_allocated_mib = round(torch.cuda.max_memory_allocated(0)/(1024**2), 2)
            self.peak_cuda_reserved_mib = round(torch.cuda.max_memory_reserved(0)/(1024**2), 2)

    def summary(self):
        calls = self.generation_calls
        return {
            "model_load_ms": round(self.model_load_ms, 2),
            "model_load_cold": self.model_load_cold,
            "model_load_warm": self.model_load_warm,
            "model_switch_ms": round(self.model_switch_ms, 2),
            "tool_ms": round(sum(self.tool_times.values()), 2),
            "tool_breakdown_ms": dict(self.tool_times),
            "ttft_ms": calls[0]["ttft_ms"] if calls else None,
            "system_prompt_tokens": calls[0]["system_prompt_tokens"] if calls else None,
            "input_tokens": calls[0]["input_tokens"] if calls else None,
            "gen_tokens": sum(int(c["gen_tokens"] or 0) for c in calls),
            "tokens_per_sec": (
                round(sum(max(0,c["gen_tokens"]-1) for c in calls if c["gen_tokens"] is not None) /
                      (sum(max(0,(c["generation_ms"]-(c["ttft_ms"] or 0)))/1000
                           for c in calls if c["gen_tokens"] is not None)), 2)
                if any(c["gen_tokens"] is not None for c in calls)
                and sum(max(0,(c["generation_ms"]-(c["ttft_ms"] or 0)))/1000
                        for c in calls if c["gen_tokens"] is not None) > 0
                else None),
            "generation_calls": list(calls),
            "placement": dict(self.placement),
            "peak_cuda_allocated_mib": self.peak_cuda_allocated_mib,
            "peak_cuda_reserved_mib": self.peak_cuda_reserved_mib,
            "total_wall_ms": round(max(0,(time.monotonic()-self.started)*1000), 2),
        }


def current():
    return _CURRENT.get()


def begin_request():
    metric = RequestTelemetry()
    token = _CURRENT.set(metric)
    return metric, token


def end_request(token):
    _CURRENT.reset(token)


def timed_tool(name, fn):
    """Wrap a tool callable with telemetry; keep its positional/keyword ABI."""
    if fn is None:
        return None
    if name not in _ALLOWED_TOOLS or not callable(fn):
        raise ValueError("Invalid diagnostic tool callback")
    def measured(*args, **kwargs):
        mark = time.monotonic()
        try:
            return fn(*args, **kwargs)
        finally:
            metric = current()
            if metric is not None:
                metric.tool(name, (time.monotonic()-mark)*1000)
    return measured


def summarize_placement(model):
    """Count actual parameter locations; no checkpoint paths or layer names."""
    obj = getattr(model, "_model", None)
    base = getattr(model, "_base_model", None)
    result = {"gpu_parameter_tensors": 0, "cpu_parameter_tensors": 0,
              "disk_parameter_tensors": 0, "other_parameter_tensors": 0,
              "hf_gpu_module_entries": 0, "hf_cpu_module_entries": 0,
              "hf_disk_module_entries": 0,
              "quantization": "4-bit NF4 double-quant",
              "context_limit_tokens":4096}
    if obj is None:
        return result
    mapping = getattr(obj, "hf_device_map", None) or getattr(base, "hf_device_map", None) or {}
    if isinstance(mapping, dict):
        for dest in mapping.values():
            label = str(dest).lower()
            key = ("hf_cpu_module_entries" if label == "cpu" else
                   "hf_disk_module_entries" if label == "disk" else
                   "hf_gpu_module_entries" if label in ("0","cuda","cuda:0") else None)
            if key:
                result[key] += 1
    # Parameter.device exposes actual placement; no tensor values are read.
    try:
        for parameter in obj.parameters():
            dev = parameter.device.type
            if dev == "cuda":
                result["gpu_parameter_tensors"] += 1
            elif dev == "cpu":
                result["cpu_parameter_tensors"] += 1
            elif dev == "meta":
                result["disk_parameter_tensors"] += 1
            else:
                result["other_parameter_tensors"] += 1
    except (AttributeError, ValueError, RuntimeError):
        result["other_parameter_tensors"] = -1
    return result


class FirstTokenStopper:
    """Transformers calls this AFTER the first sampled token was appended."""
    def __init__(self, base):
        self.base = base
        self.first_at = None
    def __call__(self, input_ids, scores, **kwargs):
        if self.first_at is None:
            self.first_at = time.monotonic()
        return self.base(input_ids, scores, **kwargs)
