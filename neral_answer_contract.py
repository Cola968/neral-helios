"""Opt-in, frozen regression-only answer contract; no model or tool changes.

The original 100-case 2026-11-12 blind evaluation remains immutable.
This module is only for a later *known development corpus* rerun.
"""
from __future__ import annotations

import re
from neral_identity import build_identity_prompt

BENCHMARK_FORMAT_V1=(
    "\n\nEVALUATION ANSWER CONTRACT: This benchmark expects one exact integer. "
    "Work briefly and accurately; avoid long explanations or verbose thinking. "
    "Your final line MUST be exactly FINAL_ANSWER: <integer> (one integer; "
    "no units or text after this line). If you cannot solve the task, "
    "do not invent a numerical result. "
    "Follow the same instructions for all tasks and all model variants."
)
_END_RE=re.compile(r"FINAL_ANSWER: ([+-]?\d{1,15})\Z")

def formatted_benchmark_system()->str:
    """One and only one shared system prompt for both paired model arms."""
    return build_identity_prompt("web","local_foundation_not_neral_adapter")+BENCHMARK_FORMAT_V1

def parse_terminal_final_integer(output:object)->int|None:
    """Recognize precisely one unambiguous terminal FINAL_ANSWER line.

    No searching arbitrary reasoning or numeric intermediate values.
    Transport truncations with done_reason=length must already be rejected
    by the Ollama client; this parser cannot certify response completeness.
    """
    if type(output) is not str or not output.strip() or len(output)>32768:
        return None
    text=output.strip()
    if text.count("FINAL_ANSWER:")!=1:
        return None
    last=text.splitlines()[-1]
    match=_END_RE.fullmatch(last)
    if not match:return None
    return int(match.group(1))

def format_verified_terminal_answer(output:str, *,independently_verified:bool)->str:
    """Only change serialization for already verified exact integer values."""
    if not independently_verified or type(output) is not str:
        return output
    if parse_terminal_final_integer(output) is not None:
        return output
    if re.fullmatch(r"[+-]?\d{1,15}",output.strip()):
        return "FINAL_ANSWER: "+output.strip()
    return output
