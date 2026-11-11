"""Opt-in LIVE same-Qwen3.5 A/B/C 36-case diagnostic on one local RTX 4070.

No GitHub upload, model install, training or release promotion. Only manually
run after a verified GREEN Windows release. Source task families are PUBLIC;
random instances are NOT an independent blind holdout.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import secrets
import time

from scripts.neral_foundation_2048_retest import ORIGINAL_DIGESTS
from scripts.neral_same_base_ab_score import ARMS, MODEL, PROFILE, TOOLS, SameBaseError, score_private_pairs

BASE_URL = "http://127.0.0.1:9690"
AUTHORIZATION = "NERAL_SAME_BASE_3ARM_36_MANUAL_ONLY"
RELEASE_NAME = "NERAL 1.0 release candidate - offline package and LIVE Windows PC smoke"
_EXPR = re.compile(
    r"(?:what is|evaluate|wie viel ist|wieviel ist|rechne|berechne)"
    r"\s*:?\s*([0-9+\-*/().\t ]+)\s*\??\Z", re.I
)

class PilotBlocked(RuntimeError):
    """Sanitized error: no raw prompts, user paths or private answers."""


def release_gate(receipt: object) -> bool:
    """Receipt MUST come from a separately verified GitHub API response.

    This schema check is not a cryptographic authenticity check. Actual
    Actions workflow must validate GitHub release ID server-side as well.
    """
    if (type(receipt) is not dict or
            any(receipt.get(key) != value for key, value in (
                ("name", RELEASE_NAME), ("status", "completed"),
                ("conclusion", "success"), ("head_branch", "main"),
                ("event", "push")))):
        raise PilotBlocked("A successful MAIN Windows release is required before GPU pilot")
    return True


def require_model(inventory: object) -> str:
    expected = ORIGINAL_DIGESTS[MODEL]
    if type(inventory) is not dict or inventory.get(MODEL) != expected:
        raise PilotBlocked("Original installed Qwen3.5:9B model digest is required")
    return expected


def route_extra_arithmetic(question: str) -> dict | None:
    """Experimental C-only grammar: standalone English/German arithmetic.

    Strict entire-input regex + existing safe fraction AST; never execute
    Python, expand algebra guesses or force a solver on natural-language tasks.
    """
    if type(question) is not str or not 1 <= len(question) <= 130 or "\n" in question:
        return None
    match = _EXPR.fullmatch(question.strip())
    if match is None:
        return None
    expr = match.group(1).strip()
    if not expr or "**" in expr or re.search(r"\d{13}", expr):
        return None
    from neral_agent_math import eval_python_expr, MathInputError
    try:
        solved = eval_python_expr(expr)
    except (MathInputError, ValueError, TypeError, OverflowError, RecursionError):
        return None
    if (not isinstance(solved, dict) or solved.get("status") != "PASSED"
            or not isinstance(solved.get("value"), str)
            or not re.fullmatch(r"-?\d{1,15}", solved["value"])):
        return None
    return {
        "text": solved["value"], "tool_used": "neral_math_exact",
        "model_generated": False, "mode": "HIGH",
        "verification_status": "tool_direct", "verification_scope": "numeric_final_only",
        "verification_method": solved.get("method"),
        "calls_used": 0,
    }



def generate_pilot_cases(seed: int) -> list[dict]:
    """Create a DEVELOPMENT-only suite with six tool-eligible arithmetic tasks.

    The other five families come from the published fresh-case generator.
    These six are freshly randomized but their family is public and deliberately
    tests the router's narrow supported domain: never call this a blind holdout.
    """
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise PilotBlocked("Invalid private development case seed")
    from scripts.neral_low_high_fresh import generate_fresh_cases
    original = generate_fresh_cases(seed)
    retained = [
        case for case in original
        if not (case["domain"] == "math" and case["family"] == "blocked_grid_paths")
    ]
    rng = random.Random(seed ^ 0x58_76_2026)
    for index in range(6):
        a, b, c, d = (rng.randint(3001, 99001) for _ in range(4))
        e = rng.randint(101, 99001)
        # Independent Python integer arithmetic as local reference.
        gold = a * b - c * d + e
        question = f"What is ({a} * {b}) - ({c} * {d}) + {e}?"
        retained.append({
            "id": f"fresh-math-arithmetic_affine-{index:02d}",
            "domain": "math", "family": "arithmetic_affine_integer",
            "kind": "integer", "answer": gold, "prompt": question,
        })
    if (len(retained) != 36 or len({c["id"] for c in retained}) != 36
            or any(sum(c["domain"] == domain for c in retained) != 12
                   for domain in ("math", "reasoning", "code_trace"))
            or len({(c["domain"], c["family"]) for c in retained}) != 6):
        raise PilotBlocked("Fresh diagnostic families are incomplete")
    return retained

def checked_high_packet(packet: object) -> str:
    if (type(packet) is not dict or packet.get("model") != MODEL
            or packet.get("inference_profile") != PROFILE
            or packet.get("model_origin") != "local_foundation_not_neral_adapter"
            or packet.get("adapter_loaded") is not False
            or packet.get("mode") != "HIGH"
            or type(packet.get("text")) is not str
            or not packet["text"].strip() or len(packet["text"].encode("utf8")) > 32768
            or packet.get("tool_used") not in TOOLS
            or type(packet.get("model_generated")) is not bool
            or (packet.get("tool_used") is not None and packet["model_generated"] is True
                and not (
                    (packet.get("verification_status") == "tool_observed_unverified"
                     and packet.get("tool_output_reviewed") is True
                     and packet.get("verified") is False)
                    or (packet.get("verification_status") ==
                        "model_extracted_schema_unverified"
                        and packet.get("verification_scope") ==
                        "model_extracted_finite_domain"
                        and packet.get("tool_output_reviewed") is False
                        and packet.get("verified") is False
                        and packet.get("extraction_status") == "accepted")
                    or (packet.get("verification_status") ==
                        "base_fallback_unverified"
                        and packet.get("verification_scope") == "none"
                        and packet.get("tool_output_reviewed") is False
                        and packet.get("verified") is False
                        # This records execution provenance only, NOT a
                        # claim that the fallback answer used tool stdout.
                        and packet.get("tool_calls_triggered") == 1
                        and packet.get("tool_calls_succeeded") == 1)
                ))
            or (packet.get("tool_used") is not None
                and packet["model_generated"] is True
                and (type(packet.get("tool_calls_succeeded")) is not int
                     or packet["tool_calls_succeeded"] < 1
                     or type(packet.get("tool_calls_triggered")) is not int
                     or packet["tool_calls_triggered"] < packet["tool_calls_succeeded"]))):
        raise PilotBlocked("Invalid HIGH provenance, tool evidence or complete answer")
    return packet["text"]


def observation(prompt: str, text: str, complete: bool, http_error: bool,
                seconds: float, tool_used: str | None) -> dict:
    """Only raw model text is held in the private caller-owned record."""
    import math
    if (type(prompt) is not str or not prompt or len(prompt.encode("utf8")) > 4096
            or type(text) is not str or len(text.encode("utf8")) > 32768
            or type(complete) is not bool or type(http_error) is not bool
            or complete == http_error or (not complete and text)
            or type(seconds) not in (int, float) or not math.isfinite(seconds)
            or not 0 <= seconds <= 900 or tool_used not in TOOLS):
        raise PilotBlocked("Invalid local app observation")
    return {
        "text": text, "complete": complete, "http_error": http_error,
        "seconds": round(seconds, 3), "tool_used": tool_used,
        "model": MODEL, "model_digest": ORIGINAL_DIGESTS[MODEL],
        "profile": PROFILE, "prompt_sha256": hashlib.sha256(prompt.encode("utf8")).hexdigest()
    }


def _high_client(backend):
    from neral_preview_server import create_preview_app
    app = create_preview_app(backend)
    client = app.test_client()
    if client.get("/", base_url=BASE_URL).status_code != 200:
        raise PilotBlocked("The actual local preview app is not operational")
    session = client.get("/api/neral/preview/session", base_url=BASE_URL)
    packet = session.get_json(silent=True)
    if (session.status_code != 200 or type(packet) is not dict
            or type(packet.get("csrf")) is not str or len(packet["csrf"]) < 24):
        raise PilotBlocked("Preview session or CSRF token unavailable")
    return client, {"X-Neral-CSRF": packet["csrf"]}


def _run_arm(arm: str, *, question: str, exact_question: str,
             backend, client, headers: dict) -> tuple[str, bool, bool, float, str | None]:
    started = time.monotonic()
    if arm == "neral_high_router":
        # C is EXPLICIT, experimental and deliberately separate from the
        # shipped application. Never claim this already runs in release 1.0.
        hit = route_extra_arithmetic(exact_question)
        if hit is not None:
            return hit["text"], True, False, time.monotonic() - started, hit["tool_used"]
    if arm == "raw_qwen35":
        from neral_identity import build_identity_prompt
        from neral_agent_autonomy import AgentModelError
        # Same system identity and user prompt as the app HIGH route.
        turns = [
            {"role": "system", "content": build_identity_prompt(
                "web", "local_foundation_not_neral_adapter")},
            {"role": "user", "content": question},
        ]
        try:
            reply = backend.respond_chat(turns)
        except AgentModelError:
            return "", False, True, time.monotonic() - started, None
        if type(reply) is not str or not reply.strip() or len(reply.encode("utf8")) > 32768:
            raise PilotBlocked("Unusable raw local model reply")
        return reply, True, False, time.monotonic() - started, None
    if arm not in ("neral_high", "neral_high_router"):
        raise PilotBlocked("Unrecognized benchmark arm")
    response = client.post("/api/neral/chat/think", json={"prompt": question},
                           base_url=BASE_URL, headers=headers)
    elapsed = time.monotonic() - started
    if response.status_code == 503:
        return "", False, True, elapsed, None
    if response.status_code != 200:
        raise PilotBlocked("Preview HIGH returned an unexpected HTTP status")
    packet = response.get_json(silent=True)
    checked_high_packet(packet)
    return packet["text"], True, False, elapsed, packet.get("tool_used")


def run_live(*, output_dir: Path, release_receipt: dict, authorization: str) -> dict:
    """One real sequential 36-case GPU diagnostic; private outputs outside repo.

    No claims about new weights, release validity, autonomous routing or the
    independent 120-case holdout follow from this development pilot.
    """
    if authorization != AUTHORIZATION:
        raise PilotBlocked("Missing explicit manual GPU diagnostic authorization")
    release_gate(release_receipt)
    root = Path(__file__).resolve().parents[1]
    output_dir = Path(output_dir)
    if (output_dir.exists() or output_dir.is_symlink()
            or not output_dir.parent.is_dir() or output_dir.parent.is_symlink()
            or output_dir.resolve().is_relative_to(root)):
        raise PilotBlocked("Private output directory must be new and outside repository")
    from scripts.neral_v7_foundation_tournament import OllamaLocal
    require_model(OllamaLocal(timeout=120).inventory())
    from scripts.neral_foundation_2048_retest import SampledVram, _read_gpu_memory
    used, total = _read_gpu_memory()
    if total - used < 7000:
        raise PilotBlocked("GPU not free: refuse overlapping inference")
    from scripts.neral_low_high_quality import make_prompt
    from neral_agent_autonomy import OllamaLocalBackend
    cases = generate_pilot_cases(secrets.randbits(64))
    if len(cases) != 36:
        raise PilotBlocked("Incomplete local 36-case development suite")
    backend = OllamaLocalBackend(MODEL, timeout=90, inference_profile=PROFILE)
    client, headers = _high_client(backend)
    rows = []
    output_dir.mkdir(exist_ok=False)
    try:
        with SampledVram() as sampler:
            for i, case in enumerate(cases):
                q = make_prompt(case)
                entries = {}
                # Rotate order; compare each arm on the same exact prompt.
                order = list(ARMS)
                random.Random(int.from_bytes(
                    hashlib.sha256(case["id"].encode("ascii")).digest()[:8], "big")
                    ).shuffle(order)
                for arm in order:
                    text, finished, failed, seconds, tool = _run_arm(
                        arm, question=q, exact_question=case["prompt"],
                        backend=backend, client=client, headers=headers)
                    entries[arm] = observation(q, text, finished, failed, seconds, tool)
                rows.append({"id": case["id"], "arms": entries})
                if (i + 1) % 6 == 0:
                    print(f"SAME_BASE_PRIVATE_PROGRESS {i+1}/36", flush=True)
            vram = sampler.report()
    finally:
        backend.unload()
    if require_model(OllamaLocal(timeout=120).inventory()) != ORIGINAL_DIGESTS[MODEL]:
        raise PilotBlocked("Model digest changed during same-base comparison")
    private = {
        "manifest": {"model": MODEL, "model_digest": ORIGINAL_DIGESTS[MODEL],
                     "profile": PROFILE, "token_budget": 2048},
        "cases": [{"id": c["id"], "domain": c["domain"], "family": c["family"],
                   "prompt": make_prompt(c), "answer": c["answer"]} for c in cases],
        "observations": rows,
    }
    (output_dir / "private-paired-answers.json").write_text(
        json.dumps(private, ensure_ascii=False), encoding="utf8")
    try:
        report = score_private_pairs(private)
    except SameBaseError as exc:
        raise PilotBlocked("Private paired scorer rejected this incomplete pilot") from exc
    report["gpu_telemetry"] = vram
    report["collector_status"] = "REAL_LOCAL_GPU_THREE_ARM_DEVELOPMENT_ONLY"
    with (output_dir / "scalar-summary.json").open("x", encoding="utf8") as fp:
        json.dump(report, fp, indent=2, sort_keys=True)
        fp.write("\n")
    print(json.dumps({
        "status": report["collector_status"],
        "correct": {arm: report["arms"][arm]["correct"] for arm in ARMS},
        "model_digest": report["model_digest"],
        "router_tool_calls": report["arms"]["neral_high_router"]["tool_calls"],
        "release_validated": False, "independent_holdout_verified": False,
    }, sort_keys=True), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--private-output-dir", required=True, type=Path)
    parser.add_argument("--release-receipt-json", required=True, type=Path)
    parser.add_argument("--authorize", required=True)
    args = parser.parse_args()
    # Provided receipt must have been independently checked against GitHub.
    # This program itself does not have network access and cannot authenticate.
    receipt = json.loads(args.release_receipt_json.read_text(encoding="utf8"))
    run_live(output_dir=args.private_output_dir, release_receipt=receipt,
             authorization=args.authorize)


if __name__ == "__main__":
    main()
