"""Read-only matched_fewshot comparison on the visible measurement slices.

Compares base Qwen, the pinned v5 adapter, and the pinned v6 adapter on
retention and transfer. The locked holdout is not imported. A CPU dry-run
uses a deterministic stub and does not load weights. The GPU path is a
separate mode, read-only, and refuses every authorization string except
its own.

A green dry-run is not a capability gain. ``training_authorized`` and
``claimed_score_gain`` stay false.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import random
import subprocess

from scripts.neral_release_gate import exact_mcnemar_p
from scripts.neral_v6_fresh_dev_eval import (
    BASE_SHA, V5_SHA, V6_SHA, VARIANTS, experiment_decision,
    messages_for_protocol, report_exposes_locked_holdout,
)
from scripts.neral_v6_measurement_slices import (
    grade_exact, partition_visible, retention_cases, slice_sha256,
    transfer_cases,
)

ARMS = VARIANTS
PROTOCOL = "matched_fewshot"
DECODING = {"do_sample": False, "greedy": True, "max_new_tokens": 160}
RETENTION_SHA = "333bfcb4e92a9b9d6d27c2d9a6e6181c13f9c629a3a8d2bb35885a104a319054"
TRANSFER_SHA = "9604fe1910698e572d408e58e478f7446f393409d0997db3ab2bd99f769b48a2"
AUTHORIZE = "EVALUATE_NERAL_V6_RETENTION_TRANSFER_MATCHED_FEWSHOT_READONLY"
BOOTSTRAP_SEED = 20261108
BOOTSTRAP_DRAWS = 2000
PAIRS = (
    ("neral_v5", "neral_v6"),
    ("qwen_fewshot", "neral_v6"),
    ("qwen_fewshot", "neral_v5"),
)
OUTCOMES = ("strict_correct", "wrong_readable_number", "unreadable", "incomplete")
_RAW_FIELDS = {"text", "complete", "new_tokens"}


def decoding_sha256() -> str:
    payload = json.dumps(DECODING, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf8")).hexdigest()


def _percentile(sorted_numbers: list[float], q: float) -> float:
    if not sorted_numbers or not 0 <= q <= 1:
        raise ValueError("Invalid percentile")
    position = q * (len(sorted_numbers) - 1)
    low, high = math.floor(position), math.ceil(position)
    weight = position - low
    return sorted_numbers[low] * (1 - weight) + sorted_numbers[high] * weight


def minimum_one_sided_discordances() -> int:
    """Smallest k with exact two-sided McNemar p <= 0.05 when the other side is 0."""
    for count in range(1, 65):
        if exact_mcnemar_p(count, 0) <= 0.05:
            return count
    raise RuntimeError("McNemar threshold drifted")


def load_visible_slices() -> tuple[list[dict], list[dict]]:
    """Pin the visible slices and refuse anything that is not one of them."""
    retention = retention_cases()
    transfer = transfer_cases()
    if slice_sha256(retention) != RETENTION_SHA:
        raise ValueError("Retention SHA-256 pin mismatch")
    if slice_sha256(transfer) != TRANSFER_SHA:
        raise ValueError("Transfer SHA-256 pin mismatch")
    if len(retention) != 164 or len(transfer) != 64:
        raise ValueError("Visible slice size pin mismatch")
    refuse_nonvisible(retention + transfer)
    return retention, transfer


def refuse_nonvisible(cases: list[dict]) -> None:
    for case in cases:
        role = case.get("role", "")
        ident = str(case.get("id", ""))
        if role not in ("retention", "transfer"):
            raise ValueError("Refusing a slice that is not retention or transfer")
        if "locked" in role or "holdout" in role or ident.startswith("locked"):
            raise ValueError("Refusing locked holdout material")
        if case.get("required_prompt_protocol") != PROTOCOL:
            raise ValueError("Slice protocol pin mismatch")
        if case.get("decoding") != "greedy" or case.get("max_new_tokens") != 160:
            raise ValueError("Slice decoding pin mismatch")


def shared_messages(case: dict) -> list[dict]:
    built = [messages_for_protocol(case, arm, PROTOCOL) for arm in ARMS]
    if any(item != built[0] for item in built):
        raise RuntimeError("Matched prompts differ across arms")
    historical = messages_for_protocol(case, "neral_v5", "historical")
    if historical == built[0]:
        raise RuntimeError("Matched few-shot collapsed to the historical adapter prompt")
    return built[0]


def arm_request(case: dict, arm: str) -> tuple[list[dict], dict]:
    if arm not in ARMS:
        raise ValueError("Unknown comparison arm")
    return shared_messages(case), DECODING


def assert_shared_protocol(cases: list[dict]) -> None:
    if not cases:
        raise ValueError("No visible cases")
    for case in cases:
        for arm in ARMS:
            messages, decoding = arm_request(case, arm)
            if decoding is not DECODING:
                raise RuntimeError("Decoding config is not the shared object")
            if decoding["do_sample"] or not decoding["greedy"] or decoding["max_new_tokens"] != 160:
                raise RuntimeError("Decoding config drifted")
            if messages != shared_messages(case):
                raise RuntimeError("Arm request rebuilt a different prompt")


def prompt_sha256(messages: list[dict]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return sha256(payload.encode("utf8")).hexdigest()


def finalize_record(case: dict, arm: str, messages: list[dict], decoding: dict,
                    raw: dict) -> dict:
    if decoding is not DECODING:
        raise RuntimeError("Completion used a different decoding object")
    if not isinstance(raw, dict) or set(raw) != _RAW_FIELDS:
        raise ValueError("Completion record fields drifted")
    text, complete, new_tokens = raw["text"], raw["complete"], raw["new_tokens"]
    if (type(complete) is not bool or type(new_tokens) is not int
            or not isinstance(text, str) or len(text) > 4096
            or not 0 <= new_tokens <= DECODING["max_new_tokens"]
            or any(ord(ch) < 32 and ch not in "\r\n\t" for ch in text)):
        raise ValueError("Completion is unsafe or outside the shared token budget")
    return {
        "id": case["id"],
        "role": case["role"],
        "family": case["family"],
        "arm": arm,
        "text": text,
        "complete": complete,
        "new_tokens": new_tokens,
        "outcome": grade_exact(case["answer"], text, complete),
        "prompt_sha256": prompt_sha256(messages),
        "decoding_sha256": decoding_sha256(),
    }


def stub_complete(case: dict, arm: str, messages: list[dict], decoding: dict) -> dict:
    """Deterministic stand-in. It is not a model and it does not load weights."""
    if decoding is not DECODING or messages != shared_messages(case):
        raise RuntimeError("Stub refused a drifted prompt or decoding config")
    if case["answer"] == "999999":
        raise RuntimeError("Stub wrong-answer token collides with an oracle answer")
    index = case["index"]
    if arm == "qwen_fewshot" and index == 0:
        return {"text": "", "complete": False, "new_tokens": 0}
    if arm == "qwen_fewshot" and index == 1:
        return {"text": "FINAL_ANSWER: 999999", "complete": True, "new_tokens": 4}
    if arm == "neral_v5" and index == 2:
        return {"text": "no numeric token", "complete": True, "new_tokens": 3}
    correct = {
        "qwen_fewshot": index % 2 == 0,
        "neral_v5": index % 3 != 0,
        "neral_v6": index % 4 != 1,
    }[arm]
    if correct:
        return {"text": case["answer"], "complete": True, "new_tokens": 8}
    return {"text": "FINAL_ANSWER: 999999", "complete": True, "new_tokens": 4}


def _item_bootstrap(reference: list[bool], candidate: list[bool]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    width = len(reference)
    diffs = []
    for _ in range(BOOTSTRAP_DRAWS):
        total = 0
        for _item in range(width):
            index = rng.randrange(width)
            total += int(candidate[index]) - int(reference[index])
        diffs.append(total / width)
    diffs.sort()
    return _percentile(diffs, 0.025), _percentile(diffs, 0.975)


def _family_bootstrap(reference: list[bool], candidate: list[bool],
                      families: list[str]) -> tuple[float, float]:
    grouped: dict[str, list[int]] = {}
    for left, right, family in zip(reference, candidate, families):
        grouped.setdefault(family, []).append(int(right) - int(left))
    names = sorted(grouped)
    rng = random.Random(BOOTSTRAP_SEED + 1)
    diffs = []
    for _ in range(BOOTSTRAP_DRAWS):
        total = count = 0
        for _draw in range(len(names)):
            picked = grouped[names[rng.randrange(len(names))]]
            total += sum(picked)
            count += len(picked)
        diffs.append(total / count)
    diffs.sort()
    return _percentile(diffs, 0.025), _percentile(diffs, 0.975)


def _ci_pair(bounds: tuple[float, float]) -> list[float]:
    return [round(bounds[0], 6), round(bounds[1], 6)]


def paired_stats(reference: list[bool], candidate: list[bool],
                 families: list[str]) -> dict:
    if not reference or not (len(reference) == len(candidate) == len(families)):
        raise ValueError("Paired flags are misaligned")
    if any(type(flag) is not bool for flag in reference + candidate):
        raise ValueError("Paired flags must be strict-correct booleans")
    improvements = sum(not left and right for left, right in zip(reference, candidate))
    regressions = sum(left and not right for left, right in zip(reference, candidate))
    point = (sum(candidate) - sum(reference)) / len(reference)
    item_ci = _item_bootstrap(reference, candidate)
    family_ci = _family_bootstrap(reference, candidate, families)
    return {
        "items": len(reference),
        "reference_strict_correct": sum(reference),
        "candidate_strict_correct": sum(candidate),
        "improvements": improvements,
        "regressions": regressions,
        "both_correct": sum(left and right for left, right in zip(reference, candidate)),
        "both_incorrect": sum(not left and not right
                              for left, right in zip(reference, candidate)),
        "gap": round(point, 6),
        "gap_pp": round(100 * point, 3),
        "exact_mcnemar_p": exact_mcnemar_p(improvements, regressions),
        "item_bootstrap_95_ci": _ci_pair(item_ci),
        "item_bootstrap_95_ci_pp": _ci_pair((100 * item_ci[0], 100 * item_ci[1])),
        "family_cluster_bootstrap_95_ci": _ci_pair(family_ci),
        "family_cluster_bootstrap_95_ci_pp": _ci_pair((100 * family_ci[0], 100 * family_ci[1])),
        "family_cluster_count": len(set(families)),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "item_bootstrap_assumes_independent_items": True,
        "family_cluster_bootstrap_resamples_templates": True,
    }


def _flags(cases: list[dict], records: list[dict], arm: str) -> list[bool]:
    by_id = {row["id"]: row for row in records if row["arm"] == arm}
    if any(case["id"] not in by_id for case in cases):
        raise ValueError("Arm records do not match the pinned cases")
    return [by_id[case["id"]]["outcome"] == "strict_correct" for case in cases]


def _slice_block(cases: list[dict], records: list[dict]) -> dict:
    by_arm = {}
    for arm in ARMS:
        arm_rows = [row for row in records if row["arm"] == arm]
        ordered = []
        by_id = {row["id"]: row for row in arm_rows}
        for case in cases:
            ordered.append({"text": by_id[case["id"]]["text"],
                            "complete": by_id[case["id"]]["complete"]})
        by_arm[arm] = partition_visible(cases, ordered)
    paired = {}
    for arm_a, arm_b in PAIRS:
        stats = paired_stats(
            _flags(cases, records, arm_a),
            _flags(cases, records, arm_b),
            [case["family"] for case in cases],
        )
        stats["arm_a"] = arm_a
        stats["arm_b"] = arm_b
        paired[f"{arm_b}_minus_{arm_a}"] = stats
    return {"by_arm": by_arm, "paired": paired}


def _scores(cases: list[dict], records: list[dict]) -> dict:
    scores = {}
    for arm in ARMS:
        counts = {name: 0 for name in OUTCOMES}
        by_id = {row["id"]: row for row in records if row["arm"] == arm}
        for case in cases:
            counts[by_id[case["id"]]["outcome"]] += 1
        if sum(counts.values()) != len(cases):
            raise ValueError("Pooled partition does not cover the visible slices")
        scores[arm] = {
            "strict_correct": counts["strict_correct"],
            "wrong_readable_number": counts["wrong_readable_number"],
            "unreadable": counts["unreadable"],
            "incomplete": counts["incomplete"],
            "correct": counts["strict_correct"],
            "ungraded_completed": counts["unreadable"],
            "total": len(cases),
        }
    return scores


def sample_size_note(transfer_count: int, transfer_families: int) -> dict:
    needed = minimum_one_sided_discordances()
    return {
        "transfer_items": transfer_count,
        "transfer_families": transfer_families,
        "minimum_one_sided_discordances_for_two_sided_p_le_0_05": needed,
        "transfer_minimum_one_sided_gap_pp": round(100 * needed / transfer_count, 3),
        "five_versus_zero_exact_p": exact_mcnemar_p(5, 0),
        "six_versus_zero_exact_p": exact_mcnemar_p(6, 0),
        "interval_covering_zero_is_not_evidence_of_equality": True,
        "approximate_power": (
            "An item-level paired difference near 15 percentage points still "
            "needs on the order of 85 independent items at 80 percent power "
            "when about a quarter of pairs disagree. Family templates are not "
            "independent items."
        ),
    }


def build_summary(cases_by_role: dict[str, list[dict]], records: list[dict],
                  backend: str) -> dict:
    retention = cases_by_role["retention"]
    transfer = cases_by_role["transfer"]
    visible = retention + transfer
    if backend not in {"deterministic_stub", "local_4bit_readonly"}:
        raise ValueError("Unknown comparison backend")
    gpu = backend == "local_4bit_readonly"
    scores = _scores(visible, records)
    summary = {
        "schema": 1,
        "status": "NERAL_V6_SLICE_COMPARE_STUB_DRY_RUN" if not gpu
        else "NERAL_V6_SLICE_COMPARE_READONLY",
        "prompt_protocol": PROTOCOL,
        "decoding": dict(DECODING),
        "decoding_sha256": decoding_sha256(),
        "arms": list(ARMS),
        "arms_share_one_decoding_object": True,
        "retention_sha256": RETENTION_SHA,
        "transfer_sha256": TRANSFER_SHA,
        "retention_count": len(retention),
        "transfer_count": len(transfer),
        "model_backend": backend,
        "gpu_inference_performed": gpu,
        "weights_loaded": gpu,
        "weights_checked": gpu,
        "read_only": True,
        "training_performed": False,
        "training_authorized": False,
        "claimed_score_gain": False,
        "claimed_capability_gain": False,
        "release_validated": False,
        "adapter_promoted": False,
        "weights_promoted": False,
        "general_improvement_proven": False,
        "sealed_slice_loaded": False,
        "decision_ok_means_protocol_admissible_not_training": True,
        "expected_base_revision": BASE_SHA,
        "expected_v5_adapter_sha256": V5_SHA,
        "expected_v6_adapter_sha256": V6_SHA,
        "total": len(visible),
        "scores": scores,
        "slices": {
            "retention": _slice_block(retention, records),
            "transfer": _slice_block(transfer, records),
        },
        "sample_size": sample_size_note(len(transfer), len({case["family"] for case in transfer})),
        "items": [{
            "id": row["id"], "role": row["role"], "family": row["family"],
            "arm": row["arm"], "outcome": row["outcome"],
            "complete": row["complete"], "new_tokens": row["new_tokens"],
        } for row in records],
    }
    prompt_groups: dict[str, set[str]] = {}
    decoding_seen = set()
    for row in records:
        prompt_groups.setdefault(row["id"], set()).add(row["prompt_sha256"])
        decoding_seen.add(row["decoding_sha256"])
    if any(len(group) != 1 for group in prompt_groups.values()):
        raise RuntimeError("Arms did not share a prompt hash on every item")
    if decoding_seen != {decoding_sha256()}:
        raise RuntimeError("Arms did not share the decoding hash")
    summary["arms_share_prompt_sha256"] = True
    summary["arms_share_decoding_sha256"] = True
    if report_exposes_locked_holdout(summary):
        raise RuntimeError("Refusing a report that carries locked-holdout material")
    gate_view = {
        "prompt_protocol": PROTOCOL,
        "release_validated": False,
        "adapter_promoted": False,
        "weights_promoted": False,
        "claimed_score_gain": False,
        "general_improvement_proven": False,
        "training_authorized": False,
        "total": len(visible),
        "scores": scores,
    }
    decision = experiment_decision(gate_view)
    if decision["training_authorized"] or decision["claimed_score_gain"]:
        raise RuntimeError("Comparison tried to authorize training or a score gain")
    if report_exposes_locked_holdout(decision):
        raise RuntimeError("Gate result carries locked-holdout material")
    summary["phase2_gate"] = decision
    summary["experiment_decision"] = decision
    return summary


def assert_new_output(dest: Path) -> None:
    if dest.exists() or dest.is_symlink():
        raise ValueError("Refusing to overwrite an existing output path")
    if dest.name in {"adapter", "candidate-adapter"}:
        raise ValueError("Refusing an adapter directory name")
    parent = dest.parent
    if not parent.is_dir() or parent.is_symlink():
        raise ValueError("Output parent must be a real directory")
    folder = parent
    while True:
        if folder.is_symlink():
            raise ValueError("Refusing a symlinked output parent")
        weights = folder / "adapter_model.safetensors"
        if weights.is_file():
            raise ValueError("Refusing to write inside an adapter directory")
        if folder.parent == folder:
            break
        folder = folder.parent


def create_fresh(parent: Path, name: str) -> Path:
    dest = parent / name
    assert_new_output(dest)
    dest.mkdir()
    return dest


def _write_new(path: Path, text: str) -> None:
    with path.open("x", encoding="utf8") as handle:
        handle.write(text)


def _write_json(path: Path, payload: dict) -> None:
    _write_new(path, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    body = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    if len(body.encode("utf8")) > 2_000_000:
        raise ValueError("Refusing an oversized private record file")
    _write_new(path, body)


def _read_jsonl(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 2_000_000:
        raise ValueError("Missing or oversized record file")
    return [json.loads(line) for line in path.read_text("utf8").splitlines() if line]


def _backend_meta(arm: str, backend: str, count: int) -> dict:
    gpu = backend == "local_4bit_readonly"
    return {
        "schema": 1,
        "arm": arm,
        "backend": backend,
        "cases": count,
        "gpu_inference_performed": gpu,
        "weights_loaded": gpu,
        "weights_checked": gpu,
        "read_only": True,
        "training_performed": False,
        "prompt_protocol": PROTOCOL,
        "decoding_sha256": decoding_sha256(),
    }


def _generate_arm(dest: Path, cases: list[dict], arm: str, complete, backend: str) -> None:
    out = dest / f"{arm}.jsonl"
    meta_path = dest / f"{arm}.meta.json"
    if out.exists() or meta_path.exists():
        raise ValueError("Refusing to overwrite an arm record")
    rows = []
    for case in cases:
        messages, decoding = arm_request(case, arm)
        raw = complete(case, arm, messages, decoding)
        rows.append(finalize_record(case, arm, messages, decoding, raw))
    _write_jsonl(out, rows)
    _write_json(meta_path, _backend_meta(arm, backend, len(rows)))


def _manifest(retention: list[dict], transfer: list[dict]) -> dict:
    return {
        "schema": 1,
        "prompt_protocol": PROTOCOL,
        "decoding_sha256": decoding_sha256(),
        "retention_sha256": RETENTION_SHA,
        "transfer_sha256": TRANSFER_SHA,
        "retention_count": len(retention),
        "transfer_count": len(transfer),
        "sealed_slice_loaded": False,
        "training_authorized": False,
        "claimed_score_gain": False,
    }


def prepare_dir(dest: Path) -> dict:
    retention, transfer = load_visible_slices()
    assert_shared_protocol(retention + transfer)
    assert_new_output(dest)
    dest.mkdir()
    _write_json(dest / "manifest.json", _manifest(retention, transfer))
    return {"run_dir": str(dest), "retention_count": len(retention),
            "transfer_count": len(transfer), "gpu_inference_performed": False}


def _require_prepared(dest: Path) -> tuple[list[dict], list[dict]]:
    if not dest.is_dir() or dest.is_symlink():
        raise ValueError("Prepared comparison directory is missing")
    manifest_path = dest / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError("Comparison manifest is missing")
    manifest = json.loads(manifest_path.read_text("utf8"))
    retention, transfer = load_visible_slices()
    if (manifest.get("retention_sha256") != RETENTION_SHA
            or manifest.get("transfer_sha256") != TRANSFER_SHA
            or manifest.get("prompt_protocol") != PROTOCOL
            or manifest.get("decoding_sha256") != decoding_sha256()
            or manifest.get("sealed_slice_loaded") is not False):
        raise ValueError("Prepared directory does not match the pinned slices")
    assert_shared_protocol(retention + transfer)
    return retention, transfer


def execute(dest: Path, complete, backend: str) -> dict:
    retention, transfer = _require_prepared(dest)
    cases = retention + transfer
    for arm in ARMS:
        _generate_arm(dest, cases, arm, complete, backend)
    return combine(dest)


def combine(dest: Path) -> dict:
    retention, transfer = _require_prepared(dest)
    cases = retention + transfer
    metas = []
    records = []
    for arm in ARMS:
        meta = json.loads((dest / f"{arm}.meta.json").read_text("utf8"))
        rows = _read_jsonl(dest / f"{arm}.jsonl")
        if (meta.get("arm") != arm or meta.get("cases") != len(cases)
                or meta.get("decoding_sha256") != decoding_sha256()
                or meta.get("prompt_protocol") != PROTOCOL
                or meta.get("read_only") is not True
                or meta.get("training_performed") is not False
                or len(rows) != len(cases)):
            raise ValueError("Arm metadata does not match the pinned run")
        metas.append(meta)
        rebuilt = []
        for case, row in zip(cases, rows):
            if row.get("id") != case["id"] or row.get("arm") != arm:
                raise ValueError("Arm records are reordered or mislabeled")
            messages, decoding = arm_request(case, arm)
            checked = finalize_record(case, arm, messages, decoding, {
                "text": row["text"], "complete": row["complete"],
                "new_tokens": row["new_tokens"],
            })
            if (checked["outcome"] != row["outcome"]
                    or checked["prompt_sha256"] != row["prompt_sha256"]
                    or checked["decoding_sha256"] != row["decoding_sha256"]):
                raise ValueError("Stored record does not match a fresh grade")
            rebuilt.append(checked)
        records.extend(rebuilt)
    backends = {meta["backend"] for meta in metas}
    if len(backends) != 1:
        raise ValueError("Arms were not produced by one backend")
    backend = backends.pop()
    if any(meta["gpu_inference_performed"] != (backend == "local_4bit_readonly")
           for meta in metas):
        raise ValueError("GPU flag disagrees with the backend")
    summary = build_summary(
        {"retention": retention, "transfer": transfer}, records, backend)
    summary_path = dest / "summary.json"
    if summary_path.exists():
        raise ValueError("Refusing to overwrite a comparison report")
    _write_jsonl(dest / "items.jsonl", records)
    _write_json(summary_path, summary)
    return summary


def dry_run(out_root: Path) -> dict:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dest = out_root / f"slice-compare-{stamp}"
    prepare_dir(dest)
    return execute(dest, stub_complete, "deterministic_stub")


def _require_gpu_authorization(authorize: str) -> None:
    if authorize != AUTHORIZE:
        raise ValueError("Refusing GPU comparison without the exact read-only permit")


def _check_readonly_models(base: Path, v5: Path, v6: Path) -> None:
    from neral_hf_identity import validate_adapter, validate_hf_snapshot
    if any(path.is_symlink() or not path.is_dir() for path in (base, v5, v6)):
        raise ValueError("Base or adapter path is missing")
    if validate_hf_snapshot(base, verify_hashes=True)["revision"] != BASE_SHA:
        raise ValueError("Unexpected foundation model revision")
    v5_manifest = validate_adapter(v5, validate_hf_snapshot(base))
    if v5_manifest["files"]["adapter_model.safetensors"] != V5_SHA:
        raise ValueError("Untrusted v5 adapter")
    v6_file = v6 / "adapter_model.safetensors"
    v6_config = v6 / "adapter_config.json"
    if (v6_file.is_symlink() or v6_config.is_symlink() or not v6_file.is_file()
            or not v6_config.is_file()
            or sha256(v6_file.read_bytes()).hexdigest() != V6_SHA):
        raise ValueError("Untrusted v6 adapter")
    config = json.loads(v6_config.read_text("utf8"))
    if (config.get("r") != 16 or config.get("peft_type") != "LORA"
            or config.get("task_type") != "CAUSAL_LM"
            or set(config.get("target_modules", [])) != {
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            }):
        raise ValueError("v6 LoRA configuration changed")


def _gpu_ready() -> None:
    probe = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=12, check=True)
    values = [int(part.strip()) for part in probe.stdout.splitlines()[0].split(",")]
    if len(values) != 2 or values[0] < 9000 or values[1] > 20:
        raise ValueError("GPU occupied")


def _gpu_complete_factory(arm: str, base: Path, v5: Path, v6: Path):
    """Load one read-only arm. Tests never call this."""
    _check_readonly_models(base, v5, v6)
    _gpu_ready()
    import os
    os.environ.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1",
                      TRANSFORMERS_OFFLINE="1", WANDB_DISABLED="true")
    import torch
    from peft import PeftModel
    from transformers import Qwen2TokenizerFast
    from neral_hf_identity import CHAT_TEMPLATE
    from scripts.neral_math_eval import _load_model
    from scripts.neral_v3_eval_isolation import clean_cuda, prepare_cuda

    tokenizer = Qwen2TokenizerFast.from_pretrained(
        str(base), local_files_only=True, trust_remote_code=False, token=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prepare_cuda(torch)
    model = _load_model(base)
    folder = v5 if arm == "neral_v5" else v6 if arm == "neral_v6" else None
    if folder is not None:
        model = PeftModel.from_pretrained(
            model, str(folder), is_trainable=False, local_files_only=True)
    model.eval()

    def complete(case, current_arm, messages, decoding):
        if current_arm != arm or decoding is not DECODING:
            raise RuntimeError("GPU arm or decoding drifted")
        if decoding["do_sample"] or decoding["max_new_tokens"] != 160:
            raise RuntimeError("GPU decoding config drifted")
        inputs = tokenizer.apply_chat_template(
            messages, chat_template=CHAT_TEMPLATE, tokenize=True,
            add_generation_prompt=True, return_tensors="pt").to("cuda:0")
        if inputs.shape[-1] > 2048:
            raise ValueError("Visible case exceeded the context cap")
        with torch.inference_mode():
            generated = model.generate(
                inputs, attention_mask=torch.ones_like(inputs),
                do_sample=decoding["do_sample"],
                max_new_tokens=decoding["max_new_tokens"],
                pad_token_id=tokenizer.pad_token_id)
        completion = generated[0, inputs.shape[-1]:]
        text = tokenizer.decode(completion, skip_special_tokens=True)
        ended = bool(len(completion) > 0 and completion[-1].item() == tokenizer.eos_token_id)
        return {"text": text[:4096], "complete": ended and len(text) <= 4096,
                "new_tokens": len(completion)}

    def close():
        nonlocal model
        del model
        clean_cuda(torch)

    return complete, close


def run_gpu_arm(dest: Path, arm: str, base: Path, v5: Path, v6: Path,
                authorize: str) -> dict:
    _require_gpu_authorization(authorize)
    if arm not in ARMS:
        raise ValueError("Unknown comparison arm")
    retention, transfer = _require_prepared(dest)
    complete, close = _gpu_complete_factory(arm, base, v5, v6)
    try:
        _generate_arm(dest, retention + transfer, arm, complete, "local_4bit_readonly")
    finally:
        close()
    return {"arm": arm, "gpu_inference_performed": True, "read_only": True,
            "training_performed": False}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--mode", choices=("dry-run", "prepare", "run", "combine"),
                        required=True)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--base", type=Path)
    parser.add_argument("--v5-adapter", type=Path)
    parser.add_argument("--v6-adapter", type=Path)
    parser.add_argument("--authorize", default="")
    args = parser.parse_args(argv)
    if args.mode == "dry-run":
        if args.out_root is None:
            parser.error("dry-run requires a parent directory that already exists")
        summary = dry_run(args.out_root)
        print(json.dumps({
            "status": summary["status"],
            "model_backend": summary["model_backend"],
            "gpu_inference_performed": summary["gpu_inference_performed"],
            "training_authorized": summary["training_authorized"],
            "claimed_score_gain": summary["claimed_score_gain"],
            "phase2_gate": summary["phase2_gate"]["decision"],
        }, sort_keys=True))
        return
    if args.run_dir is None:
        parser.error("prepare, run, and combine require --run-dir")
    if args.mode == "prepare":
        print(json.dumps(prepare_dir(args.run_dir), sort_keys=True))
        return
    if args.mode == "combine":
        summary = combine(args.run_dir)
        print(json.dumps({
            "status": summary["status"],
            "gpu_inference_performed": summary["gpu_inference_performed"],
            "training_authorized": summary["training_authorized"],
            "claimed_score_gain": summary["claimed_score_gain"],
            "phase2_gate": summary["phase2_gate"]["decision"],
        }, sort_keys=True))
        return
    if not (args.arm and args.base and args.v5_adapter and args.v6_adapter):
        parser.error("GPU run requires an arm and the three read-only model paths")
    print(json.dumps(run_gpu_arm(
        args.run_dir, args.arm, args.base, args.v5_adapter, args.v6_adapter,
        args.authorize), sort_keys=True))


if __name__ == "__main__":
    main()
