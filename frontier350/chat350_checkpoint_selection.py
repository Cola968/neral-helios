"""Read-only identity check and evaluation of the actually selected 350M SFT model.

Pure Python orchestration with dependency injection for checkpoint loader,
digest and evaluators. GPU work stays in train_chat350 and is never started
by importing this module.
"""
from __future__ import annotations

import math
import re
from pathlib import Path


def evaluate_selected_checkpoint(
    model,
    checkpoint_path: str | Path,
    *,
    loader,
    checkpoint_digest,
    expected_config: dict,
    expected_tokenizer_sha256: str,
    expected_source_sha256: str,
    expected_data_manifest_sha256: str,
    last_step: int,
    expected_best_step: int | None,
    eval_validation,
    eval_test,
    eval_replay,
) -> dict | None:
    """Evaluate best_chat.pt rather than the last-step model state.

    Caller computes last-step metrics BEFORE calling this function.
    A missing selected checkpoint means no eligible candidate; test set is
    deliberately not evaluated in that case.
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        if expected_best_step is not None:
            raise ValueError("selected checkpoint missing despite recorded best step")
        return None
    if type(expected_best_step) is not int or expected_best_step <= 0:
        raise ValueError("selected checkpoint exists without recorded best step")
    if type(last_step) is not int or not 0 < expected_best_step <= last_step:
        raise ValueError("selected checkpoint step outside completed run")

    payload = loader(path)
    if not isinstance(payload, dict):
        raise ValueError("selected checkpoint must be an object")
    identity = (
        ("config", expected_config),
        ("tokenizer_sha256", expected_tokenizer_sha256),
        ("source_checkpoint_sha256", expected_source_sha256),
        ("data_manifest_sha256", expected_data_manifest_sha256),
        ("chat_step", expected_best_step),
    )
    for key, expected in identity:
        if payload.get(key) != expected:
            raise ValueError("selected checkpoint identity mismatch: " + key)
    val_record = payload.get("validation")
    if not isinstance(val_record, dict):
        raise ValueError("selected checkpoint missing validation evidence")
    recorded_nll = val_record.get("answer_nll")
    if (type(recorded_nll) not in (int, float) or not math.isfinite(recorded_nll)
        or recorded_nll <= 0):
        raise ValueError("selected checkpoint missing finite validation NLL")
    weights = payload.get("model")
    if not isinstance(weights, dict) or not weights:
        raise ValueError("selected checkpoint must contain weights")

    model.load_state_dict(weights, strict=True)
    # All three metrics below come from the SAME explicitly reloaded weights.
    selected_val = eval_validation()
    selected_test = eval_test()
    selected_replay = eval_replay()
    digest = checkpoint_digest(path)
    if not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None:
        raise ValueError("invalid selected checkpoint digest")
    return {
        "selected_checkpoint_step": expected_best_step,
        "selected_checkpoint_sha256": digest,
        "selected_checkpoint_val_sft": selected_val,
        "selected_checkpoint_test_sft": selected_test,
        "selected_checkpoint_replay_loss": selected_replay,
        "selection_validation_nll": float(recorded_nll),
    }
