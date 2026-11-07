"""NERAL v6: experimental 120-step continuation from SEALED v5 rank16 adapter.

CPU data synthesis, deterministic proof checks and fail-closed token boundary
preflight precede every optional GPU step. No training/test leakage, cloud API,
base-weight changes, release claims, adapter overwrites or auto-deployment.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import random
import re
import subprocess
import time
import unicodedata

from scripts.neral_v6_compositional import (
    FAMILIES, PER_FAMILY, make_case, generate as compositional_data
)
from scripts.neral_reasoning_train_candidate import candidate, verify_case

AUTH = "TRAIN_NERAL_V6_R16_120_EXPERIMENTAL"
VERIFIED_STEPS_AUTH = "TRAIN_NERAL_V6_VERIFIED_STEPS_120_EXPERIMENTAL"
SEED = 20261107
STEPS = 120
MAX_LENGTH = 768
SOURCES = {"v6_composition": 600, "reasoning_counterfactual": 128}


def require_explicit_gpu_authorization(token: str) -> bool:
    if token != AUTH:
        raise ValueError("Only explicit local experimental GPU authorization permitted")
    return True


def _jsonl_bytes(rows):
    return b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True,
                                 separators=(",", ":")) + "\n").encode("utf8")
                    for row in rows)


def _compact_trace(proof: dict, raw: dict) -> str:
    """Only exact replay-verified intermediate nodes, never a fabricated CoT."""
    if not verify_case(proof, raw):
        raise ValueError("Independent Decimal arithmetic replay failed")
    nodes = proof["nodes"]
    if len(nodes) not in (3, 8):
        raise ValueError("Unexpected arithmetic graph")
    values = [f"{n['id']}={n['value']}" for n in nodes]
    unit = "Cent" if proof["family"] == "ordered_price" else "kg,kg/min"
    return ("CHECK: " + "; ".join(values) + f" ({unit}).\n"
            "FINAL_ANSWER: " + proof["answer"])


def _verified_step_target(proof: dict, raw: dict) -> str:
    """Use the assistant text that verify_case already locked.

    The compact form keeps the same final number but deletes every
    multiplication and addition the proof rendered. This target puts
    those checked operations back. It does not change the answer.
    """
    if not verify_case(proof, raw):
        raise ValueError("Verified step target failed independent replay")
    text = raw["messages"][2]["content"]
    if text.count("FINAL_ANSWER:") != 1 or "*" not in text:
        raise ValueError("Verified trace has no single answer or no multiplication")
    final = text.split("FINAL_ANSWER:", 1)[1].strip()
    if final != proof["answer"]:
        raise ValueError("Verified final answer drifted from the proof")
    compact = _compact_trace(proof, raw)
    if compact.split("FINAL_ANSWER:", 1)[1].strip() != final or text == compact:
        raise ValueError("Compact control does not match this verified answer")
    return text


_PINNED_CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}{% endif %}"
)
_QWEN_SPECIALS = ("<|im_start|>", "<|im_end|>")
_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")
_LENGTH_APPROXIMATION = (
    "No network tokenizer download. <|im_start|> and <|im_end|> count as one "
    "token each, the pinned Qwen added tokens. Every other UTF-8 byte counts "
    "as one token. Qwen2 byte-level BPE merges only shorten that count, so "
    "this does not under-estimate tokenized_completion; it can over-estimate. "
    "The pinned chat template is assumed to add no extra BOS."
)


def _is_letter(ch: str) -> bool:
    return unicodedata.category(ch).startswith("L")


def _is_number(ch: str) -> bool:
    return unicodedata.category(ch).startswith("N")


def _is_space(ch: str) -> bool:
    return ch in "\t\n\r\f\v" or unicodedata.category(ch) in ("Zs", "Zl", "Zp")


def _render_pinned_chat(messages, *, add_generation_prompt: bool) -> str:
    """Render the repo-pinned Qwen chat template without Jinja or a model."""
    from neral_hf_identity import CHAT_TEMPLATE
    if CHAT_TEMPLATE != _PINNED_CHAT_TEMPLATE:
        raise ValueError("Pinned Qwen chat template changed; refuse the length model")
    if not isinstance(messages, list) or not messages:
        raise ValueError("Chat messages required")
    parts = []
    for message in messages:
        if not isinstance(message, dict):
            raise ValueError("Chat message is not an object")
        role, content = message.get("role"), message.get("content")
        if not isinstance(role, str) or not isinstance(content, str):
            raise ValueError("Chat message is not text")
        parts.append("<|im_start|>" + role + "\n" + content + "<|im_end|>\n")
    if add_generation_prompt:
        parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def _qwen_match_at(text: str, index: int):
    """One isolated Qwen2 pretoken, or None. Same alternation order as the regex."""
    size = len(text)
    if index >= size:
        return None
    lowered = text[index:index + 3].lower()
    for contraction in _CONTRACTIONS:
        if lowered.startswith(contraction):
            return index + len(contraction)

    def letters_from(start):
        if start < size and _is_letter(text[start]):
            end = start + 1
            while end < size and _is_letter(text[end]):
                end += 1
            return end
        return None

    ch = text[index]
    if ch not in "\r\n" and not _is_letter(ch) and not _is_number(ch):
        with_prefix = letters_from(index + 1)
        if with_prefix is not None:
            return with_prefix
    letters = letters_from(index)
    if letters is not None:
        return letters
    if _is_number(ch):
        return index + 1
    punct = index + 1 if ch == " " else index
    if (punct < size and not _is_space(text[punct])
            and not _is_letter(text[punct]) and not _is_number(text[punct])):
        end = punct + 1
        while (end < size and not _is_space(text[end])
               and not _is_letter(text[end]) and not _is_number(text[end])):
            end += 1
        while end < size and text[end] in "\r\n":
            end += 1
        if end > punct:
            return end
    end = index
    while end < size and _is_space(text[end]) and text[end] not in "\r\n":
        end += 1
    if end < size and text[end] in "\r\n":
        while end < size and text[end] in "\r\n":
            end += 1
        return end
    if _is_space(ch):
        end = index + 1
        while end < size and _is_space(text[end]):
            end += 1
        return end
    return None


def _qwen_text_pieces(segment: str) -> list[str]:
    parts = []
    index = 0
    while index < len(segment):
        end = _qwen_match_at(segment, index)
        if end is None or end <= index:
            raise ValueError("Qwen pretokenizer left unmatched text; refuse length proof")
        parts.append(segment[index:end])
        index = end
    if "".join(parts) != segment:
        raise ValueError("Qwen pretokenizer partition is not exact")
    return parts


def _qwen_pieces(text: str) -> list[tuple[str, str]]:
    """Split atomic chat specials first, then the Qwen2 pretokenizer."""
    pieces = []
    index = 0
    size = len(text)
    while index < size:
        special = next((token for token in _QWEN_SPECIALS
                        if text.startswith(token, index)), None)
        if special is not None:
            pieces.append(("special", special))
            index += len(special)
            continue
        nxt = size
        for token in _QWEN_SPECIALS:
            found = text.find(token, index)
            if found != -1:
                nxt = min(nxt, found)
        for part in _qwen_text_pieces(text[index:nxt]):
            pieces.append(("text", part))
        index = nxt
    if "".join(part for _, part in pieces) != text:
        raise ValueError("Chat text was not partitioned")
    return pieces


def _piece_upper_bound(pieces: list[tuple[str, str]]) -> int:
    total = 0
    for kind, part in pieces:
        total += 1 if kind == "special" else len(part.encode("utf-8"))
    return total


def prove_qwen_completion_bound(row: dict) -> dict:
    """Same acceptance rules as tokenized_completion, as a byte upper bound.

    Prompt token ids are a prefix when the rendered prompt's pieces are a
    prefix of the full rendering: the Qwen pretokenizer is isolated, so BPE
    does not merge across those pieces. The assistant span is at least the
    number of suffix pieces, and each piece is at least one token.
    """
    messages = row.get("messages") if isinstance(row, dict) else None
    if (not isinstance(messages, list) or len(messages) != 3 or
            [m.get("role") for m in messages if isinstance(m, dict)] !=
            ["system", "user", "assistant"]):
        raise ValueError("Completion row is not system, user, assistant")
    prompt = _render_pinned_chat(messages[:2], add_generation_prompt=True)
    full = _render_pinned_chat(messages, add_generation_prompt=False)
    if not full.startswith(prompt):
        raise ValueError("Prompt is not a prefix of the rendered chat")
    prompt_pieces = _qwen_pieces(prompt)
    full_pieces = _qwen_pieces(full)
    if full_pieces[:len(prompt_pieces)] != prompt_pieces:
        raise ValueError("Qwen completion-mask boundary mismatch or truncation")
    suffix = full_pieces[len(prompt_pieces):]
    if len(suffix) < 2:
        raise ValueError("No assistant answer for loss")
    return {
        "upper_bound_tokens": _piece_upper_bound(full_pieces),
        "assistant_pieces": len(suffix),
    }


def _arithmetic_slashes(text: str) -> int:
    return text.replace(" kg/min", "").count("/")


def _assert_arithmetic_preserved(original: str, fitted: str) -> None:
    if "FINAL_ANSWER:" not in original or "FINAL_ANSWER:" not in fitted:
        raise ValueError("Verified assistant lost FINAL_ANSWER")
    if (original.split("FINAL_ANSWER:", 1)[1].strip()
            != fitted.split("FINAL_ANSWER:", 1)[1].strip()):
        raise ValueError("Boilerplate trim changed the final answer")
    if re.findall(r"\d+", original) != re.findall(r"\d+", fitted):
        raise ValueError("Boilerplate trim changed a verified number")
    for operator in ("*", "+", "-"):
        if original.count(operator) != fitted.count(operator):
            raise ValueError("Boilerplate trim dropped a verified operator")
    if _arithmetic_slashes(original) != _arithmetic_slashes(fitted):
        raise ValueError("Boilerplate trim dropped a verified division")


def _drop_rounding_sentence(text: str) -> str:
    return "\n".join(line for line in text.split("\n")
                     if not line.startswith("Erst jetzt"))


def _drop_intermediate_units(text: str) -> str:
    lines = []
    for line in _drop_rounding_sentence(text).split("\n"):
        if not line.startswith("FINAL_ANSWER:"):
            for suffix in (" kg/min", " kg", " Cent"):
                if line.endswith(suffix):
                    line = line[:-len(suffix)]
                    break
        lines.append(line)
    return "\n".join(lines)


def _drop_node_labels(text: str) -> str:
    lines = []
    for line in _drop_intermediate_units(text).split("\n"):
        if (": " in line and not line.startswith("FINAL_ANSWER:")
                and line != "CHECK:" and not line.startswith("Erst jetzt")):
            line = line.split(": ", 1)[1]
        lines.append(line)
    return "\n".join(lines)


def _drop_parentheses(text: str) -> str:
    return _drop_node_labels(text).replace("(", "").replace(")", "")


def _squeeze_operator_spaces(text: str) -> str:
    return re.sub(r"[ \t]*([*/+=-])[ \t]*", r"\1", _drop_parentheses(text))


_BOILERPLATE_LADDER = (
    ("verified_text", lambda text: text),
    ("drop_rounding_sentence", _drop_rounding_sentence),
    ("drop_intermediate_units", _drop_intermediate_units),
    ("drop_node_labels", _drop_node_labels),
    ("drop_parentheses", _drop_parentheses),
    ("squeeze_operator_spaces", _squeeze_operator_spaces),
)


def _fit_verified_assistant(prefix: list, text: str, *,
                            max_length: int = MAX_LENGTH) -> tuple[str, str]:
    """Drop non-operator boilerplate only until the byte bound fits.

    The ladder stops at the first rendering that still carries every
    verified number, sign and final answer. Rows that already fit keep
    the locked assistant text.
    """
    if not isinstance(prefix, list) or len(prefix) != 2:
        raise ValueError("Verified prefix must be system and user")
    for name, transform in _BOILERPLATE_LADDER:
        fitted = transform(text)
        _assert_arithmetic_preserved(text, fitted)
        bound = prove_qwen_completion_bound(
            {"messages": [*prefix, {"role": "assistant", "content": fitted}]})
        if bound["upper_bound_tokens"] <= max_length:
            return fitted, name
    raise ValueError(
        "Verified assistant still exceeds the token budget after boilerplate trim")


def _trace_family(text: str) -> str:
    if "kg/min" in text or ("CHECK:" in text and " kg" in text and "EUR" not in text):
        return "segmented_fuel"
    if "CHECK:" in text and ("Cent" in text or "EUR" in text):
        return "ordered_price"
    return "other"


def _load_offline_qwen_tokenizer():
    """Use a local pinned snapshot when one is already on disk. Never download."""
    from neral_hf_identity import BASE_MODEL, BASE_REVISION, validate_hf_snapshot
    explicit = os.environ.get("NERAL_QWEN_SNAPSHOT", "").strip()
    candidates = []
    if explicit:
        candidates.append((Path(explicit), True))
    candidates.append((
        Path.home() / ".cache" / "neral" / "qwen25-smoke" / BASE_REVISION
        / "installed_models" / "hf-qwen25-coder-7b",
        False,
    ))
    for folder, required in candidates:
        if folder.is_symlink() or not folder.is_dir():
            if required:
                raise ValueError("NERAL_QWEN_SNAPSHOT is not a local model directory")
            continue
        if not ((folder / "tokenizer.json").is_file()
                and (folder / "tokenizer_config.json").is_file()):
            if required:
                raise ValueError("NERAL_QWEN_SNAPSHOT has no local tokenizer")
            continue
        os.environ.update({
            "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        })
        try:
            snapshot = validate_hf_snapshot(folder, verify_hashes=False)
        except ValueError:
            if required:
                raise
            continue
        if (snapshot["base_model"] != BASE_MODEL
                or snapshot["revision"] != BASE_REVISION):
            raise ValueError("Offline snapshot is not the pinned Qwen revision")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(
            str(folder), use_fast=True, local_files_only=True,
            trust_remote_code=False)
    return None


def assert_verified_steps_fit_max_length(rows, *, max_length: int = MAX_LENGTH,
                                         tokenizer=None, chat_template: str | None = None
                                         ) -> dict:
    """Fail closed unless every row would pass tokenized_completion.

    With a local tokenizer this calls that function unchanged. Otherwise it
    uses the byte upper bound, which must not be shorter than a real Qwen
    tokenization of the same rendered chat.
    """
    if type(max_length) is not int or max_length < 2:
        raise ValueError("Invalid completion max length")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Verified-step corpus is empty")
    if tokenizer is not None:
        if chat_template is None:
            from neral_hf_identity import CHAT_TEMPLATE as chat_template
        lengths = []
        for index, row in enumerate(rows):
            try:
                encoded = tokenized_completion(
                    row, tokenizer, chat_template=chat_template,
                    max_length=max_length)
            except ValueError as exc:
                raise ValueError(
                    f"Verified-step row {index} rejected by tokenized_completion: {exc}"
                ) from exc
            exact = len(encoded["input_ids"])
            upper = prove_qwen_completion_bound(row)["upper_bound_tokens"]
            if upper < exact:
                raise ValueError(
                    "Byte length model under-estimated the offline Qwen tokenizer "
                    f"at row {index}: bound {upper}, tokenizer {exact}"
                )
            lengths.append(exact)
        return {
            "method": "offline_qwen_tokenizer",
            "offline_qwen_tokenizer": True,
            "max_length": max_length,
            "rows": len(rows),
            "max_upper_bound_tokens": max(lengths),
            "min_upper_bound_tokens": min(lengths),
            "rows_over_max_length": 0,
            "rows_at_max_length": sum(length == max_length for length in lengths),
            "min_assistant_pieces": None,
            "approximation": (
                "Exact local tokenizer via tokenized_completion. The byte "
                "bound was at least that long on every row."
            ),
            "training_authorized": False,
            "claimed_score_gain": False,
            "release_validated": False,
        }
    bounds = []
    spans = []
    over = []
    for index, row in enumerate(rows):
        proof = prove_qwen_completion_bound(row)
        bounds.append(proof["upper_bound_tokens"])
        spans.append(proof["assistant_pieces"])
        if proof["upper_bound_tokens"] > max_length:
            content = row["messages"][2]["content"]
            over.append((proof["upper_bound_tokens"], _trace_family(content)))
    if over:
        families = {}
        for _bound, family in over:
            families[family] = families.get(family, 0) + 1
        raise ValueError(
            "verified_steps corpus exceeds "
            f"{max_length} tokens under the Qwen byte upper bound: "
            f"{len(over)} row(s), max {max(bound for bound, _family in over)}, "
            f"families {families}"
        )
    return {
        "method": "qwen2_byte_bpe_upper_bound",
        "offline_qwen_tokenizer": False,
        "max_length": max_length,
        "rows": len(rows),
        "max_upper_bound_tokens": max(bounds),
        "min_upper_bound_tokens": min(bounds),
        "rows_over_max_length": 0,
        "rows_at_max_length": sum(bound == max_length for bound in bounds),
        "min_assistant_pieces": min(spans),
        "approximation": _LENGTH_APPROXIMATION,
        "training_authorized": False,
        "claimed_score_gain": False,
        "release_validated": False,
    }


def _summarize_verified_trim(notes: list[dict]) -> dict:
    over = [note for note in notes if note["upper_bound_before_trim"] > MAX_LENGTH]
    families = {}
    for note in over:
        family = note["family"]
        families[family] = families.get(family, 0) + 1
    levels = {name: 0 for name, _transform in _BOILERPLATE_LADDER}
    for note in notes:
        levels[note["level"]] += 1
    return {
        "counterfactual_rows": len(notes),
        "rows_over_max_length": len(over),
        "max_upper_bound_tokens": max(note["upper_bound_before_trim"] for note in notes),
        "by_family": families,
        "boilerplate_trim_levels": levels,
    }


def _screen_prompts() -> list[str]:
    """Development screens. The composition holdout is checked exactly, below."""
    from scripts.neral_v6_fresh_dev_eval import build_cases
    from scripts.neral_v61_new_dev_eval import cases72
    return ([row["prompt"] for row in build_cases()]
            + [row["prompt"] for row in cases72()])


def _holdout_prompts() -> list[str]:
    return [
        make_case(family, "holdout", index)["messages"][1]["content"]
        for family in FAMILIES
        for index in range(PER_FAMILY["holdout"])
    ]


def _reject_protected_overlap(rows: list[dict]) -> None:
    """Block screens by exact text and 10-grams.

    The composition holdout reuses family boilerplate, so a 10-gram check
    false-alarms on hundreds of legal train rows. Exact prompt identity is
    the split the generator already enforces.
    """
    from scripts.neral_decontamination import normalize, shingles
    screens = _screen_prompts()
    screen_exact = {normalize(text) for text in screens}
    screen_grams = set().union(*(shingles(text, n=10) for text in screens))
    holdout_exact = {normalize(text) for text in _holdout_prompts()}
    for row in rows:
        prompt = row["messages"][1]["content"]
        key = normalize(prompt)
        if (key in screen_exact or key in holdout_exact
                or (shingles(prompt, n=10) & screen_grams)):
            raise ValueError("Training prompt overlaps a screen or the composition holdout")


def build_experiment(*, counterfactual_target: str = "compact"
                     ) -> tuple[list[dict], list[dict], dict]:
    """Generate train/val only; never train on the synthetic 100-case holdout.

    ``compact`` is the shipped v6 recipe. ``verified_steps`` changes only the
    128 counterfactual assistant texts. Prompts, final answers, learning rate
    and step count stay the same. Neither target is a claimed score gain.
    """
    if counterfactual_target not in {"compact", "verified_steps"}:
        raise ValueError("Unknown counterfactual target")
    data, sources = compositional_data()
    if sources["counts"] != {"train": 600, "validation": 100, "holdout": 100}:
        raise ValueError("Unreviewed split counts")
    if sources["training_allowed"] is not False:
        raise ValueError("Source data unexpectedly marked approved")
    def verified_messages(split):
        originals = [json.loads(line)
                     for line in data[split].decode("utf8").splitlines()]
        expected = [make_case(family, split, i) for family in FAMILIES
                    for i in range(PER_FAMILY[split])]
        if originals != expected:
            raise ValueError("Generator examples or source hashes were modified")
        return [{"messages":record["messages"]} for record in originals]
    train = verified_messages("train")
    validation = verified_messages("validation")
    reason_rows, proofs = candidate()
    if len(reason_rows) != 128 or len(proofs) != 128:
        raise ValueError("Counterfactual reasoning source changed")
    verified_trim_notes = []
    for original, proof in zip(reason_rows, proofs):
        if counterfactual_target == "compact":
            content = _compact_trace(proof, original)
        else:
            verified = _verified_step_target(proof, original)
            pre_row = {"messages": [*original["messages"][:2],
                                    {"role": "assistant", "content": verified}]}
            pre_bound = prove_qwen_completion_bound(pre_row)["upper_bound_tokens"]
            content, level = _fit_verified_assistant(original["messages"][:2], verified)
            verified_trim_notes.append({
                "family": proof["family"],
                "upper_bound_before_trim": pre_bound,
                "level": level,
            })
        train.append({"messages":[*original["messages"][:2],
                                   {"role": "assistant", "content": content}]})
    rng = random.Random(SEED)
    rng.shuffle(train)
    users = [row["messages"][1]["content"] for row in train + validation]
    if len(users) != 828 or len(set(users)) != len(users):
        raise ValueError("Exact task overlap or incomplete training corpus")
    for split in (train, validation):
        for row in split:
            if (set(row) != {"messages"} or
                    [m.get("role") for m in row["messages"]] !=
                    ["system","user","assistant"] or
                    any(not isinstance(m.get("content"), str) or
                        not m["content"].strip() for m in row["messages"])):
                raise ValueError("Unverified conversation pair")
    meta = {
        "schema": 1,
        "status": "NERAL_V6_R16_120_EXPERIMENTAL_TRAIN_CANDIDATE",
        "source_counts": dict(SOURCES),
        "train_rows": len(train), "validation_rows": len(validation),
        "holdout_excluded": 100,
        "steps": STEPS, "max_length": MAX_LENGTH,
        "lora_rank": 16, "adapter_lineage": "v5-r16-480",
        "optimizer": "paged_adamw_8bit",
        "gradient_accumulation": 4, "microbatch": 1,
        "learning_rate": 1e-5,
        "train_sha256": sha256(_jsonl_bytes(train)).hexdigest(),
        "validation_sha256": sha256(_jsonl_bytes(validation)).hexdigest(),
        "original_generator_sha256": sha256(
            Path(__file__).with_name("neral_v6_compositional.py").read_bytes()
        ).hexdigest(),
        "reasoning_generator_sha256": sha256(
            Path(__file__).with_name("neral_reasoning_train_candidate.py").read_bytes()
        ).hexdigest(),
        "source_validation_completed": True,
        "tokenizer_verified": False, "human_semantic_review_performed": False,
        "training_allowed_without_manual_gpu_optin": False,
        "training_performed": False,
        "independent_transfer_proven": False,
        "release_validated": False,
        "adapter_promoted": False,
    }
    if counterfactual_target == "verified_steps":
        _reject_protected_overlap(train + validation)
        length_preflight = assert_verified_steps_fit_max_length(
            train + validation, tokenizer=_load_offline_qwen_tokenizer())
        length_preflight["before_boilerplate_trim"] = _summarize_verified_trim(
            verified_trim_notes)
        meta = {
            **meta,
            "status": "NERAL_V6_VERIFIED_STEP_TARGET_NOT_TRAINED",
            "counterfactual_target": "verified_steps",
            "isolated_change": "restore_verified_arithmetic_in_128_counterfactual_targets",
            "learning_rate_unchanged": True,
            "replay_ratio_unchanged": True,
            "claimed_score_gain": False,
            "length_preflight": length_preflight,
        }
    return train, validation, meta


def training_contract(output: Path, *, counterfactual_target: str = "compact") -> dict:
    """Materialize deterministic training candidate, create-only and private."""
    output = Path(output)
    if (output.is_symlink() or output.exists() or
            not output.parent.is_dir() or output.parent.is_symlink()):
        raise ValueError("New, isolated training-data directory required")
    train, val, meta = build_experiment(counterfactual_target=counterfactual_target)
    paths = {
        "train.jsonl": _jsonl_bytes(train),
        "validation.jsonl": _jsonl_bytes(val),
        "manifest.json": (json.dumps(meta, indent=2, sort_keys=True) + "\n").encode("utf8"),
    }
    if any(len(x) >= 2_000_000 for x in paths.values()):
        raise ValueError("Pilot data too large")
    output.mkdir(exist_ok=False)
    for name, content in paths.items():
        (output / name).write_bytes(content)
    return meta


def tokenized_completion(row: dict, tokenizer, *, chat_template: str,
                         max_length: int = MAX_LENGTH) -> dict:
    """No truncation, answer-only labels; require exact verified Qwen boundary."""
    messages = row["messages"]
    prompt = tokenizer.apply_chat_template(
        messages[:2], chat_template=chat_template,
        tokenize=True, add_generation_prompt=True)
    full = tokenizer.apply_chat_template(
        messages, chat_template=chat_template, tokenize=True,
        add_generation_prompt=False)
    if (not isinstance(prompt,list) or not isinstance(full,list) or
            full[:len(prompt)] != prompt or
            len(full) > max_length or len(full) <= len(prompt)):
        raise ValueError("Qwen completion-mask boundary mismatch or truncation")
    if len(full) - len(prompt) < 2:
        raise ValueError("No assistant answer for loss")
    labels = [-100]*len(prompt) + full[len(prompt):]
    return {"input_ids": full, "attention_mask": [1]*len(full), "labels": labels}


def _gpu_is_free() -> tuple[int, int]:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,utilization.gpu",
         "--format=csv,noheader,nounits"],
        check=True, capture_output=True, text=True, timeout=12)
    parts = result.stdout.strip().splitlines()[0].split(",")
    if len(parts) != 2:
        raise ValueError("Unreadable NVIDIA GPU")
    free, busy = map(lambda x: int(x.strip()), parts)
    if free < 10000 or busy > 20:
        raise ValueError("RTX GPU occupied, insufficient VRAM or high utilization")
    return free, busy


def _gpu_train(args, *, data_factory=build_experiment, steps=STEPS,
               lr=1e-5, authorize=AUTH,
               status="NERAL_V6_120_STEPS_TRAINED_EXPERIMENT_ONLY") -> dict:
    # Keep the original v6 defaults unchanged; allow explicit reviewed
    # experiments to supply a different provenance-checked data factory.
    if (args.authorize != authorize or type(steps) is not int or
            not 1 <= steps <= 120 or not 0 < lr <= 1e-5):
        raise ValueError("Unsafe training authorization, step count or LR")
    root = Path(__file__).resolve().parents[1]
    dest = Path(args.out)
    model_dir = Path(args.base)
    adapter_dir = Path(args.v5_adapter)
    if (dest.exists() or dest.is_symlink() or not dest.parent.is_dir()
            or dest.parent.is_symlink() or
            dest.resolve().is_relative_to(root) or
            dest.resolve().is_relative_to(model_dir.resolve()) or
            dest.resolve().is_relative_to(adapter_dir.resolve())
            or not model_dir.is_dir() or not adapter_dir.is_dir()):
        raise ValueError("Never overwrite v5 adapter or model or write into repo")
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
        "WANDB_DISABLED": "true", "TOKENIZERS_PARALLELISM": "false",
    })
    free_mib, _ = _gpu_is_free()
    from neral_hf_identity import (
        validate_adapter, validate_hf_snapshot, CHAT_TEMPLATE,
    )
    snapshot = validate_hf_snapshot(model_dir, verify_hashes=True)
    predecessor = validate_adapter(adapter_dir, snapshot)
    old_sha = predecessor["files"]["adapter_model.safetensors"]
    if predecessor["release_validated"] is not False:
        raise ValueError("Expected nonpromoted experimental v5")
    expected_sha = "2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5"
    if old_sha != expected_sha:
        raise ValueError("Wrong v5 predecessor SHA; refuse unreviewed lineage")
    train, val, meta = data_factory()
    if (len(train) != meta.get("train_rows") or
            len(val) != meta.get("validation_rows") or
            meta.get("release_validated") is not False or
            meta.get("training_performed") is not False):
        raise ValueError("Untrusted experimental training dataset")
    import torch
    from datasets import Dataset
    from peft import PeftModel, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer,
        TrainingArguments, TrainerCallback,
    )
    if (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
            or torch.cuda.device_count() != 1):
        raise ValueError("Single BF16 CUDA GPU required")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), use_fast=True, local_files_only=True,
        trust_remote_code=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    train_tokens = [tokenized_completion(x,tokenizer,chat_template=CHAT_TEMPLATE)
                    for x in train]
    val_tokens = [tokenized_completion(x,tokenizer,chat_template=CHAT_TEMPLATE)
                  for x in val]
    max_used = max(len(row["input_ids"]) for row in train_tokens + val_tokens)
    data_sha = meta["train_sha256"]
    def collate(rows):
        max_len=max(len(row["input_ids"]) for row in rows)
        return {
            "input_ids": torch.tensor(
                [row["input_ids"]+[tokenizer.pad_token_id] *
                 (max_len-len(row["input_ids"])) for row in rows],
                dtype=torch.long),
            "attention_mask": torch.tensor(
                [row["attention_mask"]+[0] *
                 (max_len-len(row["attention_mask"])) for row in rows],
                dtype=torch.long),
            "labels": torch.tensor(
                [row["labels"]+[-100] *
                 (max_len-len(row["labels"])) for row in rows],
                dtype=torch.long),
        }

    quant=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
          bnb_4bit_use_double_quant=True,
          bnb_4bit_compute_dtype=torch.bfloat16)
    started=time.monotonic()
    torch.cuda.set_per_process_memory_fraction(0.94,0)
    torch.cuda.reset_peak_memory_stats(0)
    foundation=AutoModelForCausalLM.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False,
        attn_implementation="sdpa", quantization_config=quant,
        device_map={"":0}, torch_dtype=torch.bfloat16)
    if not getattr(foundation,"is_loaded_in_4bit",False):
        raise ValueError("Expected actual NF4 loading")
    foundation.config.use_cache=False
    foundation=prepare_model_for_kbit_training(
        foundation,use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False})
    policy=PeftModel.from_pretrained(
        foundation,str(adapter_dir),is_trainable=True,local_files_only=True)
    trainable=[n for n,p in policy.named_parameters() if p.requires_grad]
    if not trainable or any("lora_" not in n for n in trainable):
        raise ValueError("Only original LoRA weights may be trainable")
    count=sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if count != 40_370_176:
        raise ValueError("Unexpected trainable v5 LoRA parameter count")

    run_logs=[]
    class Guard(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            entry={k:v for k,v in (logs or {}).items()
                   if isinstance(v,(int,float))}
            if any(not math.isfinite(float(v)) for v in entry.values()):
                raise FloatingPointError("Nonfinite training loss/metric")
            if len(run_logs) < 160:
                run_logs.append({"step":state.global_step,**entry})
    # Hugging Face Trainer may create output_dir during initialization.
    # Reserve our private root once, BEFORE TrainingArguments/Trainer exist.
    dest.mkdir(exist_ok=False)
    output_temp=dest/"trainer-temp"
    config=TrainingArguments(
        output_dir=str(output_temp), max_steps=steps,
        per_device_train_batch_size=1, per_device_eval_batch_size=1,
        gradient_accumulation_steps=4, learning_rate=lr,
        bf16=True, fp16=False,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant":False},
        max_grad_norm=1.0, lr_scheduler_type="cosine", warmup_ratio=0.05,
        eval_strategy="no",save_strategy="no",logging_steps=5,
        seed=SEED,dataloader_num_workers=0,
        remove_unused_columns=False,report_to="none",push_to_hub=False,
        disable_tqdm=True,
    )
    trainer=Trainer(
        model=policy,args=config,train_dataset=Dataset.from_list(train_tokens),
        eval_dataset=Dataset.from_list(val_tokens),
        data_collator=collate,callbacks=[Guard()],
    )
    before=time.monotonic()
    result=trainer.train()
    if trainer.state.global_step != steps or not math.isfinite(result.training_loss):
        raise ValueError("Incomplete/nonfinite requested optimizer steps")
    evals=trainer.evaluate()
    eval_loss=float(evals.get("eval_loss",float("nan")))
    if not math.isfinite(eval_loss):
        raise ValueError("Nonfinite held-out validation loss")
    total_vram=int(torch.cuda.get_device_properties(0).total_memory//1024**2)
    peak=int(torch.cuda.max_memory_reserved(0)//1024**2)
    if total_vram-peak<500:
        raise ValueError("Unacceptable memory headroom; adapter not saved")
    dest_adapter=dest/"candidate-adapter"
    policy.save_pretrained(str(dest_adapter),safe_serialization=True)
    weights=dest_adapter/"adapter_model.safetensors"
    if not weights.is_file():
        raise ValueError("Experiment did not produce safetensors LoRA")
    new_sha=sha256(weights.read_bytes()).hexdigest()
    if new_sha==old_sha:
        raise ValueError("No measurable change to v5 adapter weights")
    if sha256((adapter_dir/"adapter_model.safetensors").read_bytes()).hexdigest()!=old_sha:
        raise ValueError("Existing v5 adapter unexpectedly mutated")
    report={
        **meta,"status":status,
        "predecessor_adapter_sha256":old_sha,
        "candidate_adapter_sha256":new_sha,
        "foundation_revision":snapshot["revision"],
        "free_vram_before_mib":free_mib,
        "max_tokenized_length":max_used,
        "steps_completed":int(trainer.state.global_step),
        "training_loss":float(result.training_loss),
        "validation_loss":eval_loss,
        "peak_reserved_vram_mib":peak,
        "total_vram_mib":total_vram,
        "elapsed_seconds":round(time.monotonic()-before,2),
        "total_elapsed_seconds":round(time.monotonic()-started,2),
        "lora_trainable_parameters":count,
        "training_performed":True,
        "release_validated":False,
        "independent_transfer_proven":False,
        "adapter_promoted":False,
        "smoke_diagnostic_only":True,
        "recent_train_logs":run_logs[-12:],
    }
    (dest/"scalar-private.json").write_text(
        json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+"\n",
        encoding="utf8")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("--mode",choices=("cpu-data","train-local-gpu"),required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--base",type=Path)
    parser.add_argument("--v5-adapter",type=Path)
    parser.add_argument("--authorize",default="")
    parser.add_argument("--counterfactual-target",
                        choices=("compact","verified_steps"), default="compact")
    args=parser.parse_args(argv)
    if args.mode=="cpu-data":
        report=training_contract(
            args.out, counterfactual_target=args.counterfactual_target)
        print(json.dumps({"status":report["status"],"train_rows":report["train_rows"],
                          "validation_rows":report["validation_rows"],
                          "training_performed":False},sort_keys=True))
    else:
        if args.base is None or args.v5_adapter is None:
            parser.error("Explicit pinned local Qwen and v5 adapter required")
        if args.counterfactual_target == "verified_steps":
            if args.authorize != VERIFIED_STEPS_AUTH:
                parser.error("Verified-step training requires its own permit")
            result=_gpu_train(
                args,
                data_factory=lambda: build_experiment(
                    counterfactual_target="verified_steps"),
                authorize=VERIFIED_STEPS_AUTH,
                status="NERAL_V6_VERIFIED_STEPS_120_EXPERIMENTAL",
            )
        else:
            if args.authorize != AUTH:
                parser.error("Compact v6 training requires its own permit")
            result=_gpu_train(args)
        print(json.dumps({
            "status":result["status"],"steps_completed":result["steps_completed"],
            "adapter_sha256":result["candidate_adapter_sha256"],
            "validation_loss":result["validation_loss"],
            "peak_reserved_vram_mib":result["peak_reserved_vram_mib"],
            "release_validated":False,
        },sort_keys=True))


if __name__=="__main__":
    main()
