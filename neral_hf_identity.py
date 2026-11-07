"""Stable, self-contained NERAL Qwen2.5 Coder HF/LoRA identity verification.

Pinned compatibility contract from the immutable Qwen trainer (commit
a082738f9145722dec443141e5d7801a3ead93af). Helios' native
model_catalog.py is for its own 20/50M checkpoints and is NOT the same module.
Never execute, pickle, dynamically import or download model files.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
BASE_REVISION = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
# Regressed or not independently improved experiments may still be assessed in
# isolated research jobs, but never through the normal sealed runtime path.
# v6: measured degradation 21/72 vs v5 24/72.
# v6.1: no measured gain 16/72 vs v5 16/72; not release validated.
QUARANTINED_ADAPTER_SHA256 = frozenset({
    "8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd",
    "33e939f6063d4215807f593816a057e88d8938b30615c6000d6a0bf8c7965279",
})
HF_CONFIG = {
    "model_type": "qwen2",
    "architectures": ["Qwen2ForCausalLM"],
    "hidden_size": 3584,
    "intermediate_size": 18944,
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "vocab_size": 152064,
    "max_position_embeddings": 32768,
    "eos_token_id": 151645,
    "bos_token_id": 151643,
    "tie_word_embeddings": False,
}
CHAT_TEMPLATE = (
    "{% for message in messages %}"
    "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>\n' }}"
    "{% endfor %}{% if add_generation_prompt %}"
    "{{ '<|im_start|>assistant\n' }}{% endif %}"
)
TARGETS = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
METADATA = {
    "config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "generation_config.json", "vocab.json", "merges.txt", "model.safetensors.index.json",
}
ADAPTER_ALLOW = {
    "adapter_config.json", "adapter_model.safetensors", "neral-adapter.json",
    "neral-training-manifest.json", "training-report.json", "README.md",
}
REJECT_CONFIG_KEYS = (
    "auto_mapping", "modules_to_save", "use_dora", "use_rslora",
    "rank_pattern", "alpha_pattern", "layers_to_transform", "layers_pattern",
    "layer_replication", "trainable_token_indices", "target_parameters",
    "megatron_config", "exclude_modules", "loftq_config", "eva_config",
    "corda_config", "fan_in_fan_out", "use_qalora", "lora_bias",
)


def _json(path: Path, limit: int = 1024 * 1024):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError("Unexpected metadata file")
    def no_duplicates(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON metadata key")
            result[key] = value
        return result
    return json.loads(path.read_text(encoding="utf8"), object_pairs_hook=no_duplicates)


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _flat_directory(folder: Path, minimum: int, maximum: int):
    if folder.is_symlink() or folder.resolve() != folder.absolute() or not folder.is_dir():
        raise ValueError("Unsafe local model directory")
    entries = list(folder.iterdir())
    if not minimum <= len(entries) <= maximum:
        raise ValueError("Unexpected local model file count")
    for file in entries:
        if file.is_symlink() or not file.is_file() or file.resolve().parent != folder:
            raise ValueError("Unsafe model file or link")
    return entries


def validate_hf_snapshot(folder, verify_hashes=False) -> dict:
    """Validate approved revision/config/flat safetensors and optional full SHA-256."""
    folder = Path(folder)
    entries = _flat_directory(folder, 5, 64)
    manifest = _json(folder/"helios-manifest.json")
    if (not isinstance(manifest, dict) or set(manifest) !=
            {"base_model", "revision", "files"} or
            manifest["base_model"] != BASE_MODEL or
            manifest["revision"] != BASE_REVISION or
            not isinstance(manifest["files"], dict)):
        raise ValueError("Unapproved Qwen model/revision/manifest")
    files = manifest["files"]
    if set(files) != {p.name for p in entries}-{"helios-manifest.json"}:
        raise ValueError("Snapshot file inventory mismatch")
    if not {"config.json", "tokenizer.json", "tokenizer_config.json"} <= set(files):
        raise ValueError("Missing Qwen tokenizer/config")
    weights = set()
    for name, digest in files.items():
        if not isinstance(digest, str) or not re.fullmatch("[a-f0-9]{64}", digest):
            raise ValueError("Unexpected snapshot SHA")
        if re.fullmatch(r"model(?:-\d{5}-of-\d{5})?\.safetensors", name):
            weights.add(name)
        elif name not in METADATA:
            raise ValueError("Non-safetensors or unexpected model file")
    if not weights:
        raise ValueError("Model weights missing")
    if "model.safetensors.index.json" in files:
        index = _json(folder/"model.safetensors.index.json", 8*1024*1024)
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        if (not isinstance(mapping, dict) or not mapping or
                any(type(v) is not str for v in mapping.values()) or
                set(mapping.values()) != weights or
                "model.safetensors" in weights):
            raise ValueError("Invalid Qwen safetensors shard index")
    elif weights != {"model.safetensors"}:
        raise ValueError("Shards require explicit index")
    config = _json(folder/"config.json")
    tokenizer = _json(folder/"tokenizer_config.json")
    if (not isinstance(config, dict) or
            any(config.get(k) != v for k,v in HF_CONFIG.items()) or
            any(k in config for k in ("auto_map", "quantization_config", "peft_config")) or
            not isinstance(tokenizer, dict) or "auto_map" in tokenizer or
            tokenizer.get("tokenizer_class") not in ("Qwen2Tokenizer", "Qwen2TokenizerFast")):
        raise ValueError("Unexpected Qwen model/tokenizer configuration")
    if sum((folder/n).stat().st_size for n in weights) > 20*1024**3:
        raise ValueError("Qwen weights exceed 20 GB")
    if any((folder/n).stat().st_size > 64*1024**2 for n in files.keys()-weights):
        raise ValueError("Qwen metadata too large")
    if verify_hashes:
        for name,digest in files.items():
            if _sha(folder/name) != digest:
                raise ValueError("Qwen model SHA-256 mismatch")
    identity = hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    return {"folder": folder, "base_model": BASE_MODEL,
            "revision": BASE_REVISION, "sha256": identity}


def validate_adapter(folder, base) -> dict:
    """Reject unsealed adapters, extra executable modules and wrong base hashes."""
    folder = Path(folder)
    entries = _flat_directory(folder, 3, 6)
    if {file.name for file in entries} - ADAPTER_ALLOW:
        raise ValueError("Unsafe adapter inventory")
    manifest = _json(folder/"neral-adapter.json")
    if (not isinstance(manifest, dict) or set(manifest) != {
            "schema", "product", "version", "base_model", "base_revision",
            "base_snapshot_integrity", "dataset_sha256", "files",
            "release_validated",
        } or manifest["schema"] != 1 or manifest["product"] != "NERAL" or
            not isinstance(manifest["version"],str) or
            not re.fullmatch(r"\d+\.\d+\.\d+(?:-pilot\.\d+)?",manifest["version"]) or
            manifest["base_model"] != BASE_MODEL or
            manifest["base_revision"] != base["revision"] or
            manifest["base_snapshot_integrity"] != base["sha256"] or
            not isinstance(manifest["release_validated"], bool) or
            not isinstance(manifest["dataset_sha256"],str) or
            not re.fullmatch("[0-9a-f]{64}",manifest["dataset_sha256"])):
        raise ValueError("Unapproved NERAL adapter provenance")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != {
        "adapter_config.json","adapter_model.safetensors"
    }:
        raise ValueError("Invalid adapter inventory or SHA")
    for name, digest in files.items():
        path = folder/name
        if (not path.is_file() or not isinstance(digest, str) or
                not re.fullmatch("[0-9a-f]{64}",digest) or
                path.stat().st_size > 512*1024**2 or _sha(path) != digest):
            raise ValueError("Adapter integrity failed")
    if files["adapter_model.safetensors"] in QUARANTINED_ADAPTER_SHA256:
        raise ValueError("NERAL-Adapter nicht freigegeben oder wegen Regression gesperrt")
    config = _json(folder/"adapter_config.json")
    if not isinstance(config, dict):
        raise ValueError("Adapter config malformed")
    rank = config.get("r")
    if (config.get("peft_type") != "LORA" or
            config.get("task_type") != "CAUSAL_LM" or
            type(rank) is not int or rank not in (16,32) or
            config.get("lora_alpha") != rank or
            config.get("lora_dropout") != 0 or
            config.get("bias") != "none" or
            config.get("init_lora_weights",True) is not True or
            not isinstance(config.get("target_modules"),list) or
            set(config["target_modules"]) != TARGETS or
            any(config.get(k) for k in REJECT_CONFIG_KEYS)):
        raise ValueError("Unapproved NERAL LoRA recipe")
    return manifest
