"""Small dependency-free safety/identity checks for the experimental Space."""
from __future__ import annotations

import re

_REPO_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def adapter_id_or_fail(raw: str | None) -> str:
    value = (raw or "").strip()
    if not _REPO_ID.fullmatch(value) or ".." in value:
        raise ValueError("NERAL_ADAPTER_REPO muss ein explizites Hugging-Face-Modellrepo sein; kein Qwen-Fallback.")
    return value


def validate_adapter_config(config: dict, base_repo: str) -> None:
    if not isinstance(config, dict) or config.get("peft_type") != "LORA":
        raise ValueError("Kein echter PEFT-LoRA-Adapter.")
    rank = config.get("r")
    if type(rank) is not int or not 4 <= rank <= 64:
        raise ValueError("Ungültiger Adapter-Rang.")
    source = str(config.get("base_model_name_or_path") or "").replace(chr(92), "/")
    basename = source.rstrip("/").split("/")[-1].lower()
    # A reviewed Windows local model snapshot can be recorded as the original base.
    if source and source != base_repo and basename not in {
        "hf-qwen25-coder-7b", "qwen2.5-coder-7b-instruct"
    }:
        raise ValueError("Adapter wurde offenbar für eine andere Foundation trainiert.")


def prepare_messages(message: str, history: list[dict] | None) -> list[dict[str, str]]:
    if not isinstance(message, str) or not 1 <= len(message.strip()) <= 4000:
        raise ValueError("Bitte eine Nachricht mit 1 bis 4000 Zeichen eingeben.")
    if not isinstance(history or [], list):
        raise ValueError("Ungültiger Chatverlauf.")
    result = []
    for row in (history or [])[-12:]:
        if not isinstance(row, dict) or row.get("role") not in ("user", "assistant"):
            raise ValueError("Ungültige Rolle im Verlauf.")
        value = row.get("content")
        if not isinstance(value, str) or len(value) > 4000:
            raise ValueError("Ungültiger Verlaufseintrag.")
        result.append({"role": row["role"], "content": value})
    if sum(len(row["content"]) for row in result) > 12000:
        raise ValueError("Verlauf zu lang. Bitte einen neuen Chat starten.")
    result.append({"role": "user", "content": message.strip()})
    return result
