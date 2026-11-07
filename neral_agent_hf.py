"""Optional CUDA backend: NERAL's local Qwen2.5 Coder foundation + local LoRA.

This backend is loaded explicitly for GPU smoke/runtime. It never downloads
models, executes generated code, trains, or claims validated model quality.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

from neral_agent_autonomy import AgentModelError
from neral_hf_identity import CHAT_TEMPLATE, BASE_REVISION, validate_hf_snapshot, validate_adapter


def enforce_offline():
    for k, v in {
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1", "WANDB_DISABLED": "true",
        "WANDB_MODE": "disabled", "HF_HUB_DISABLE_TELEMETRY": "1",
    }.items():
        os.environ[k] = v


class HFAdapterLocalBackend:
    def __init__(self, base: str | Path, adapter: str | Path, max_new_tokens=384):
        self.base = Path(base).resolve(strict=True)
        self.adapter = Path(adapter).resolve(strict=True)
        if not self.base.is_dir() or not self.adapter.is_dir() or self.base.is_symlink() or self.adapter.is_symlink():
            raise AgentModelError("Lokale Modellverzeichnisse fehlen.")
        if not 64 <= max_new_tokens <= 512:
            raise ValueError("Ungültiges Antwortbudget.")
        file = self.adapter / "adapter_model.safetensors"
        config = self.adapter / "adapter_config.json"
        if not file.is_file() or file.is_symlink() or not config.is_file():
            raise AgentModelError("Adapter und Konfiguration sind nicht vollständig.")
        try:
            settings = json.loads(config.read_text(encoding="utf-8"))
            rank = settings["r"]
            if settings.get("peft_type") != "LORA" or type(rank) is not int or not 4 <= rank <= 64:
                raise ValueError("Unsupported adapter")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise AgentModelError("Adapter-Konfiguration ungültig.") from exc
        enforce_offline()
        self.model = f"NERAL-local-r{rank}-experimental"
        with file.open("rb") as stream:
            self.adapter_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None
        # Shared by NERAL chat and agent: never load/generate twice on 12 GB GPU.
        self._inference_lock = threading.Lock()

    def _load(self):
        if self._model is not None:
            return
        enforce_offline()
        import torch
        from transformers import BitsAndBytesConfig, Qwen2ForCausalLM, Qwen2TokenizerFast
        from peft import PeftModel

        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise AgentModelError("Keine unterstützte lokale CUDA-GPU verfügbar.")
        free, _ = torch.cuda.mem_get_info(0)
        if free < 9 * 1024**3:
            raise AgentModelError("Zu wenig freier GPU-Speicher für den lokalen Adapter.")
        foundation = validate_hf_snapshot(self.base, verify_hashes=True)
        if foundation["revision"] != BASE_REVISION:
            raise AgentModelError("Nicht freigegebene Qwen-Foundation.")
        proof = validate_adapter(self.adapter, foundation)
        if proof["files"]["adapter_model.safetensors"] != self.adapter_sha256:
            raise AgentModelError("Adapter wurde seit Helios-Start verändert.")
        tokenizer = Qwen2TokenizerFast.from_pretrained(str(self.base),
                          local_files_only=True, trust_remote_code=False, token=False)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = Qwen2ForCausalLM.from_pretrained(
            str(self.base), local_files_only=True, trust_remote_code=False, token=False,
            use_safetensors=True, attn_implementation="sdpa", device_map={"": 0},
            dtype=torch.bfloat16, quantization_config=BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
        model = PeftModel.from_pretrained(model, str(self.adapter),
                                          is_trainable=False, local_files_only=True)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model

    def respond(self, messages: list[dict]) -> str:
        with self._inference_lock:
            return self._respond_once(messages)

    def _respond_once(self, messages: list[dict]) -> str:
        try:
            self._load()
            import torch
            tokens = self._tokenizer.apply_chat_template(
                messages, chat_template=CHAT_TEMPLATE, tokenize=True,
                add_generation_prompt=True, return_tensors="pt").to("cuda:0")
            if tokens.shape[1] > 3072:
                raise AgentModelError("Modellkontext überschritten.")
            with torch.inference_mode():
                generated = self._model.generate(
                    tokens, attention_mask=torch.ones_like(tokens),
                    max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self._tokenizer.pad_token_id)
            return self._tokenizer.decode(generated[0, tokens.shape[-1]:],
                                          skip_special_tokens=True)
        except AgentModelError:
            raise
        except Exception as exc:
            raise AgentModelError("Lokale Adapter-Inferenz gescheitert.") from exc
