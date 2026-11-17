"""Opt-in NERAL Luna/Terra/Helios locally pinned HF runtime.

Luna uses the unchanged Qwen3.5-4B checkpoint, with thinking disabled.
Terra and Helios use exactly the same frozen Qwen3.5-9B v2-full PEFT
adapter (different output limits); only independent Z3/Python-Docker proofs
may set verified=True. This is NOT an unmeasured speed or quality claim.

Model loads are lazy, offline and mutually exclusive under the parent local
preview chat lock. Nothing is downloaded or executed on the host.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
from pathlib import Path
import re
import time
from collections.abc import Mapping

from scripts.neral_three_tier_runtime_inventory import frozen_v2_full

LUNA = "qwen3.5:4b"
HELIOS = "qwen3.5:9b+neral-v2-full"
TIERS = {"Luna": 96, "Terra": 256, "Helios": 768}


def _offline():
    os.environ.update({
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1", "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false", "WANDB_DISABLED": "true",
    })


def probe_cuda_free_after_cache_flush(torch, minimum_free_mib: int):
    """Release only this process's idle CUDA cache before rejecting a 9B switch.

    Do not terminate other GPU workers and do not lower the original free-VRAM
    admission threshold. A fresh driver measurement after empty_cache remains
    authoritative; allocated tensors are never force-freed.
    """
    def snapshot():
        free, _ = torch.cuda.mem_get_info(0)
        allocated = torch.cuda.memory_allocated(0)
        reserved = torch.cuda.memory_reserved(0)
        return tuple(round(size / 1024**2, 1)
                     for size in (free, allocated, reserved))
    free, allocated, reserved = snapshot()
    if free < minimum_free_mib:
        torch.cuda.synchronize(0)
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize(0)
        free, allocated, reserved = snapshot()
    return free, allocated, reserved


class FrozenQwen35HF:
    """Lazy NF4 CUDA inference; frozen 9B PEFT weights or unchanged 4B."""

    def __init__(self, checkpoint: str | Path, *, full_root: str | Path | None = None):
        base = Path(checkpoint)
        if base.is_symlink() or not base.is_dir() or not (base / "config.json").is_file():
            raise ValueError("Pinned local Qwen3.5 model snapshot is missing")
        self.base = base.resolve(strict=True)
        self.root = None
        self.adapter_path = None
        self.adapter_sha256 = None
        if full_root is not None:
            full = Path(full_root)
            if not frozen_v2_full(full):
                raise ValueError("Frozen Helios v2-full receipt or adapter SHA mismatch")
            self.root = full.resolve(strict=True)
            self.adapter_path = self.root / "adapter"
            manifest = json.loads((self.root / "manifest.json").read_text("utf-8"))
            self.adapter_sha256 = manifest["frozen_adapter_sha256"]
            self.model = HELIOS
            self.minimum_free_mib = 9500
        else:
            # The exact unmodified 4B HF snapshot was used in GPU run 44067939731.
            if not (base / "model.safetensors.index.json").is_file():
                raise ValueError("Complete local Qwen3.5-4B HF snapshot missing")
            self.model = LUNA
            self.minimum_free_mib = 4500
        self._model = None
        self._base_model = None
        self._tokenizer = None
        self.free_mib_before_load = None
        self.allocated_mib_before_load = None
        self.reserved_mib_before_load = None

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load(self):
        if self.loaded:
            return
        _offline()
        import torch
        from transformers import AutoTokenizer, Qwen3_5ForCausalLM, BitsAndBytesConfig
        from peft import PeftModel
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Exactly one local CUDA GPU is required")
        (self.free_mib_before_load, self.allocated_mib_before_load,
         self.reserved_mib_before_load) = probe_cuda_free_after_cache_flush(
             torch, self.minimum_free_mib)
        if self.free_mib_before_load < self.minimum_free_mib:
            raise RuntimeError("Insufficient free GPU memory; no other model was stopped")
        if self.root is not None and not frozen_v2_full(self.root):
            raise RuntimeError("Frozen Helios adapter bytes changed")
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dtype)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                str(self.base), local_files_only=True, trust_remote_code=False,
                use_fast=True)
            base = Qwen3_5ForCausalLM.from_pretrained(
                str(self.base), local_files_only=True, trust_remote_code=False,
                quantization_config=config, dtype=dtype, device_map={"": 0},
                low_cpu_mem_usage=True)
            model = (PeftModel.from_pretrained(
                base, str(self.adapter_path), is_trainable=False,
                local_files_only=True)
                if self.adapter_path is not None else base)
            model.eval()
            self._tokenizer = tokenizer
            self._base_model = base
            self._model = model
        except Exception:
            self.unload()
            raise

    def unload(self):
        self._model = None
        self._base_model = None
        self._tokenizer = None
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize(0)
                torch.cuda.empty_cache()
                torch.cuda.synchronize(0)
        except ImportError:
            pass

    def respond_chat(self, turns: list[dict], *, max_new_tokens: int) -> str:
        if type(max_new_tokens) is not int or max_new_tokens not in (96, 256, 768):
            raise ValueError("Unapproved NERAL inference budget")
        if self.model == LUNA and max_new_tokens != 96:
            raise ValueError("Luna is explicitly bounded to 96 tokens")
        if self.model == HELIOS and max_new_tokens == 96:
            raise ValueError("Never impersonate the 4B Luna model")
        self._load()
        import torch
        encoded = self._tokenizer.apply_chat_template(
            turns, tokenize=True, add_generation_prompt=True,
            enable_thinking=False)
        if isinstance(encoded, Mapping):
            encoded = encoded["input_ids"]
        if type(encoded) is not list or not 1 <= len(encoded) <= 4096-max_new_tokens:
            raise ValueError("Bounded local HF context exceeded")
        ids = torch.tensor([encoded], dtype=torch.long, device="cuda")
        try:
            with torch.inference_mode():
                output = self._model.generate(
                    input_ids=ids, do_sample=False, max_new_tokens=max_new_tokens,
                    max_time=50.0 if max_new_tokens == 768 else 25.0,
                    use_cache=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id)
            produced = output[0, len(encoded):].tolist()
            eos = {self._tokenizer.eos_token_id}
            for special in ("<|im_end|>", "<|endoftext|>"):
                token_id = self._tokenizer.convert_tokens_to_ids(special)
                if type(token_id) is int and token_id >= 0:
                    eos.add(token_id)
            if not produced or produced[-1] not in eos:
                raise RuntimeError("HF response did not finish inside its token/time budget")
            final = self._tokenizer.decode(produced, skip_special_tokens=True).strip()
            if "<think>" in final:
                if "</think>" not in final:
                    raise RuntimeError("Unclosed private model reasoning")
                final = final.rsplit("</think>", 1)[-1].strip()
            if (not final or "<think>" in final or "</think>" in final or
                    len(final.encode("utf-8")) > 32768):
                raise RuntimeError("Incomplete or untrusted HF final answer")
            return final
        finally:
            del ids


class ThreeTierRuntime:
    def __init__(self, luna, helios, *, legacy_backend=None, execute_python=None):
        if (getattr(luna, "model", None) != LUNA or
                getattr(luna, "adapter_sha256", None) is not None or
                getattr(helios, "model", None) != HELIOS or
                re.fullmatch(r"[0-9a-f]{64}",
                             str(getattr(helios, "adapter_sha256", ""))) is None or
                not callable(getattr(luna, "respond_chat", None)) or
                not callable(getattr(helios, "respond_chat", None)) or
                not callable(getattr(luna, "unload", None)) or
                not callable(getattr(helios, "unload", None))):
            raise ValueError("Exactly pinned Luna and frozen Helios v2-full required")
        if legacy_backend is not None and not callable(getattr(legacy_backend, "unload", None)):
            raise ValueError("Legacy backend needs explicit safe unload")
        self.luna = luna
        self.helios = helios
        self.legacy_backend = legacy_backend
        self.active_backend = None
        if execute_python is None:
            from neral_adaptive_planner import run_python_in_docker
            execute_python = run_python_in_docker
        if not callable(execute_python):
            raise ValueError("Docker execution callback missing")
        self.execute_python = execute_python

    def release_for_legacy(self):
        """Drop only this service's HF CUDA state before an Ollama reply.

        Caller owns the preview-wide GPU lock; the source-verified path never
        invokes an Ollama model or bypasses the lock.
        """
        if self.active_backend is not None:
            current = self.active_backend
            current.unload()
            self.active_backend = None

    def _switch(self, chosen):
        if self.active_backend is chosen:
            return
        if self.active_backend is None:
            if self.legacy_backend is not None:
                self.legacy_backend.unload()
        else:
            self.active_backend.unload()
        self.active_backend = chosen

    def _verified_source(self, turns: list[dict]) -> dict | None:
        if len(turns) != 2:
            return None
        question = turns[-1]["content"]
        from neral_nl_verified_route import strict_source_certificate
        from neral_verified_source_code import source_program
        try:
            full = strict_source_certificate(question) is not None or source_program(question) is not None
        except (TypeError, ValueError, RuntimeError):
            return None
        if not full:
            return None
        from neral_adaptive_planner import run_adaptive_planner
        from neral_python_source_router import verify_python_full_source

        def no_model(*_args, **_kwargs):
            raise RuntimeError("Trusted full-source proof must not call a model")
        report = run_adaptive_planner(
            turns, no_model, no_model,
            self.execute_python, verify_python_full_source)
        if (report.get("verified") is True and
                report.get("safe_to_promote") is True and
                report.get("model_generated") is False and
                report.get("response_source") in (
                    "independent_source_python", "independent_source_parser_z3") and
                report.get("proof_backend") in (
                    "z3_unsat", "independent_enumeration",
                    "trusted_source_arithmetic", "independent_unique_enumeration",
                    "independent_user_source_docker")):
            return report
        return None

    def answer(self, turns: list[dict], tier: str) -> dict:
        if tier not in TIERS:
            raise ValueError("Unapproved mode tier")
        if (type(turns) is not list or len(turns) < 2 or
                turns[-1].get("role") != "user"):
            raise ValueError("Invalid local chat turns")
        if tier in ("Terra", "Helios"):
            checked = self._verified_source(turns)
            if checked is not None:
                return {
                    **checked, "tier": tier, "mode": tier.upper(),
                    "model": self.helios.model, "model_generated": False,
                    "adapter_loaded": False, "adapter_sha256": None,
                    "model_origin": "local_source_verified_tools_only",
                    "quality_improvement_proven": False,
                }
        chosen = self.luna if tier == "Luna" else self.helios
        self._switch(chosen)
        response = chosen.respond_chat(turns, max_new_tokens=TIERS[tier])
        if type(response) is not str or not response.strip():
            raise RuntimeError("Selected local HF model returned empty result")
        if chosen is self.helios and not chosen.loaded:
            raise RuntimeError("Helios adapter loading was not confirmed")
        return {
            "tier": tier, "mode": tier.upper(), "text": response,
            "model": chosen.model, "model_generated": True,
            "verified": False, "safe_to_promote": False,
            "verification_scope": "none", "proof_backend": None,
            "adapter_loaded": bool(chosen is self.helios and chosen.loaded),
            "adapter_sha256": (self.helios.adapter_sha256
                               if chosen is self.helios and chosen.loaded else None),
            "model_origin": ("local_neral_lora_experimental"
                             if chosen is self.helios else
                             "local_foundation_not_neral_adapter"),
            "quality_improvement_proven": False,
        }


def register_three_tier_api(app, runtime: ThreeTierRuntime, *, busy):
    # The CUDA evaluation environment need not install the Flask preview.
    from flask import jsonify, request
    from neral_local_chat_api import _messages
    if not isinstance(runtime, ThreeTierRuntime):
        raise ValueError("Three-tier runtime not installed")
    if not callable(getattr(busy, "acquire", None)):
        raise ValueError("Global NERAL GPU lock required")

    # Hook the existing Ollama prepare_model closure, not just the externally
    # exposed app._neral_prepare_model: normal, streaming, THINK, and file
    # routes all use that closure and must release HF CUDA allocations.
    app._neral_before_legacy_model = runtime.release_for_legacy

    @app.get("/api/neral/chat/three-tier/status")
    def three_tier_status():
        return jsonify(enabled=True, tiers=list(TIERS),
                       luna=runtime.luna.model, helios=runtime.helios.model,
                       adapter_loaded=runtime.helios.loaded,
                       adapter_sha256=(runtime.helios.adapter_sha256
                                       if runtime.helios.loaded else None),
                       quality_improvement_proven=False)

    @app.post("/api/neral/chat/three-tier")
    def three_tier_generate():
        if not request.is_json:
            return jsonify(error="JSON für Drei-Modell-Chat erforderlich."), 415
        payload = request.get_json(silent=True)
        if (type(payload) is not dict or
                set(payload) not in ({"tier", "prompt"}, {"tier", "prompt", "history"}) or
                payload.get("tier") not in TIERS):
            return jsonify(error="Ungültiger Luna/Terra/Helios-Modus."), 400
        try:
            turns = _messages({key: value for key, value in payload.items()
                               if key != "tier"})
            augment = getattr(app, "_neral_cpu_context", None)
            if callable(augment):
                augment(turns)
        except (TypeError, ValueError, UnicodeError):
            return jsonify(error="Ungültiger oder zu langer Chatverlauf."), 400
        if not busy.acquire(blocking=False):
            return jsonify(error="NERAL bearbeitet bereits eine Antwort."), 409
        try:
            started = time.monotonic()
            result = runtime.answer(turns, payload["tier"])
            return jsonify({**result, "seconds": round(time.monotonic()-started, 4),
                            "runtime_tier_verified": True})
        except Exception:
            return jsonify(error="Drei-Modell-Laufzeit nicht bereit oder Modellantwort unvollständig."), 503
        finally:
            busy.release()
