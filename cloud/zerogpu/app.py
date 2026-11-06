"""Experimental PRIVATE NERAL LoRA chat on Hugging Face ZeroGPU.

This is not deployed by GitHub and never silently serves foundation-only.
Copy these files into the root of a PRIVATE Gradio Space and set secrets first.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import spaces
import torch
from huggingface_hub import snapshot_download
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from policy import adapter_id_or_fail, prepare_messages, validate_adapter_config

BASE = "Qwen/Qwen2.5-Coder-7B-Instruct"
ADAPTER = adapter_id_or_fail(os.getenv("NERAL_ADAPTER_REPO"))
TOKEN = os.getenv("HF_TOKEN")

# Download only adapter files from the authorized private Hugging Face repository.
path = Path(snapshot_download(
    repo_id=ADAPTER, repo_type="model", token=TOKEN,
    allow_patterns=["adapter_config.json", "adapter_model.safetensors"],
))
config_path = path / "adapter_config.json"
weights_path = path / "adapter_model.safetensors"
if not config_path.is_file() or not weights_path.is_file():
    raise RuntimeError("Der echte NERAL-Adapter fehlt. Kein Foundation-Fallback.")
validate_adapter_config(json.loads(config_path.read_text(encoding="utf-8")), BASE)

# ZeroGPU recommends module-level CUDA placement, emulated at app startup.
tokenizer = AutoTokenizer.from_pretrained(BASE, token=TOKEN, trust_remote_code=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
foundation = AutoModelForCausalLM.from_pretrained(
    BASE, token=TOKEN, trust_remote_code=False, use_safetensors=True,
    device_map={"": 0}, torch_dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    ),
)
model = PeftModel.from_pretrained(foundation, str(path), is_trainable=False)
model.eval()


@spaces.GPU(duration=60)
def reply(message: str, history: list[dict]) -> str:
    messages = [{
        "role": "system",
        "content": "Du bist NERAL, ein experimenteller Coding-Assistent "
                   "mit aktivem NERAL-LoRA auf Qwen2.5-Coder-7B. "
                   "Kennzeichne Unsicherheit und erfinde keine durchgeführten Tools.",
    }] + prepare_messages(message, history)
    ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    )
    if ids.shape[1] > 2048:
        raise gr.Error("Kontext zu lang; bitte Chat zurücksetzen.")
    ids = ids.to("cuda:0")
    with torch.inference_mode():
        output = model.generate(
            ids, max_new_tokens=192, do_sample=False,
            attention_mask=torch.ones_like(ids),
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, ids.shape[1]:], skip_special_tokens=True)


gr.ChatInterface(
    fn=reply,
    type="messages",
    title="NERAL – privater LoRA-Test",
    description="Experimentelle Adapter-Inferenz. Gehostet auf Hugging Face ZeroGPU; "
                "Eingaben werden auf fremder Cloud-Infrastruktur verarbeitet. "
                "Kein Agent, kein Computer Use und keine Release-Freigabe.",
).launch(share=False, show_error=False)
