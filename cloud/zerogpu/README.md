---
title: NERAL LoRA – private test
emoji: 🧪
colorFrom: gray
colorTo: blue
sdk: gradio
app_file: app.py
python_version: "3.10"
suggested_hardware: zero-a10g
short_description: Private NERAL adapter test; not an official release
---

# NERAL LoRA ZeroGPU (experimental)

**This GitHub folder is not a running demo.** Create a **private** Hugging Face Gradio Space with **ZeroGPU** hardware and copy app.py, policy.py, requirements.txt and this README to its **root**. Read docs/NERAL_HF_ZEROGPU.md in the original GitHub repository first.

The app intentionally fails to start without NERAL_ADAPTER_REPO and a genuine compatible LoRA checkpoint. It does not mislabel foundation-only Qwen as NERAL.

ZeroGPU + bitsandbytes has not yet been GPU-tested; first deployment may require compatibility adjustments.
