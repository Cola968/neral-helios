NERAL local Qwen read-only synthetic E2E test approved 2026-11-01. No model downloads, API payments, training, checkpoint changes, persistent chat writes, or uncontrolled workflows. One push only.
Retry 1: previous attempt stopped at Windows PowerShell policy before model inference. Exact same permitted synthetic three-case probe only; no additional tasks.
Retry 2: one final diagnostic of GPU free VRAM and utilization, still refuse model inference if GPU busy. The prior two attempts made zero model calls.
Last bounded attempt: only proceed if local Ollama already running and GPU under conservative utilization limit for two readings. No further automatic retries after this.
Follow-up diagnosis authorized under same read-only troubleshooting request: identify exact static AssistantError category for observed HTTP400, no full generated code or answer logging, no training/downloads, fixed bounded GPU idle guard.

2026-11-01 follow-up authorization: rerun synthetic read-only Qwen/Flask diagnostic to classify previous HTTP400. Respect 8 GiB free and two <=35% utilization samples, no model installs, shell commands beyond existing verified probe, weight or training changes.

2026-11-01 explicitly authorized verification of strict metadata compatibility fix commit 79fe4d7523c4d51bdac3b51d814f936d75dc0f85. Synthetic Flask-to-preinstalled-Qwen test only; stop if GPU guard is not met, no other action.

2026-11-01 final user-authorized verification: after valid real code_check call and truncated 128-token final answer, validate bounded 320-token code case with exact tool provenance. Respect existing GPU idle guard, no new downloads, weights, training or payments.
