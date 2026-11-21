# NERAL V3.10: private holdout and verified notes

The user-approved one-step RTX4070 QLoRA smoke on the 97 actual passing V3.9 benchmark examples was executed locally on Nov 21, 2026. The trainer generated an `adapter_saved_not_evaluated` manifest with `global_step=1`, finite positive gradient norm `4.723523`, distinct before/after LoRA weight hashes, PyTorch reserved peak `8.057 GiB`, and sampled whole-GPU peak `9261 MiB`. The originally failed workflow `44889263214` was a **PowerShell Start-Process null ExitCode false-negative**; independent read-only owner workflow `44889590555` confirms the manifest and service HTTP200. The smoke adapter is *not* promoted as a chat model.

After the CPU and Windows PowerShell syntax gates, the owner-only activation job:
- rechecks the exact saved adapter manifest and source dataset digest,
- freezes 100 fresh tasks across math, logic, Python, knowledge and web into **D: private storage**, preserving the SHA256 manifest and refusing overwrites,
- installs a fail-closed **exact-match** note lookup on `data/verified_training_dataset.jsonl` in the user's local NERAL service and merges the original 97 graded examples, without uploading conversations,
- restarts only the identity-checked NERAL scheduled task with local HTTP200 proof and rollback on failure.

**Important limitations:** The 97 seed examples are benchmark-graded; not all have independent runtime proof. Web notes are excluded from instant recall to prevent stale claims. New verified everyday examples are appended locally, and exact re-questions can be answered without loading the model. This is retrieval, **not automatic model-weight learning or generalization**. New adapter improvements must be separately trained and evaluated on the private frozen holdout. Never commit the private holdout or local user dataset.
