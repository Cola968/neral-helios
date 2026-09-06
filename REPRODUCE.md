# Reproducing the local preview

Python 3.12 is used. No external model weights or cloud model API are required.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-cuda.txt -r requirements.txt
```

Start `START_NERAL.bat`. Without an installed own checkpoint the app offers the local project tools, SOLVE and static code checking; raw language-model generation fails clearly instead of serving random weights.

Own checkpoints are local/release artifacts, not ordinary Git blobs. Place the supplied 20M checkpoint at `checkpoints/neral20m.pt` with its matching `.sha256` file. Optional inference exports go into `export/pilot_02/`, `export/neral40m/` and `export/neral50m/`, each with matching SHA-256 files. Do not replace these with another model family. Preview 02 remains default when present because no larger checkpoint has passed the free-language quality gate.

CPU tool/model tests without local training assets:

```powershell
.venv/Scripts/python.exe -m pytest tests/test_projects.py tests/test_model50.py -q
```

The complete local regression suite additionally requires the original handoff checkpoints and locally prepared curricula. Missing private artifacts are not downloaded silently.

## General-language pilot

Run each step explicitly and inspect its report before advancing:

```powershell
.venv/Scripts/python.exe fetch_wiki_pilot.py
.venv/Scripts/python.exe prepare_pretrain50.py
.venv/Scripts/python.exe free_generation50.py --prepare
.venv/Scripts/python.exe pretrain50.py --tune
.venv/Scripts/python.exe pretrain50.py --tokens 1000000 --out runs/neral50m_pilot_a
.venv/Scripts/python.exe free_generation50.py --checkpoint runs/neral50m_pilot_a/best_validation.pt --out reports/free50_pilot_a.json
```

The frozen benchmark generator refuses to overwrite an existing suite. Keep the generated manifest. Do not include `benchmarks50/holdout.jsonl` in routine runs or training. Repository-provided regression prompts should be used unchanged.

Continuation is token-budgeted and explicit; for example, an additional pilot targeting **cumulative** five million tokens:

```powershell
.venv/Scripts/python.exe pretrain50.py --resume runs/neral50m_pilot_a/latest.pt --tokens 5000000 --out runs/neral50m_pilot_b
```

Do not interpret a lower validation loss as a release gate. First inspect raw free generation, repetition, UTF-8 validity, language consistency, EOS, coding and math. A larger language-model run and instruction tuning remain separate future stages. No `best_free_generation` checkpoint is designated until that gate passes.

Data downloads use the named Wikipedia 20231101 snapshot. The row API does not support the repository revision pin used for the dataset card: saved SHA-256 hashes pin the actual bytes. Preserve original cached files to reproduce a specific run exactly. The pipeline does not ingest held-out binary streams during training or validation.

The local energy/cost estimate remains unfilled until integrated power and the user's electricity tariff are known. No per-token cloud charges apply to this standalone inference code; this is not a claim about development-service charges.
