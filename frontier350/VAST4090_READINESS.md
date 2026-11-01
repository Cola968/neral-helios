# Arctus350 on Vast.ai — one RTX 4090 (24 GB), safely staged

**Status: preparation code only. No GPU rented, training started, weights
uploaded, model downloaded, or money spent by this pull request.**
This is a separate Draft on `arctus350-chat-cloud-prep`, where the real
native Arctus350 BF16 trainer already exists. It is **not** based on the
newer Helios UI Drafts #203–#206, which do not include the native
continued-pretraining entrypoint.

## Offer from the user's screenshot (29 Oct 2026)

- Texas, US, 1× RTX 4090 / 24 GB VRAM; host reportedly Verified;
  99.76% displayed reliability; 32 GB system RAM; ~554 GB displayed disk.
- GPU price **$0.366 per hour, plus bandwidth and possibly storage**.
  Price, availability and disk terms may change; check again before
  confirming RENT. The listed machine uses PCIe 3.0 x16.
- 5 USD is an example **planning budget**, not an enforceable billing cap.
  Never assume stopping a GPU instance ends storage charges. Terminate/
  delete the instance after backing up all data. Disable automatic
  payments where possible; verify outstanding billing separately.

## Why the script is strict

Issue #141 records a native verified 2B+ Arctus checkpoint
(`2,073,161,216` processed tokens) with original SHA-256:

`f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7`

**That digest is a record, not a model download link.** The actual
checkpoint, complete AdamW optimizer state and locked tokenizer still
exist only in user-controlled storage. This kit refuses to start from
random weights, a Qwen checkpoint or a model-only export. The full
training schedule from the resume checkpoint must still have
unprocessed tokens; if it is exhausted, it **fails closed** rather
than resetting the learning-rate schedule. A legitimate subsequent
schedule requires a separately designed, evaluated continuation protocol.

The original native Arctus BPE data prep writes **uint16 `.bin` shards**
and a `schema:2` manifest containing exact hashes and a frozen
benchmark-contamination gate. Data staged by PR #205 are JSONL and
**are not directly compatible with this trainer**. Do not feed that
output directly into the native trainer.

## Stage 1 — before renting (no charge or GPU)

On GitHub, review this Draft PR and verified CI. Run on a local trusted
machine with the code already checked out:

```bash
python -m frontier350.vast4090_guard
```

This is a **read-only JSON plan**. You may run it from the phone with
a Python-capable shell after obtaining source code; it does not use
any server or require model weights.

Before spending money, locate on your trusted storage:

1. The original 2B+ native `resume.pt` **including optimizer/RNG**,
   digest-checked against the original report.
2. The locked 16K BPE `tokenizer.json` with SHA-256
   `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`.
3. Matching, already prepared, provenance-verified
   `data/manifest.json`, `data/train/*.bin`, `data/valid/*.bin`.
4. A **trusted backup destination** for newly trained results,
   not just the ephemeral rental disk.
5. A safe method to transfer private files without copying a GitHub
   personal access token, passwords or SSH private keys into a
   potentially untrusted community-host container.

**Security trade-off:** even if an offer says "verified", the GPU host
operates the remote machine. This code cannot guarantee the host
cannot inspect unencrypted training data/weights. If Arctus weights
must remain private, use a stronger-trust provider or refrain from
uploading them to a marketplace host.

## Stage 1.5 — NOW: pack the private trial files before paying for a GPU

A new **local-only** exporter `frontier350/vast4090_transfer.py` completes
the original checkpoint/data transfer workflow. It was tested with synthetic
files; **it has not accessed or transferred the owner's real PC files**.

On the trusted PC (or a trusted machine where your original checkpoint and
locked corpus actually reside), first run the read-only command:

```powershell
python -m frontier350.vast4090_transfer
```

Locate the original **full optimizer** checkpoint (not an inference-only
`best_validation.pt`) and native `schema=2` BPE `.bin` train/valid
corpus. Check its SHA against Issue #141. To prepare a small pilot subset,
replace the three input path placeholders with actual local paths:

```powershell
python -m frontier350.vast4090_transfer --pack `
  --checkpoint "D:\\NERAL\\runs\\ORIGINAL-resume.pt" `
  --checkpoint-sha256 f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7 `
  --tokenizer "D:\\NERAL\\tokenizer.json" `
  --data-manifest "D:\\NERAL\\BPE-CORPUS\\manifest.json" `
  --out "D:\\NERAL-PRIVATE\\vast-transfer-01"
```

This is not a claim that these paths exist. The script **fails before
creating the output** if the checkpoint digest, complete AdamW optimizer,
original training-token schedule, locked tokenizer, or corpus integrity
cannot be verified. For the normal 131,072-token GPU engineering pilot it
selects **at least 512K train and 16K validation BPE tokens** from the
hash-checked original corpus, not the complete 1B+ corpus. All files
are copied to a new private folder; the originals remain untouched.
The exported JSONL-free `schema=2` subset is directly readable by the
native `train350_compiled.py` input reader. A smaller data subset is
useful for **performance/integration experiments**, *not* for claiming
new high-quality pretraining, novel documents or a validated improvement.

The output uses this structure:

```text
vast-transfer-01/
  inputs/
    original-resume.pt
    tokenizer.json
    data/
      manifest.json
      train/*.bin
      valid/*.bin
  transfer-receipt.json
```

The command prints an **independently recorded**
`receipt_sha256` and original checkpoint hash. Keep both digests
on your trusted phone/PC; **do not store SSH keys or passwords in the
transfer folder or commit the folder to GitHub**. The private folder
contains full original optimizer weights and must be treated as sensitive.

Now you can rent the GPU *after* checking actual package size, chosen
host, live price, and a **separate secured backup destination**.
A transfer typically requires a one-time action from the machine
holding those private files. ChatGPT/GitHub cannot reach local
PC disks or initiate the transfer itself.

After renting, move the package over an authenticated SSH/SFTP connection
provided by Vast. For a trusted Windows/WSL machine with `rsync`:

```bash
# Replace HOST, PORT and PRIVATE-LOCAL-PATH with values from your actual instance
rsync -avP -e "ssh -p <PORT>" /trusted/PRIVATE-LOCAL-PATH/vast-transfer-01/ root@<HOST>:/workspace/vast-transfer-01/
```

Do **not** put an actual SSH private key, GitHub token, or cloud
storage credentials inside the untrusted GPU host. Verify the SSH
host key using Vast's documented connection details, do not
disable host-key checks to get a command working. Do not upload
any files until you consciously accept the provider's data
confidentiality trade-off.

**On the Vast machine, BEFORE any training**, independently check:

```bash
python -m frontier350.vast4090_transfer --verify \
  --root /workspace/vast-transfer-01 \
  --checkpoint-sha256 f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7 \
  --receipt-sha256 <RECEIPT_SHA_PRINTED_ON_TRUSTED_PC>
```

Then pass these already verified file locations into the existing
`vast4090_guard --prepare` command in Stage 2:

- `--checkpoint /workspace/vast-transfer-01/inputs/original-resume.pt`
- `--tokenizer /workspace/vast-transfer-01/inputs/tokenizer.json`
- `--data-manifest /workspace/vast-transfer-01/inputs/data/manifest.json`

Check CPU/RAM and the exact 4090 offer again. The `--verify`
operation reads and hashes only; no CUDA/remote billing action
is taken by the code.

**Important possible blocker:** the Issue #141 checkpoint reports
2,073,161,216 processed tokens but does **not** establish whether
its original `target_tokens` is still larger. If the recorded schedule
is already complete, this packer refuses to create a misleading
resume plan. A separate, validated continuation recipe will be needed
before further model weight updates.

---

## Stage 2 — only after the host is rented and those files are present

Select a Vast image with PyTorch 2.8.0, compatible NVIDIA CUDA and
Python 3.12. Avoid pasting credentials into container shells. The
existing training code is CUDA/BF16-specific; it is **not** Docker
image provisioning automation. Use official
[Vast.ai documentation](https://docs.vast.ai/) for SSH/image/connect/
storage setup; install dependencies only on the selected host.

Using a prepackaged **credential-free** repository checkout
containing `arctus350-chat-cloud-prep` + this PR (rather than
`git clone` with a personal access token on the host),
verify the GPU without training:

```bash
nvidia-smi
python -m frontier350.vast4090_guard
```

After explicit permission to transfer the checkpoint and corpus,
run the following. The paths are **examples**, not claims that your
files have already been uploaded:

```bash
python -m frontier350.vast4090_guard \
  --prepare \
  --checkpoint /workspace/inputs/original-resume.pt \
  --checkpoint-sha256 f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7 \
  --tokenizer /workspace/inputs/tokenizer.json \
  --data-manifest /workspace/inputs/data/manifest.json \
  --out /workspace/neral-runs/vast4090-pilot-01 \
  --extra-tokens 131072 \
  --gpu-hourly-usd 0.366 \
  --budget-usd 5
```

This checks actual `nvidia-smi` (exactly one RTX 4090), RAM,
dataset shard hashes, original tokenizer and model/optimizer checkpoint.
A verified source checkpoint is copied to the NEW, isolated output
folder. The source remains untouched, and the output folder may **not**
already exist. The checker writes
`start-vast4090.sh` and `prepare-report.json`.
It **does not start training**.

If the code prints "Original checkpoint LR schedule already exhausted",
stop: do not fabricate a new target or overwrite metadata. The
2B+ checkpoint's actual `target_tokens` must be examined before a
continuation can be legitimately planned.

## Stage 3 — separately approved and supervised GPU test

Only after checking the completed preparation report and the live Vast
billing estimate:

```bash
bash /workspace/neral-runs/vast4090-pilot-01/start-vast4090.sh
```

The emitted command reuses `frontier350/train350_compiled.py`
with the original optimizer and a small bounded
`--stop-after-tokens` increment; its `--compile-mode none` startup
avoids unverified compilation/memory overhead. It uses sequence
length 512, microbatch 2, gradient accumulation 8, frequent
checkpoints and validation. The trainer now loads trusted
checkpoints via the existing restricted `secure_checkpoint350.py`
instead of unrestricted `weights_only=False`.

No guarantee of speed, learning improvement, automatic backup, or
maximum spending. Supervise the actual `processed_tokens`,
loss, errors and output hashes. New training data do not automatically
turn into NERAL intelligence; compare the frozen capability benchmarks
and checkpoint quality before any promotion.

After saving output `resume.pt`, `metrics.jsonl`,
`summary.json` and checksums outside Vast, explicitly terminate
the instance in the Vast control panel. **STOP alone may still incur
storage charges.** No automatic Vast API or background rental is
called from this kit.

## Sicherere spätere Checkpoint-Auswahl (V6 Bilingual Recovery)

Der V6-Auswahlmechanismus wurde im **gleichen Draft**, ohne GPU,
gehärtet. Unveränderte Holdout-Fälle reichen nicht: Beide
Sprachrichtungen müssen **je 200 Fälle** haben und die exakten Treffer
`correct`/`total` mit den gemeldeten Prozentwerten übereinstimmen.
Das gilt für Baseline **und** Kandidaten. `NaN`, `Infinity`, fehlende
Daten und negative Validierungsverluste werden nicht akzeptiert.

Der Entwicklungs-/Validierungs-Gate bewahrt den verifizierten
`checkpoint_sha256` für jeden Kandidaten auf; die unabhängige
Holdout-Auswertung muss genau diesen Checkpoint bewertet haben.
Unterschiedliche Hashes führen zu
`checkpoint_identity_mismatch`, nie zu einem stillen Merge oder
einer Modellbeförderung. Alle drei V6-GPU-Evaluatoren laden zudem
Checkpoints mit dem schon existierenden eingeschränkten
`secure_checkpoint350.load_checkpoint`, statt mit unrestricted Pickle.

Die zugehörigen CPU-Tests:
```bash
python -m pytest -q tests/test_select_bilingual_recovery_v6_safety.py
```

**Stand 1. November 2026:** Der ursprüngliche lokale 2,073-Mrd.-Token-
Checkpoint wurde auf dem eigenen Windows-Runner per SHA-256 und
schreibgeschütztem Metadaten-Preflight geprüft (Model, Optimizer, RNG,
Lernratenplan). Das ist weiterhin **kein** neuer Trainingslauf und
**kein** Nachweis besserer Arctus-Ergebnisse. Die Windows-Sicherheits-
tests für Draft #207 bestanden zuletzt mit **60/60**.
Neue Prüfergebnisse sind stets dem jeweiligen Commit zuzuordnen.

## GitHub-Actions-unabhängiger Checkpoint-Selector-Test

Der echte V6-Selector lässt sich mit **rein synthetischen
Bewertungsberichten** und **ausschließlich Python-Standardbibliothek**
prüfen. Das funktioniert auf einem Rechner mit Python, auch wenn
GitHub-Actions keine Runner bereitstellt:

```bash
python frontier350/offline_selector_selftest.py
```

Der Befehl umfasst sieben Szenarien: reguläre Auswahl,
falscher Gate-Kandidat, falsche Checkpoint-SHA, NaN, gefälschte
Prozentzahlen, eine Holdout-Ranglisten-Manipulation sowie
falsche Baseline-Trefferzahlen. Erwartet wird
`state=PASS_OFFLINE` mit `tests_passed=7`.
Es werden weder Gewichte noch Daten eingelesen, keine GPU
gemietet und keine Netzwerkanfragen ausgeführt.

**Trennung:** Der Sprach-Holdout ist ein striktes
*Nicht-Regressions-Veto*, kein Tie-Breaker oder Ranking-Signal.
Bei gleicher Validierungsloss entscheidet der
Entwicklungswert, nicht der höhere Holdout-Wert. Der Holdout
ist **kein** unabhängiges finales Benchmark-Zeugnis, weil er
zur Kandidatenfreigabe benutzt wird. Das abschließende
Benchmark-Set muss gesondert eingefroren bleiben.

Ein solcher Selbsttest ersetzt nicht die vollständige
GitHub-CI, reale 350M-Checkpoint-Evaluationen und GPU-Tests.
Die Offline-Probe wurde mit einer separaten lokalen Kopie des
Selectors und synthetischen Daten am 31.10.2026 durchgeführt.

## Originale BPE-Trainingsdaten: reale Windows-Prüfung am 1. November 2026

Der schreibgeschützte Metadaten-Prüfer
`frontier350/readonly_corpus_inventory.py` wurde hinzugefügt und
auf dem eigenen Windows-Runner mit **drei bereits vorhandenen lokalen
BPE-Korpora** ausgeführt. Es wurden keine privaten Dateipfade,
Rohtexte oder Tokens in den Ergebnisbericht übernommen.

| Korpus | Train-Tokens laut Manifest | Valid-Tokens laut Manifest | Dateien Train/Valid | Metadaten | SHA-256-Dateiintegrität |
|---|---:|---:|---:|---|---|
| Recovery-Pilot | 20.000.000 | 200.000 | 2 / 1 | kompatibel | bestanden |
| Continuation | 500.000.000 | 100.000 | 10 / 1 | kompatibel | bestanden |
| Foundation | 1.050.000.000 | 50.000.000 | 21 / 1 | kompatibel | bestanden |

Die SHA-256-Prüfung erfolgte durch die **bestehende**
`validate_data_manifest`-Routine gegen die originalen lokalen
Shard-Manifeste. GitHub-PC-Run:
[41302805510](https://github.com/Cola968/neral-helios/actions/runs/41302805510).
Synthetische Windows-Metadaten-Prüftests: **11 bestanden, 1 übersprungen**.

Die Summe von **1,57 Milliarden deklarierten Train-Tokens** ist
ausdrücklich **kein** Hinweis auf 1,57 Milliarden neue oder zusätzliche
Lern-Tokens; die Datensätze können sich untereinander und mit früher
trainierten Quellen überlappen. SHA-256 sichert nur lokale Integrität,
nicht Provenienz, Nutzungsrechte, historische Deduplizierung,
Train-vs-Holdout-Leakage, die korrekte Fortsetzung des bestehenden
2,073-Milliarden-Token-Korpus oder Trainingsfreigabe.

Die Ausgabe setzt deshalb immer
`training_ready=false`, `corpus_provenance_verified=false`
und `training_started=false`. Kein Training, Upload, Cloud-GPU-Job
oder Merge wurde ausgelöst.

## CPU-only verification

```bash
python -m pytest -q tests/test_vast4090_guard.py
python -m py_compile frontier350/vast4090_guard.py \
  frontier350/train350.py frontier350/train350_compiled.py
```

These do not need a GPU, user weights or any rental. No automatic
training job is enabled.
