# Neral / Helios — self-hosted RTX runner

This runner is intended only for the private `Cola968/neral-helios` repository. Training data and model checkpoints remain local on the Windows PC. GitHub receives source code plus small benchmark/test reports only.

## 1. Add the repository runner

In GitHub open:

`neral-helios -> Settings -> Actions -> Runners -> New self-hosted runner`

Choose **Windows / x64** and follow the repository-specific commands shown by GitHub. Do not paste the temporary registration token into issues, commits, chat messages, or source files.

During runner configuration, add the custom label:

`neral-gpu`

For the first setup, run the runner interactively with `run.cmd`. This makes diagnosis easier. Installing it as a Windows service can be done later.

## 2. Keep private assets outside the Git checkout

Use a persistent directory, for example:

```text
C:\NeralPrivate\
  pretrain50_data\
    train_de.bin
    train_en.bin
    val_de.bin
    val_en.bin
  runs\
    neral50m_pilot_a\
      latest.pt
      best_validation.pt
  reports\
  .venv\
```

The workflow creates an NTFS junction from its temporary checkout to `pretrain50_data`; the actual corpus is never copied into the repository.

## 3. Configure runner environment

In PowerShell for the Windows account that starts `run.cmd`:

```powershell
[Environment]::SetEnvironmentVariable('NERAL_ASSETS_ROOT', 'C:\NeralPrivate', 'User')
[Environment]::SetEnvironmentVariable('NERAL_PYTHON', 'C:\NeralPrivate\.venv\Scripts\python.exe', 'User')
```

Close and reopen the terminal after setting them, then start `run.cmd` from the GitHub runner directory.

If your existing Neral virtual environment lives elsewhere, set `NERAL_PYTHON` to that exact `python.exe` instead. It must contain the CUDA PyTorch build and the packages from the Neral requirements files.

## 4. First run: health check only

Open **Actions -> Neral GPU Lab -> Run workflow** and choose:

- `mode = healthcheck`
- keep the other defaults

The job must report CUDA available, BF16 available, the NVIDIA GPU name, VRAM, driver, temperature and power state. It performs no training and does not require private checkpoints.

## 5. Benchmark existing 50M checkpoint

Choose:

- `mode = benchmark`
- `run_name = neral50m_stage_b`
- `max_new_tokens = 96`

If Stage B does not yet exist, the workflow evaluates `runs/neral50m_pilot_a/latest.pt`.

It runs:

1. frozen 100-prompt raw-generation regression,
2. benchmark v3 (conditional preferences, true byte BPB, base continuation, chat formatting, context lengths, CUDA forward throughput/VRAM),
3. repository regression tests.

Small JSON/XML reports are persisted under `C:\NeralPrivate\reports\runner_<run_name>` and uploaded as a short-lived GitHub Actions artifact. Weights and `.bin` corpus files are explicitly excluded.

## 6. Continue 50M training

Choose:

- `mode = train_and_benchmark`
- `target_tokens = 5000000`
- `run_name = neral50m_stage_b`
- `max_new_tokens = 96`

The token target is **cumulative**. The workflow resumes Stage B if it already has a `latest.pt`; otherwise it resumes the original `neral50m_pilot_a/latest.pt`.

After training, it automatically evaluates the best-validation checkpoint, without promoting it to the default model. A lower validation loss is not sufficient for promotion: free generation, repetition, EOS, language stability and functional benchmarks still have to pass.

## Security model

- Workflow trigger is manual only (`workflow_dispatch`).
- The job targets `[self-hosted, windows, x64, neral-gpu]`.
- Repository permission is read-only for workflow contents.
- GitHub checkout credentials are not persisted.
- Corpus and checkpoints remain under `NERAL_ASSETS_ROOT`.
- Upload directory rejects `.pt`, `.pth`, `.bin` and `.safetensors` files.
- No automatic training runs on `push` or pull requests.

Do not register this runner for arbitrary public repositories. A self-hosted runner executes repository workflow code with the permissions of its Windows account.
