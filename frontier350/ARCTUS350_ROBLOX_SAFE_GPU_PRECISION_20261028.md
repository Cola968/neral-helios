# Arctus 350M — erster echter, Roblox-kompatibler RTX-4070-GPU-Befund

**Datum:** 2026-10-28  
**GitHub Actions:** https://github.com/Cola968/neral-helios/actions/runs/40683632316  
**Ausführung:** `W39-GAMINGPC-Neral`, Label `neral-gpu`, NVIDIA GeForce RTX 4070, mit geöffnetem Roblox.

## Test und Sicherheit

- Auf dem Windows-PC meldete der Runner **zwei Roblox-bezogene Prozesse**, Nvidia-Gerät **12.282 MiB insgesamt / 10.265 MiB frei** vor GPU-Eintritt.
- Fail-closed VRAM-Gate: **mindestens 4.096 MiB frei**, sonst keine CUDA-Arbeit. WSL überprüft nach CUDA-Initialisierung erneut den freien Speicher.
- `frontier350/gpu_precision_probe350.py` führt ausschließlich acht AdamW-Schritte auf je **4.096 synthetischen konstanten Gewichten** aus; zwei Wertebereiche (0,02 und 1,0), jeweils BF16 vs FP32.
- **Kein** echtes Arctus-Checkpoint geladen, **kein** SFT-/Pretraining gestartet, **keine** Modelldateien verändert, keine kostenpflichtigen Hugging-Face-Jobs.
- Maximal durch PyTorch erfasste **reservierte 2 MiB** (`max_memory_reserved`), nicht der gesamte durch CUDA-Treiber/WSL verursachte GPU-Speicherbedarf. CUDA meldete 12.281 MiB insgesamt / 11.091 MiB frei nach seiner Initialisierung; **Windows- und WSL-Werte weichen ab** und sind daher nicht austauschbar.
- GitHub-Workflow erfolgreich abgeschlossen, Ergebnis als kleines `arctus350-roblox-safe-synthetic-precision`-Artefakt hochgeladen.

## Tatsächlich gemessene Updates

| Parameterart (Startgewicht) | BF16: Anteil verändert | FP32: Anteil verändert | FP32: mittlerer Betrag der Gewichtsänderung |
|---|---:|---:|---:|
| Lineare Gewichte (0,02) | 0,0 % | 100,0 % | 0,00012007356 |
| RMSNorm-artige Gewichte (1,0) | 0,0 % | 100,0 % | 0,00012397766 |

Beide Tests: `torch.optim.AdamW(fused=True)`, `lr=1.5e-5`, `weight_decay=.03`, `betas=(.9,.95)`, `eps=1e-8`, konstante synthetische Gradienten `0.01`, **acht** Optimizer-Steps.

**Interpretation:** Die Rundungsproblematik des aktuellen BF16-Weight-Storage ist nun **auf der echten RTX 4070 für diese synthetischen Werte nachgewiesen**. Es ist **kein** Nachweis, dass „84,5 % der tatsächlichen Arctus-Gewichte eingefroren“ sind oder dass FP32-Autocast in 12 GB VRAM neben Roblox garantiert passt. Dafür müssen echte Checkpoint-Daten und ein enger, ausdrücklich begrenzter A/B-Test verwendet werden.

## Nächste sichere Schritte

1. Für das echte 350M-Modell zunächst read-only Checkpoint-Identität und verfügbare GPU-Kapazität prüfen. 2B-Foundation und 1B-SAFE eindeutig auseinanderhalten.
2. Ein FP32-Master/BF16-Autocast-Experiment **als getrennte Opt-in-Laufkonfiguration** und mit freiem-VRAM-Gate entwickeln; keine Umschreibung des bestehenden Trainers ohne Test.
3. Vor jeder Inferenz/GPU-Experiment wieder GPU-RAM und Roblox prüfen; bei weniger als 4 GiB frei NICHT starten, bei schwereren Modellen deutlich größeren Puffer fordern. Erst nach echter Peak-Messung Trainingszeit-/VRAM-Grenzen festlegen.

**Qualitäts- und Kostenaussage:** 0 € direkte Rechenkosten, Hardware-GPU-Microexperiment erfolgreich; **Modellqualität nicht gemessen oder verbessert**. PR bleibt Draft, Modellgewichte unverändert.
