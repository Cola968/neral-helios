# Arctus 350M: authentische GPU-Evidenz 1B vs. 2B vs. Chat-SFT (28.10.2026)

**Status:** Auf dem echten selbst-gehosteten `W39-GAMINGPC-Neral`-Runner mit **zwei erkannten Roblox-Prozessen** ausgeführt. Keine Trainingsläufe gestartet, keine Modellgewichte überschrieben, keine kostenpflichtigen Hugging-Face-Jobs.

## Reproduzierbare Läufe

- Synthetischer BF16-vs-FP32-AdamW-Nachweis: [GitHub Actions #40683632316](https://github.com/Cola968/neral-helios/actions/runs/40683632316).
- Originaler **1B-SAFE**-Checkpoint, echte Kurzantworten: [GitHub Actions #40685210495](https://github.com/Cola968/neral-helios/actions/runs/40685210495).
- Originaler **2B-Fortsetzungscheckpoint** und **1B-basierter SFT-Checkpoint**, dieselben drei Prompts: [GitHub Actions #40685712803](https://github.com/Cola968/neral-helios/actions/runs/40685712803).
- CPU-Prüfungen / Syntax: [GitHub Actions #40686492526](https://github.com/Cola968/neral-helios/actions/runs/40686492526): **53 fokussierte Tests grün**, **93 bestehende Arctus-350M-Tests ausgeführt, 3 optionale übersprungen**.

## Identität des ursprünglichen 350M-Modells

| Checkpoint / Trainingslinie | Metadaten: processed_tokens | SHA-256 | Größe (Bytes) |
|---|---:|---|---:|
| Sicherer Foundation-Stand 1B | 1.010.003.584 | `ef3ee045203f1bfb7b9f501db800408b095aa5ef29da5d22a8a9486e115e9f7c` | 737.503.900 |
| Native 2B-Fortsetzung (lokaler `_continuity_v5`-Stand) | **2.073.161.216** | `f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7` | 737.507.501 |
| Historischer `humanbalanced1200` SFT-Kandidat, basiert auf 1B | 1.010.003.584 **Source-Tokens**, NICHT neue 2B-Foundation | `408998c0d0816527341253fe37c2d9e2f1216c5cad8500dee247d11c72cc8d7b` | 699.749.883 |

Alle drei wurden mit `secure_checkpoint350.load_checkpoint`, der Originalarchitektur `frontier350-v1` mit **349.825.152 Parametern** und dem gelockten 16K-BPE-SHA `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd` geladen. Die SFT-Metadaten bezeichnen ihre **Foundation-Source-Tokens**, nicht die Anzahl der späteren SFT-Tokens.

## Echte Mini-Ausgaben (greedy, EOS, max. 16 neue Tokens)

| Öffentliche, nicht aus der eingefrorenen Testsuite stammende Frage | 1B Foundation | 2B Foundation | 1B-basiertes SFT |
|---|---|---|---|
| Hallo! Antworte kurz auf Deutsch. | „Hallo! Antworte kurz auf Deutsch.“ | „Hallo! Antworte kurz auf Deutsch.“ | „Ich habe mich in der Schweiz sehr wohl gefühlt.“ |
| Was ist zwei plus zwei? | „Was ist zwei plus zwei?“ | „Was ist zwei plus zwei?“ | „Das ist zwei plus zwei.“ |
| Nenne eine Farbe. | „Nenne eine Farbe.“ | „Nenne eine Farbe.“ | „Die Farbe ist die Farbe, die wir uns vorstellen.“ |

Bei allen drei Modellständen löste der native Greedy-Decoder in allen drei Mini-Ausgaben innerhalb von 16 Tokens das EOS-Token aus. **Nur drei Prompts, keine valide Capability-Prozentzahl.** 1B und 2B geben die Fragen praktisch nur wieder; das historische SFT liefert teils flüssigere, aber immer noch sachlich unzureichende Antworten. Das neue Modell ist durch seine 2B-Tokens **nicht automatisch nachweislich intelligenter geworden**.

## Game-kompatible VRAM-Messungen

- Windows-NVIDIA-Preflight: um 10.232–10.254 MiB frei von 12.282 MiB VRAM; Roblox-Prozesse erkannt.
- 1B- und 2B-Read-only-Inferenz: `peak_cuda_allocated_mib = 681`, `peak_cuda_reserved_mib = 794`. Historischer SFT-Kandidat: 681 MiB allocated / 796 MiB reserved.
- Diese Zahlen sind PyTorch-Peaks, **nicht** der gesamte CUDA-Treiber-/Windows-/Roblox-Verbrauch.
- Hardware-Killswitch: mindestens 6.144 MiB frei in Windows-Preflight **und** nach CUDA-Initialisierung, bei unter 3.072 MiB neuem freien Speicher im Generierungslauf Stop; keine trainierenden Schritte.
- Der lokale, begrenzt tiefe Kandidaten-Inventur-Lauf fand **22 große Modell-/Checkpoint-Dateien**, aber deren Namen sind keine Architekturbestätigung. Nur die drei oben aufgelisteten Dateien wurden als echte 350M-Modellmetadaten verifiziert.

## Neue, für Training vorbereitete Implementierung (nicht auf der GPU trainiert)

- `frontier350/train_precision350.py` und die entsprechenden Änderungen in `train_chat350.py` bieten **opt-in `--precision-mode fp32-master`**: FP32-Mastergewichte und AdamW-Zustand; BF16-CUDA-Autocast im Forward/Backward. `--precision-mode legacy-bf16` bleibt der bisherige Standard.
- Ausgewählte Inferenz-Checkpoints dürfen im neuen Modus nach BF16 exportiert werden; der **Optimizer-Resume-Checkpoint behält FP32-Mastergewichte**. So wird nicht versehentlich ein FP32-Trainingszustand in BF16 zerstört.
- Die neue Policy ist auf CPU mit Mockmodellen regressiongetestet; **kein echter RTX4070-FP32-Trainingsschritt, keine echte FP32-VRAM-Messung und keine Modellqualitätsverbesserung dadurch nachgewiesen**.

## Nächste wissenschaftlich sinnvolle Gates

1. Basis- und SFT-Pilotfälle *außerhalb der versiegelten finalen Evaluationssuite* analysieren: Template-Repetition, bedingte Inhalts-NLL vs. Format-NLL, EOS und Trunkierung.
2. FP32-Autocast/Norms an der echten Architektur zunächst als kurzer, eng begrenzter GPU-Diagnoselauf testen, **nicht** bei VRAM-Druck durch Roblox, mit separaten frischen Ausgabeordnern.
3. Erst dann den **2B**-Checkpoint als unveränderliche neue Foundation für einen kleinen SFT-A/B-Pilot wählen. Nicht SFT 1B mit 2B Foundation als gleiches Trainingsergebnis verkaufen.
4. Nach Auswahl auf Entwicklungsfällen einmalige Auswertung der eingefrorenen 30er-Suite mit **identischem Evaluationsvertrag und Suite-SHA**. Keine automatische Promotion bei fehlenden Ergebnissen oder regressiven Kategorien.

**Vergleich mit externen Modellen:** `HuggingFaceTB/SmolLM2-360M-Instruct` ist ein externes öffentliches Forschungs-/Benchmark-Beispiel (361,8M Parameter, Apache-2.0), keine Quelle von Gewichten oder Trainingsdaten für das native Arctus-Modell.

**Kosten:** 0 € neue Cloud-GPU-Jobs; keine HF-Compute-Jobs aktiv. Stromkosten nicht erfasst. Das gesamte verbleibende Projektbudget ist weiterhin maximal 25 € (Strom ausgenommen). **Release-/Merge-Status: Draft, manuell.**
