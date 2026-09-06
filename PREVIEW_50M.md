# Neral-50M: Sprachpilot A

**Status: Pretraining-Experiment, kein fertiger Chatbot.** Der bestehende 20M-Stand bleibt Standard. Kein fremdes Sprachmodell wurde geladen oder zur Laufzeit aufgerufen.

## Modell und Daten
- Exakt **50.012.976 trainierbare Parameter**; acht Blöcke, d_model 624, zwölf Heads mit Dimension 52, SwiGLU 2496, 260 Byte-Token, Kontextlimit 4096.
- Neu und lokal initialisierte eigene Neral-Architektur, Seed 50013. Der schlechtere 40M-Versuch wurde nicht als Ausgangspunkt verwendet.
- 5.000 Wikipedia-Artikel geladen, 4.120 behalten. Training: 3.736 Artikel, 21.915.506 Tokens einschließlich BOS/EOS. Herkunft, Lizenz, Filter und dokumentweise Splits siehe DATA_PROVENANCE.md.
- Separates neues Test- und Holdout-Material blieb unbenutzt. Der Pilot nutzt etwa 4,56 % des vorbereiteten Trainings-Tokenumfangs; zufällige Fenster können sich überlappen, dies ist keine behauptete Zahl einzigartig gelernter Tokens.

## Effizienz tatsächlich gemessen
- Neun Batch-/Sequenzkombinationen vor und nach der Optimierung gemessen. Beste Konfiguration: Sequenz 256, Micro-Batch 32, 8192 Tokens pro Update.
- Bester gemessener Durchsatz: 53,709 → 76,614 Tokens/s (42.6 % mehr). Kurze Messungen mit drei gewerteten Schritten nach Warmup, keine langfristige Leistungszusage.
- Ursache: Flash Attention ist im installierten Windows-PyTorch nicht eingebaut. Der effiziente CUDA-SDPA-Kernel verlangt durch acht teilbare Head-Dimensionen. Parameterloses Null-Padding 52 → 56, Skala weiterhin 1/√52, anschließend Abschneiden der vier Zusatzdimensionen aktiviert den effizienteren Kernel. CPU-FP32-Vergleich identisch auf dem dokumentierten Prüftensor.
- uint16-Memmap, vorberechnete Byte-Token, zusammengepackte Dokumente mit BOS/EOS, 0 Padding-Tokens, BF16, fused AdamW. Dokumentübergreifende Attention ist möglich und nicht isoliert.
- Größte Batch ist nicht automatisch schneller. Der unoptimierte 1024×16-Fall belegte über 14 GB PyTorch-Speicher und fiel auf rund 3.358 Tokens/s; auf Windows trat kein CUDA-OOM auf, Speicherüberschreitung/Paging ist eine plausible Erklärung. Diese Kombination wurde nicht gewählt.
- torch.compile, pinned memory und Gradient Checkpointing wurden nicht als schneller behauptet oder aktiviert; keine Messung dieser optionalen Varianten.

## Ausgeführter Pilot
- Verarbeitete Tokens: **999,424**, 122 Updates. Tatsächliche Trainingsschleife inklusive Evaluation/Schreiben: **18.38 Sekunden**.
- Davon Update-Berechnung: 13.24 Sekunden, 75,507 Tokens/s. Deutsch 532,480, Englisch 466,944.
- Peak allocated 3.34 GiB, reserved 3.60 GiB; GPU: RTX 4070. Dies ist PyTorch-Speicher, kein vollständiger Systemverbrauch.
- Feste Validation-Slices: Loss **5.6234 → 2.3636**. Best-validation-Checkpoint bei Schritt 122. Das ist keine Auswahl als bester Chatbot.
- Pretraining wurde gemäß Tokenbudget beendet. Keine weitere Skalierung und kein Instruction-Tuning nach dem fehlgeschlagenen freien Sprachtest.

## Freie Generierung: wichtiger als der Loss
100 vorher eingefrorene Prompts, gleiche Eingaben, greedy, maximal 64 Byte-Token; rohe Ausgaben und Token-IDs unverändert gespeichert. Dieses kurze Antwortbudget begrenzt die Aussagekraft über mittellange Antworten. Zusätzliche 20 Holdout-Prompts wurden nicht ausgeführt.

| Messung | Neral-20M Preview 02 | 50M Pilot A |
|---|---:|---:|
| Valides UTF-8 | 100.00% | 100.00% |
| EOS-Abschluss | 29.00% | 0.00% |
| Wiederholte Wort-Trigramme | 2.95% | 76.62% |
| Rechenantwort exakt | 0/10 | 0/10 |
| Verständlichkeit, manuelle Stichprobe 0–4 | 1.8 | 1.0 |

Manuelle Stichprobe: erste fünf deutsche und erste fünf englische Chatfragen, nachträgliche Bewertung durch den Codex-Agenten; kein unabhängiges menschliches Gutachten. Sprachkonsistenz wird nur durch eine kleine Stopwort-Heuristik diagnostiziert. Codeantworten wurden gespeichert, aber nicht auf dem Host ausgeführt; kein funktionaler Coding-Score.

**Der 50M-Pilot besteht das Qualitätsgate nicht.** Er produziert insbesondere „der der der …“ und beendet keine der 100 Antworten innerhalb des Budgets mit EOS. Die Verständlichkeit verschlechtert sich gegenüber dem ohnehin schwachen 20M-Stand. Es gibt daher keinen `best_free_generation`-Checkpoint und keine automatische Ablösung des Standards.

## Laufzeitprojektion aus gemessenem Durchsatz
Nur reine Trainings-Token-Verarbeitung, ohne zusätzliche Datenbeschaffung, Evaluation, Checkpoints oder Hardwarekosten:

- 100 Mio. Tokens: etwa 0.36 Stunden.
- 250 Mio. Tokens: etwa 0.91 Stunden.
- 500 Mio. Tokens: etwa 1.81 Stunden.
- 1000 Mio. Tokens: etwa 3.63 Stunden.

Stromverbrauch und Euro-/Tokenkosten sind nicht berechnet: keine integrierte Leistungsmessung und kein Nutzer-Stromtarif vorhanden.

## Veröffentlichung und nächste Schritte
142 lokale Tests bestanden; bestehende Security- und Helios-Features bleiben erhalten. Reproduktions- und Fortsetzungsbefehle stehen in REPRODUCE.md. Die Fortsetzung ist vorbereitet, wurde nach dem fehlgeschlagenen freien Sprachgate aber nicht automatisch gestartet.

Privates Repository: https://github.com/Cola968/neral-helios. Quellcode, Dokumentation, Tests und ausgewählte Messberichte werden übertragen. Rohtrainingsdaten, lokale Projekt-Datenbank, Chats, virtuelle Umgebung und vollständige Optimizer-Checkpoints bleiben lokal. Kompakte eigene Modellgewichte werden gegebenenfalls getrennt als private Release-Dateien bereitgestellt.

Der nächste Entwicklungsbedarf ist eine längere, sorgsam beobachtete allgemeine Sprachlernphase auf breiterem Material und eine belastbarere Generationsbewertung. Ein Nutzer-fähiger 50M-Chatbot wurde mit diesem Pilot nicht erreicht.
