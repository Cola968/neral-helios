# NERAL v6.1 – gezieltes Replay-Training gegen Vergessen

Stand 07.11.2026. Der experimentelle v6-Adapter ist **nicht für den Release
freigegeben**: 21/72 gegenüber v5 24/72 und Qwen+Few-Shot 29/72 auf einem
verbrauchten synthetischen Entwicklungssatz. v6 erzeugte 19 statt 7
vollständige, nicht eindeutig extrahierbare Endantworten.

Der anschließende CPU-Root-Cause-Audit ergab: alle drei v5→v6-Regressionsfälle
sind **unentscheidbare Antworttexte**, kein eindeutig falscher gelesener
Zahlenwert; das kann fehlende Mathematikfähigkeiten **nicht** ausschließen.

## Experimentelle Änderung (eine kontrollierte Intervention)

Statt ein erneut auf dem schlechteren v6 trainiertes Modell zu starten,
lädt der Workflow **den unveränderten SHA-geprüften v5-Adapter** als
Startpunkt und trainiert einen **separaten** v6.1-Adapter. Die Quellen
sind ausschließlich ursprüngliche, deterministisch rekonstruierte
TRAIN-Daten:

- **560 v5-Trainingsbeispiele** aus acht bisherigen Aufgabenfamilien,
  70 pro Familie. Damit werden Format, bestehende Mathematik und
  **Python-Codeaufgaben** explizit wiederholt (70 % der Mischung).
- **240 v6-Trainingsbeispiele** aus sechs Familien,
  40 pro Familie, einschließlich 40 überprüfter
  kontrafaktischer Mehrschrittrechnungen (30 %).
- **140 getrennte Validation-Aufgaben**, darunter 100 v6-
  Validierungsfälle und 40 v5-Validierungsfälle.
- Keine EXT-400, keine verbrauchten dev72-Fälle, keine v5/v6-Tests,
  keine neuen Release-Aufgaben im Training.
- Vor dem Training exakte und 10-Gramm-Dekontamination gegen
  den bereits verbrauchten 72er-Entwicklungstest. Nicht überprüfte
  externe Pretraining-Überschneidungen bleiben möglich.

### Was sich gegenüber dem schädlichen v6-Lauf ändert

| Parameter | v6 | v6.1-Kandidat |
|---|---|---|
| Ausgangsadapter | v5 Rank 16 | **v5 Rank 16** (nicht v6) |
| Alte v5-Trainingsbeispiele im neuen Mix | 0 | **560 von 800** |
| Neuere v6-Aufgaben | 728 | **240 von 800** |
| Learning Rate | 1e-5 | **2e-6** |
| Optimizer-Schritte | 120 | 120 |
| Kontextmaximum | 768 | 768 |
| Gewichte | v6 experimentell | neuer v6.1-Adapter; v5 bleibt erhalten |

LoRA/ NF4 / BF16 / paged AdamW 8-bit / 12-GiB-RTX4070 werden aus dem
schon erfolgreich getesteten v6-Trainer wiederverwendet. Geändert wurden
nur die explizit angebotenen Parameter für Datenfabrik, Lernrate,
Schrittzahl, Genehmigung und Reportstatus. Die alten v6-Defaults
bleiben gleich. Kein zusätzlicher GPU-/Cloud-Dienst.

## Reproduktion und Sicherheitsbedingungen

CPU:
```powershell
python -m unittest discover -s tests -p test_neral_v61_replay_train.py -v
python -m unittest discover -s tests -p test_neral_v6_r16_gpu.py -v
python -m scripts.neral_v61_replay_train --mode cpu-data --out C:\private\v61-review
```

Der **separate experimentelle Trainingstrigger** ist
`.github/workflows/neral-v61-replay-gpu.yml`,
`ref=main`, mit Eingabefeld
`authorize=TRAIN_NERAL_V61_R16_120_REPLAY_EXPERIMENTAL`.

Es wird nur trainiert, wenn die CPU-Vertragstests grün sind, v5-SHA
übereinstimmt, mindestens 10 GiB VRAM frei sind, und keine parallelen
GPU-Jobs laufen. Das ursprüngliche v5 bleibt unangetastet. Die
LoRA-Gewichte bleiben privat lokal; GitHub bekommt nur den
Trainingsmetrikbericht (Loss/VRAM/Hash/Anzahl).

## Entscheidung nach GPU-Abschluss

**Eine veränderte LoRA und kleiner Loss sind kein Nachweis für mehr
Intelligenz.** Der bereits verbrauchte 72er-Satz darf höchstens als
Regressionsdiagnose dienen; für eine Leistungsbehauptung brauchen wir
einen neuen, vorher festgelegten, familien- und sprachdiversen Vergleich
v5 / v6.1 / Qwen+Few-Shot und unabhängige Code-Testfälle.

Falls v6.1 schlechter abschneidet, **keine Promotion**.
Falls es auf neuen Aufgaben besser abschneidet, bleibt es bis zu einem
unabhängigen größeren Test ein experimenteller Kandidat. Der
Release am 10.11.2026 darf keine behauptete GPT-Level- oder OOD-
Verbesserung enthalten, wenn dafür kein Beleg existiert.
