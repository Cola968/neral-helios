# Arctus350 – opt-in kompakte SFT-Batches (29.10.2026)

## Ziel und aktueller Befund

Im bestehenden Trainingsskript `frontier350/train_chat350.py` gab es vor diesem PR bereits `bucketed` und `human-balanced`-Sampling sowie Assistant-only-Cross-Entropy, Replay, EOS-Gates, Metadaten und Checkpoint-Schutz. Es wäre irreführend, diese Fähigkeiten als neu zu verkaufen.

Der Standard-`bucketed`-Sampler sortiert Beispiele nach Tokenlänge, verteilt sie in Buckets mit standardmäßig 64 Beispielen und mischt **innerhalb des gesamten Buckets**, bevor er Minibatches mit standardmäßig 4 Beispielen bildet. Bei breiter Längenverteilung führt das zu Padding innerhalb der Minibatches: Das Modell berechnet auch die aufgefüllten Positionen, obwohl die Ziele `-100` und damit loss-maskiert sind.

**Neu:** `--sampling-mode compact` sortiert Beispiele nach Länge, bildet **zuerst** möglichst gleich lange Minibatches und mischt **dann die Batch-Reihenfolge** (und innerhalb jedes Batches die Beispielreihenfolge). Ein unvollständiger Epochenschluss wird beim nächsten Batch mit Beispielen der folgenden Epoche vervollständigt. Es gehen keine Zeilen verloren. Der ursprüngliche Standard `bucketed` und die human-balanced Option bleiben unverändert.

## CPU-Test: echter Beleg, eng begrenzte Aussage

- TDD RED: [Run 40784723938](https://github.com/Cola968/neral-helios/actions/runs/40784723938) – die neuen Funktionen fehlten und der Test schlug fehl.
- GREEN: [Run 40784828487](https://github.com/Cola968/neral-helios/actions/runs/40784828487) – **11/11** bestehende und neue CPU-Batching-Tests bestanden.
- [Syntax](https://github.com/Cola968/neral-helios/actions/runs/40784828393) und [Release-Evidence-Integrity](https://github.com/Cola968/neral-helios/actions/runs/40784828736) ebenfalls erfolgreich.
- Der Regressionssatz prüft exakt einmalige Samples je Epoche, reproduzierbare RNG, unveränderte Nutz-Tokens, gleich große Batches, ungerade Datensatzgrößen, ungültige Budgets und weniger Padding auf **synthetischen** verschieden langen Beispielen.
- Im synthetischen Benchmark wird **mindestens 65 % des Padding-Overheads** gegenüber dem bestehenden 64er-Bucket-Sampler eingespart. Der konkrete Effekt hängt von der Datenlängenverteilung ab. Das ist **KEIN** GPU-Speedup-Prozentsatz, kein geringerer Trainingsverlust und keine bessere Rohmodellqualität. Attention-Kosten sind sogar nicht einfach linear in Tokenanzahl.

## Bevor der lokale PC wieder verfügbar ist

- `compact` ist nur eine zusätzliche Option; nichts startet automatisch. Keine GPU-/HF-Jobs und keine privaten Modellgewichte werden benutzt.
- Zuerst echte lokale Längenhistogramme und Padding-Telemetrie **ohne Rohdatenexport** vergleichen.
- Danach – nur bei ausdrücklicher Freigabe – ein kurzes, gleiches GPU-A/B-Experiment mit identischer Quelle, gleichem Seed, gleichen Beispielen und Budget: `bucketed` vs. `compact`. Protokolliere Tokens/sec, Peak-VRAM, Padding-Quote, effektive Steps/sec, EOS, Replay/NLL, **eigene Rohmodell-Antworten** auf eingefrorenen Holdouts.
- Längenharmonisierte Batches könnten die Reihenfolge der Beispiele und damit den Optimierungsverlauf ändern. Auch bei weniger Padding ist ein Qualitätssieg **nicht** garantiert. Beim Ausbleiben eines Vorteils nicht übernehmen.
- **Wichtig:** Das ist eine Trainings-Effizienzoptimierung und **keine Methode, 350M automatisch auf GPT-3 oder GPT-5 zu heben**. Die getrennte gepaarte Modellevidenz wird bereits in [PR #180](https://github.com/Cola968/neral-helios/pull/180) entwickelt. Keine Überschneidung/Score-Erfindung.

## Test- und Budgetgrenzen

`python -m unittest tests.test_sft_batching350 tests.test_arctus350_compact_sft_batches -v`

Zusätzliche direkte Kosten: **0 €** (innerhalb regulärer GitHub-Actions-Nutzung); absolute Gesamtausgabengrenze für das Projekt 25 € ohne Haushaltsstrom, weitere Ausgaben nur nach neuer Freigabe. Kein automatischer Merge/Release, keine lokalen Trainingsjobs, kein Upload von Gewichten.
