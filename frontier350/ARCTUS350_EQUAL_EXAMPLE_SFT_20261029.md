# Arctus350 – Experiment „Equal-Example Assistant Loss“ (29.10.2026)

## Hintergrund und Ziel

Der bestehende native Trainer `frontier350/train_chat350.py` nutzt bereits Assistant-only-Cross-Entropy, EOS-Gewichtung, Replay, stratifizierte/kompakte Batches und geschützte Checkpoint-Auswahl. Der Trainingsverlust wird im bisherigen Standard über **sämtliche supervisierten Antwort-Tokens** des Minibatches gewichtet gemittelt. Das ist korrekt, kann aber beispielsweise dazu führen, dass eine sehr lange Erklärung den Gradient stärker beeinflusst als eine kurze, ebenfalls wichtige präzise Antwort. Diese relative Gewichtung ist eine Optimierungsentscheidung, kein Rechenfehler.

**Neue, explizit experimentelle Option:** `--loss-aggregation per-example`. Der neue native `frontier350/sft_loss350.py`-Reducer berechnet einen EOS-gewichteten mittleren Tokenverlust **pro kodiertem Chatbeispiel**, mittelt anschließend gleich über Beispiele und leitet echte Gradienten mit PyTorch-Autograd ab. Die schon vorhandene EOS-Gewichtung wird **innerhalb** jedes Beispiels erhalten.

Formel (für Beispiel `i`, assistierte Tokenpositionen `t`, Gewichte `w`):
```text
Legacy token:
  L = sum_i sum_t w_it * CE_it / sum_i sum_t w_it

Experimental per-example:
  L = mean_i [sum_t w_it * CE_it / sum_t w_it]
```

Beispiel: Zwei gleich zuverlässige Antworten mit 100 bzw. 5 supervisierten Tokens bekommen bei `token` näherungsweise Gewichte 20:1, bei `per-example` dagegen 1:1 (jeweils unter Berücksichtigung der EOS-Gewichte).

**Erwartete mögliche Wirkung:** Besseres Erlernen kurzer Antworten und weniger Dominanz langer Beispiele. **Möglicher negativer Effekt:** Weniger Gewicht für informationsreiche längere Erklärungen und schlechtere Mehrschritt-Dialogqualität. Ein echter Vorteil ist vor Training und vollständiger unabhängiger Evaluation **unbekannt**.

## Sicherheit / Rückwärtskompatibilität

- Die bestehende Einstellung `--loss-aggregation token` ist **Default** und mathematisch identisch mit dem bisherigen Reducer; ohne opt-in wird kein Trainingsziel verändert.
- Bei `per-example` führen Beispiele mit null Assistant-Supervision zum expliziten Abbruch statt zu einer versteckten Division durch Null. Unzulässige Tensorformen, Klassen-IDs und EOS-Gewichte werden abgewiesen.
- Original Arctus350: **349.825.152 Parameter**, unveränderte `frontier350-v1`-Architektur, unveränderter gesperrter 16K-BPE, keine Änderung an Checkpoint, Zustandsschlüssel oder Modellsignaturen.
- Trainer protokolliert die Loss-Wahl in `run.json`, `metrics.jsonl` und `summary.json`. Selektionskriterien, EOS-Gate, Replay-Gate und Test-Holdouts bleiben erhalten.
- Keine Verbindung zu externen fremden Gewichten: SmolLM2-360M ist **nur ein HF-Größenvergleich**, kein Arctus-Checkpoint und kein Teacher-Modell.

## Echte Tests – nur kleine synthetische CPU-Tensoren

- **RED**: [GitHub Actions 40865321956](https://github.com/Cola968/neral-helios/actions/runs/40865321956) – erwarteter Importfehler, neues Trainingsmodul existierte noch nicht.
- **GREEN**: [GitHub Actions 40865527414](https://github.com/Cola968/neral-helios/actions/runs/40865527414) – 5/5 mathematische und Autograd-Tests bestanden. Unter anderem exakt gleicher Legacy-Tensorverlust, anderer pro-Beispiel-Verlust auf ungleich langen Antworten, echte endliche Gradienten, maskierte Tokens, ungültige Formen und Gewichte.
- Die vorliegenden **Tests benutzen keine echten Arctus-Gewichte, keine privaten Trainingsdaten und keinerlei Modellinferenz**. Sie beweisen die Implementierung der Loss-Funktion, **nicht** höhere Modellqualität.

## Echte A/B-Evaluation nach gesonderter GPU-Freigabe

1. Den **gleichen** lokal SHA-256-gesicherten 2B-Foundation-Checkpoint laden, Original-16K-BPE und unveränderte originale NeralTalk-Splits verwenden; zuerst Label-/Split-/Supervision-Audits aus den gestapelten Forschungs-PRs ausführen.
2. `token` gegen `per-example` unter *gleichem Seed, gleicher Datenreihenfolge, Sampling-Modus, effektiven Tokens, Schritten, Replay, LR und Präzision* vergleichen. Unterschied nur `loss-aggregation`; beide Ausgaben in frischen Ordnern.
3. Beispiel nach **ausdrücklicher** späterer GPU-Freigabe (nur Illustration, **NICHT ausführen ohne Entscheidung**):
   ```text
   python frontier350/train_chat350.py --init-from <local-2b.pt> --tokenizer <locked-tokenizer.json> --data <private-neraltalk> --replay-data <local-shards> --out <fresh-control-dir> --steps 64 --loss-aggregation token
   python frontier350/train_chat350.py --init-from <local-2b.pt> --tokenizer <locked-tokenizer.json> --data <private-neraltalk> --replay-data <local-shards> --out <fresh-experiment-dir> --steps 64 --loss-aggregation per-example
   ```
4. Beste Kandidaten **gepaart** gegen eingefrorene DEV-Aufgaben prüfen: rohe native Antworten, korrekte kurze/mehrschrittige Instruktionen, deutsche/englische Dialoge, Mathematik, EOS, Wiederholung, Val-NLL und Foundation-Replay. Die getesteten Checkpoints niemals allein auf Basis niedrigerer Train-Loss auswählen.
5. Checkpoint-Vergleich als `NOT_MEASURED`, `INSUFFICIENT_STATISTICAL_EVIDENCE` oder Review-Kandidat belegen. Vor jeder Promotion erneut separat freigeben. **TEST**-Split unangetastet bis zum finalen Kandidatenentscheid.

## Verbindliche Grenzen

Keine Cloud-GPU/Hugging Face Jobs, keine neuen Ausgaben, keine lokalen 4070-Jobs bei ausgeschaltetem PC, keine Checkpoint-/Daten-Uploads, kein Merge/Release. Dieser Draft-PR ist auf der bestehenden Arctus-Forschungskette PR #190 gestapelt, damit keine Arbeit parallel überschrieben wird.

**Leistungsbewertung des echten Modells: NOT_MEASURED. GPU-Gewinn: NOT_MEASURED. GPT-3-/Giotto-Äquivalenz: NICHT nachgewiesen.** Ziel: einen konkreten kontrollierten Forschungsschritt vorbereiten, der die eigene 350M-Modelllinie beim nächsten freigegebenen Training möglicherweise messbar verbessert.
