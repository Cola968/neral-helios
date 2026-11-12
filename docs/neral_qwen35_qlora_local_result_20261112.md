# Qwen3.5-9B QLoRA: tatsächliches lokales Ergebnis vom 12.11.2026

Eine vollständige Epoche wurde auf der lokalen RTX 4070 abgeschlossen. Der Adapter wurde nach GGUF exportiert, lokal registriert und mit dem unveränderten 24er-Holdout geprüft. Gegenüber der identisch exportierten untrainierten Kontrolle ist das Controller-Ergebnis **verbessert (+6/24)**.

| Bereich | Bekannte Baseline¹ | Untrainierte Kontrolle² | NERAL QLoRA v1² |
|---|---:|---:|---:|
| Boolesche Logik | 2/12 | 1/12 | 7/12 |
| Ordnung | 12/12 | 12/12 | 12/12 |
| Gesamt | 14/24 (58,3 %) | 13/24 (54,2 %) | 19/24 (79,2 %) |

¹ Vom Nutzer vorgegebener früherer Ollama-Baselinewert; in diesem Lauf nicht erneut mit dem alten Tag gemessen. ² Gleicher offizieller HF-Checkpoint, gleicher Konverter, Q4_K_M, Ollama-Runtime, Prompt, Schema, Decoder-Einstellungen und Solver-/Reparaturcontroller. Nur die trainierte Variante enthält den LoRA-Delta. Die Kontroll-GGUF ist neu und ersetzt das alte Basismodell nicht.

## Modellbeitrag und Werkzeuge getrennt

Diese Aufgaben bewerten strukturierte Extraktion mit lokalem exaktem Solver. Die Zahlen sind **kein werkzeugfreier Modell-Rechentest**. Auch „Erstextraktion“ enthält die nachgelagerte Solverprüfung.

| Messgröße (24 Aufgaben) | Kontrolle | NERAL |
|---|---:|---:|
| Korrekte Erstextraktion + Solver | 13 | 19 |
| Erstextraktionen schema-akzeptiert | 18 | 19 |
| Controller-Endantwort korrekt | 13 | 19 |
| Davon formale Solverantwort korrekt | 13 | 19 |
| Reparatur versucht | 5 | 5 |
| Reparatur korrekt | 0 | 0 |

Der eingefrorene Holdout ist kuratiert, sichtbar und nicht verblindet. 24 Fälle erlauben keine Aussage über allgemeine NERAL-Fähigkeiten oder eine garantierte fehlerfreie Übersetzung. Es wurde nach Betrachtung der Holdout-Ergebnisse nicht nachtrainiert.

Im gepaarten Vergleich verbessern sich 6 Fälle; 0 zuvor richtige Fälle werden falsch. Die fünf verbleibenden NERAL-Fehler sind Boolesche Aufgaben mit `solver_type_error`; alle fünf Reparaturen bleiben erfolglos. Die neu exportierte Kontrolle weicht um einen Treffer von der historischen Baseline ab; deshalb wird der Trainingseffekt primär gegenüber dieser aktuellen Kontrolle ausgewiesen.

## Training und Speicher

- Modell: `Qwen/Qwen3.5-9B`, Revision `c202236235762e1c871ad0ccb60c8ee5ba337b9a`. Alle vier Original-Shards SHA-256-verifiziert; keine Originalgewichte geändert.
- Windows, lokale RTX 4070 (12 GB); keine Cloud-GPU. Isolierter Venv `D:\neral-qlora\venv`, keine globalen Site-Packages. PyTorch 2.8.0+cu128, Transformers 5.22.2; komplette Paketliste in `environment-lock.txt`.
- 720 Trainingszeilen: 540 Boolean, 180 Ordnung; 360 unterschiedliche Quellaufgaben mit je Erstversuch und Reparatur. Alle Zielantworten mit Orakel und Quellprüfung validiert. Keine Holdout-Fragen oder identischen Holdout-Schemas im Training. Trainingshash und Holdouthash in `data-audit.json`.
- Maximal 1.458 Tokens, Kontextlimit 2.048, keine Kürzung, ausschließlich Antworttoken im Loss.
- NF4, Double Quantization, BF16, Rang 8/Alpha 16, all-linear, 21,639,168 trainierbare Parameter, Batch 1, Akkumulation 8, Seed 2026111201.
- Erfolgreicher Smoke `smoke04`: exakt ein Optimierungsschritt, Lernrate 5e-5, echte Gewichtsänderung, 21,335 s reine Trainingszeit. Volltraining startet anschließend frisch vom selben Basismodell.
- Volltraining: **90 Optimierungsschritte / eine vollständige Epoche**, **1998.328 s (33.31 min)** reine Trainingszeit. Gesamter Volltrainingsprozess inklusive Laden/Speichern: 2038.041 s.
- PyTorch-Spitzenallokation: **8.593 GiB**; Spitzenreservierung: 8.986 GiB.
- Gesamte GPU-Belegung, ungefähr sekündlich per `nvidia-smi` gemessen: **10331 MiB (10.089 GiB)**, einschließlich Windows/Hintergrundbelegung. Kurze Treiber-Spitzen können zwischen Stichproben liegen. Der 9-GiB-PyTorch-Allocatorschalter begrenzt nicht sämtliche Treiberallokationen.
- Native Windows-Referenzpfade für Gated DeltaNet/causal-conv1d; keine ungemessene Beschleunigung behauptet. Vollständige manifeste Schrittzahl, Loss, Gewichtsänderung und Telemetrie in `full.json`.

## Behobene technische Fehler

1. Transformers-5-`BatchEncoding` beim Chat-Template wurde zuvor fälschlich abgelehnt. Tokenstrukturprüfung, Präfixabgleich und Assistant-Maskierung bleiben erhalten.
2. Entferntes `warmup_ratio` durch kompatible ganzzahlige `warmup_steps` ersetzt. Der Ein-Schritt-Smoke hat keine Null-Lernraten-Aufwärmphase.
3. PEFTs FP32-Upcast der zwei großen eingefrorenen Matrizen führte zum OOM. Diese werden bei der Vorbereitung kurz auf der CPU gehalten und in ihrer ursprünglichen BF16-Präzision auf die GPU zurückgebracht; Normen bleiben FP32. Sämtliches Training erfolgt auf CUDA.
4. Logits für maskierte Promptpositionen verursachten unnötigen OOM. Native `logits_to_keep` berechnet nur Positionen, die Zieltoken vorhersagen. Vollständiger Kontext, korrekte kausale Verschiebung und Akkumulationsnormierung bleiben erhalten; Loss- und Gradientenäquivalenz sind regressiongetestet.
5. Fehlendes HF-`datasets` kollidierte mit dem gleichnamigen lokalen Namespace-Ordner. Die echte Abhängigkeit ist im isolierten Venv installiert und gelockt.

Die ersten drei Versuche scheiterten vor einem abgeschlossenen Optimierungsschritt; die lokalen Tracebacks und Prozessmessungen bleiben unter `D:\neral-qlora`. 24 betroffene Unit-/Regressionstests bestanden. Die CPU-CI prüft diese Verträge ohne GPU-Paketinstallation.

## Export, Registrierung und Reproduktion

Die quantisierten GGUFs enthalten jeweils 442 Tensoren: 248 unterscheiden sich, 194 sind byteidentisch. Eingabe-Embedding und Ausgabekopf sind unverändert; der trainierte Delta ist somit auch im quantisierten Export nachgewiesen. Details stehen in `gguf-tensor-diff.json`.

Der Adapter wurde auf der CPU shardweise mit dem ursprünglichen BF16-Checkpoint zusammengeführt, mit llama.cpp `a8575a3ef4e2631d0540b2879fff86ad8411dbf6` konvertiert und mit Windows-Build `b12110` Q4_K_M-quantisiert. Die ursprünglichen Dateien und bestehenden Ollama-Modelle wurden nicht überschrieben. Ein echter GGUF-Laufzeittest und ein lokaler Ollama-Decodiertest waren erfolgreich. Die Registrierung enthält den Textdecoder; ein Vision-Projektor wurde nicht exportiert.

- Trainiertes Modell: `neral-qwen35-qlora-v1:latest`
- Passende untrainierte Kontrolle: `qwen35-9b-qlora-control-20261112:latest`
- Adapter: `D:\neral-qlora\full01\adapter`
- Trainierte GGUF: `D:\neral-qlora\posttrain01\neral-qwen35-qlora-v1-Q4_K_M.gguf`
- Hashes und Converterversionen: `artifacts.json`; reale Laufzeiten der Exportphasen: `export-stages.json`.
- Holdout-Protokoll: `scripts/neral_qlora_holdout_live.py`, 24 identische Fälle, Temperatur 0, First-Turn-Limit 384, Reparaturlimit 1.536, Kontext 4.096, Thinking aus, ursprüngliche Seeds. Pro-Fall-Skalare und Modell-Digests sind in beiden Holdout-JSON-Dateien erhalten.

```powershell
# Reproduktion in einem frischen Ausgabeverzeichnis; bestehende Runs werden abgelehnt.
D:\neral-qlora\venv\Scripts\python.exe -m scripts.neral_qlora_local_run smoke --label smoke-new
D:\neral-qlora\venv\Scripts\python.exe -m scripts.neral_qlora_local_run full --label full-new --smoke D:\neral-qlora\smoke-new
# Export-CLI erwartet die vollständigen 90 Schritte und neue Ausgabe-/Modellnamen.
```

Entwicklungsbranch: `codex/neral-qwen35-gpu-validation-20261112`, Ausgangscommit `7c2c0909`. Der Implementierungscommit ist `4df8e4d1`; Messergebnisse sind in einem separaten Folgecommit gesichert. Es wurde kein Produktionsdefault umgestellt.
