# NextGen V1 — isolierter Neral-350M-SFT-Pilot (NICHT ausgeführt)

**Aktualisiert am 18.09.2026.** Experimenteller Code im Branch `experiment/phoenix-sft-fp32-audit`, nicht gemergt, keine Modellgewichte geändert, kein SFT-Checkpoint, keine belegte Qualitätsverbesserung, keine 60–70k akzeptierten Trainingstokens/s. **Kein automatischer GitHub-Trainingsstart.**

## Warum ein SFT-Pilot noch nicht gestartet wird

Der SHA-verifizierte eigene Pretraining-Checkpoint umfasst **2.073.161.216 Tokens**, nicht die geplanten 10 Milliarden. Das Fundament-Corpus besteht aus Dokumenten mit BOS/EOS, nicht aus systematisch serialisierten `User: ...\nAssistant:`-Konversationen. Der erste echte Test zeigte 0/3 streng einstellige Antworten, sowohl raw als auch guarded. Der anschließende **24-Item-Entwicklungstest** fand bei vorgegebenen Ziffern lediglich 4/24 korrekte Top-1-Präferenzen im Instruktions-Chatformat und 0/8 beim Rechnen als Gleichung. Das sind öffentlich dokumentierte Diagnosen, **kein unabhängiger Benchmark, keine frei generierte Chatqualität**. Siehe [`QUALITY_FINDINGS_2026_09_18_V2.md`](QUALITY_FINDINGS_2026_09_18_V2.md) und [eigener Gewichts-Lauf](https://github.com/Cola968/neral-helios/actions/runs/35343702334).

**Quelle:** Das vorhandene OASST2-Archiv wurde unverändert, SHA-gepinnt und lokal geprüft (135.174 Nachrichten). Das separat gescreente Kandidateninventar umfasst 1.200 Paare aus verschiedenen Bäumen, nominell 1.094 Train / 106 Validation. Ein wiederholter normalisierter Antworttext und mehrere Heuristik-Warnungen erfordern Nachprüfung. Keine automatische Prüfung beweist Antwortkorrektheit, Rechtmäßigkeit oder umfassende PII-Freiheit. Die 24 Originaldialoge der Offline-Rezensionsprobe liegen privat unter `D:\NeralPrivateAssets\phoenix\screened_review_queue_v1.jsonl`; das Bedienformular `screened_review_queue_v2.html` befindet sich im gleichen lokalen Verzeichnis. Menschen haben die Prüfentscheidungen noch nicht abgeschlossen.

Der frühere Holdout ist verloren. Ein **neuer, eigenständig formulierter und vor allen Experimenten privat eingefrorener** 24–100-Fragen-Holdout v2 mit DE/EN, verifizierten Lösungen und Rubriken fehlt weiterhin. Die öffentlichen Diagnosefragen sind kein Ersatz und dürfen nicht als unabhängiges Evaluationsset ausgegeben werden.

## Obligatorische Reihenfolge und Freigaben

1. V2-Holdout mit `phoenix_freeze_holdout_v2.py` unabhängig erstellen, über `phoenix_holdout_rubric_check.py` validieren, Hashes und Receipt privat bewahren. Keine alten NIS2-Werte als vergleichbar ausgeben.
2. Originaldaten auf Quelle, Nutzungsrechte, Datenschutz, falsche/unsichere Antworten und exakte **sowie semantische** Überschneidungen prüfen. Die vorhandene Offline-UI und der automatische 24er-Test ersetzen keine echte menschliche Bewertung der ausgewählten Daten.
3. Separates Hash-gebundenes Manifest und unabhängigen Screening-Report erst **nach tatsächlicher Prüfung** freigeben. Weder `approved:true` noch Reviewer-Entscheidungen erfinden. Der Screen muss null ungelöste Exact-/Near-Overlap-/Privacy-Hits haben. Train/Validation/Holdout dürfen sich nicht überschneiden.
4. Nur dann lokal einen neuen SFT-Pack erstellen. Verifiziertes originales 16K-BPE, BOS/EOS, Antwort-only-Loss, kausale Labelverschiebung und Hashes sind Pflicht. `--out` muss ein **neues** Verzeichnis sein. Keine Originaldaten, Gewichte, Holdout-Fragen oder private Reports als GitHub-Artefakte hochladen.
5. Vor einem ausdrücklich genehmigten kleinen Pilot stabile GPU-Leerlaufwerte und Speicher prüfen; danach mit identischem, unabhängigem Holdout die ursprünglichen Gewichte und einen isolierten Kandidaten vergleichen. Validation-Loss oder Formatkosmetik zählen nicht als Chat-Qualitätsbeleg.

## Vorbereiteter Pack-Befehl — ausschließlich nach vollständiger Freigabe

Vom Repository-Stamm unter der vorhandenen WSL-Python-Umgebung (Windows `python.exe` wurde auf dem Runner nicht zuverlässig gefunden):

```bash
~/.venvs/neral-triton-py312/bin/python frontier350/nextgen_v1/sft_contract.py \
  --manifest PATH_TO_APPROVED_MANIFEST.json \
  --train PATH_TO_APPROVED_TRAIN.jsonl \
  --validation PATH_TO_APPROVED_VALIDATION.jsonl \
  --tokenizer PATH_TO_PINNED_TOKENIZER.json \
  --holdout-hashes PATH_TO_FROZEN_PRIVATE_HOLDOUT_HASHES.json \
  --screening PATH_TO_APPROVED_SCREEN.json \
  --out NEW_PRIVATE_PACK_DIRECTORY --seq 512
```

Der Pack verwirft ungültige Daten, überschreibt keine bestehende Ausgabe und darf niemals mit Dummy-Hashes oder einem frei erfundenen Freigabereport erstellt werden.

## Korrigierter GPU-Pilot-Befehl — NICHT jetzt ausführen

Der frühere Beispielbefehl ließ drei **erforderliche** Nachweisargumente aus. Die hier dokumentierte Version enthält `--source-manifest`, `--screening-report` und `--holdout-hashes`. Der zusätzliche Entry-Point `train_sft_guarded_entry.py` erzwingt drei getrennte bestandene Leerlaufmessungen, **bevor** der bisherige Trainer Torch/CUDA initialisieren darf. Der alte `train_sft_pilot.py`-Direktaufruf besitzt diese zusätzliche dreifache Vorprüfung nicht und ist daher nicht der empfohlene Startpfad.

```bash
~/.venvs/neral-triton-py312/bin/python frontier350/nextgen_v1/train_sft_guarded_entry.py \
  --execute --trusted-local-checkpoint \
  --checkpoint PATH_TO_TRUSTED_IMMUTABLE_ORIGINAL_CHECKPOINT.pt \
  --checkpoint-sha256 EXACT_64_HEX_ORIGINAL_SHA256 \
  --pack LOCAL_APPROVED_PACK_DIRECTORY \
  --tokenizer PATH_TO_PINNED_TOKENIZER.json \
  --source-manifest PATH_TO_APPROVED_MANIFEST.json \
  --screening-report PATH_TO_APPROVED_SCREEN.json \
  --holdout-hashes PATH_TO_FROZEN_PRIVATE_HOLDOUT_HASHES.json \
  --out NEW_ISOLATED_SFT_RESULT_DIRECTORY \
  --train-blocks 4 --batch 2 --steps 20
```

**GPU-Leerlaufprüfung (nur neuer Entry-Point):** drei Messungen mit je 2 Sekunden Abstand; **jede** muss GPU-Auslastung maximal 15 %, belegten VRAM maximal 2.500 MiB und Temperatur maximal 78 °C erfüllen. Bei fehlenden Werten, ausgelasteter oder heißer GPU: Abbruch ohne Trainingsstart. Der ursprüngliche Trainer prüft zusätzlich GPU-Typ RTX 4070, BF16, mindestens 9.000 MiB freien VRAM und Starttemperatur bis 78 °C. Während des maximal 50-Schritt-Pilots sind 82 °C sowie 9 GiB reservierter VRAM die Abbruchgrenzen. Diese Schutzmaßnahmen garantieren keine Hardware-Sicherheit oder gleichzeitige Ausführungsfreiheit; andere GPU-Aufgaben vorher beenden und kein anderes Training starten.

Die bestehende Modellarchitektur muss exakt 349.825.152 Parameter umfassen; der Source-Checkpoint und der Tokenizer werden hashgebunden. `torch.load(weights_only=False)` wird im alten Trainer **nur nach expliziter Vertrauensbestätigung und SHA-Check** verwendet. Der neue Entry-Point allein ersetzt diese Vertrauensprüfung nicht.

## Validierung und Nachweisgrenzen

Die bounded SFT-Funktion prüft bis zu 16 deterministisch über die Validierung verteilte Beispiele vor/nach Training und gewichtet Cross-Entropy nach tatsächlich supervisierten Antworttokens. Nur ein verbesserter Validierungs-Loss darf einen **isolierten, nicht produktionsberechtigten** Kandidaten erzeugen. Ein echtes Qualitätsurteil verlangt einen unabhängigen eingefrorenen Holdout, freie Antworten, Blindbewertung und identische Decoder-/Prompt-Einstellungen. Jeder Pilot bleibt ausdrücklich `production_eligible:false`.

[24er-Eigengewichtsdiagnose](https://github.com/Cola968/neral-helios/actions/runs/35343702334) und [CI nach Korrektur der Splitprüfung](https://github.com/Cola968/neral-helios/actions/runs/35344126991) sind abgeschlossen. Die CI-Tests verwenden synthetische Daten, nicht echte SFT-GPU-Berechnungen. **Status: Datenfreigabe ausstehend, unabhängiger Holdout ausstehend, SFT nicht gestartet, Qualitätsgewinn nicht nachgewiesen.**
