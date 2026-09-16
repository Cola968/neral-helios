# NextGen V1 — isolierter 350M-SFT-Pilot (noch NICHT ausgeführt)

**Stand 16.09.2026.** Experimenteller Code, kein trainiertes SFT-Modell, keine belegte Antwortverbesserung oder 60–70k akzeptierten Trainingstokens/s. Nur im Entwicklungsbranch; eigener SAFE-Checkpoint und bisherige Trainingspipeline bleiben unberührt. Der SFT-Pilot wird **nicht** automatisch über GitHub Actions gestartet oder in Produktion übernommen.

## Verifizierter Datenstand

- Das originale OASST2-Archiv wurde auf dem verbundenen Windows-Rechner wiederhergestellt und anhand des fest gepinnten SHA-256 und exakt 54.301.900 Byte geprüft: [Archiv-Run 35141229336](https://github.com/Cola968/neral-helios/actions/runs/35141229336). Die vorherige `python.exe`-Ausführung auf dem Self-hosted Runner scheiterte wegen fehlendem Python auf `PATH`; der archivierte Download erfolgte anschließend erfolgreich mit nativem PowerShell. **Python-/Torch-Trainingsumgebung auf diesem Runner ist damit NICHT nachgewiesen.**
- Vollständiger, schreibgeschützter Metadatencheck [Run 35142018319](https://github.com/Cola968/neral-helios/actions/runs/35142018319): 135.174 eindeutige IDs, null fehlende Tree-IDs, 3.510 DE- und 37.669 EN-geeignete Eltern-Kind-Paare **vor** Rang-Auswahl/Dublettenentfernung; 1.480 verschiedene geeignete DE-Prompter und 14.716 EN-Prompter; 579 DE- und 5.402 EN-Gesprächsbäume. 14 Kandidaten hatten eine falsche Elternrolle/Sprache und wurden von der Zählung ausgeschlossen; 29 Paare hatten leere/zu lange Texte. Das ist kein freigegebener Train/Validation-Split und überprüft weder PII noch tatsächliche Antworten oder semantische Überschneidungen.
- Der frühere private Holdout ist verloren. Ein neuer, gesondert benannter und **nicht mit alten Scores vergleichbarer** v2-Holdout mit 24–100 unabhängig formulierten Fragen wurde noch nicht erstellt. `phoenix_freeze_holdout_v2.py` und `PHOENIX_HOLDOUT_V2.md` dokumentieren die *künftige* lokale Erstellung. Fehlende Registry niemals durch Dummy-Hashes oder vorhandene Smoke-Fragen ersetzen.
- `phoenix_oasst2_bridge.py` kann das Originalarchiv erst **nach echtem v2-Freeze** lokal nach Menschen-/Rollen-/Eltern-/Text-/Tree-Kriterien und Holdout-Kollisionen exportieren. Der Quellreport bleibt `approved:false`; unabhängige Lizenz-/Herkunfts-/Privacy-/PII-/Nahe- und semantische Dublettenprüfung sind zusätzliche Stop/Go-Gates. Die realen Trainingszeilen wurden noch nicht exportiert.

## Voraussetzungen für `sft_contract.py` — verbindliche Reihenfolge

1. Lokale, vertrauenswürdige **Kopie** des eigenen stabilen `frontier350`-Checkpoints samt exaktem SHA-256 und passendem 16K-Tokenizer. `torch.load(weights_only=False)` ist nur für eigenständig als vertrauenswürdig bestätigte Pickle-Checkpoints zulässig; `--trusted-local-checkpoint` beweist die Herkunft nicht.
2. Neuen v2-Testbestand und separaten Receipt **vor** Training privat einfrieren. Originaltexte, Bewertungsschlüssel, Registry und Backups bleiben außerhalb des Git-Repositories. Kein Rebranding alter NIS2-Resultate.
3. Das lokal geprüfte OASST2-Archiv mit dem Bridge-Konverter in **neue** `train.jsonl`/`validation.jsonl`-Dateien aufteilen; Schema pro Zeile exakt `id`, `source_id`, `user`, `assistant`. Die Brücke gibt niemals automatisch eine Trainingsfreigabe. Kein Benchmarkmaterial in den Trainingsdaten.
4. `phoenix_data_audit.py` als Warnhinweis-Prescreen ausführen. Die automatische Prüfung erkennt nur exakte Prompt-Hashes, manche near duplicates und E-Mail-/Telefonmuster, nicht alle PII/semantischen Leaks; Ausgabe bleibt `approved:false`. Unabhängige Prüfung und separate **echte** Freigabe müssen dokumentiert werden.
5. `screen.json` aus unabhängigem Review: `version:1`, echte zu Train/Validation/Holdout passende SHA-256, `approved:true` nur nach tatsächlichem Review, `exact_overlap_count=near_overlap_count=privacy_hits=0`. `manifest.json`: `version:1`, Hashes `train_sha256`, `validation_sha256`, `tokenizer_sha256`, `holdout_sha256`, `screen_report_sha256`; Quellen jeweils `id`, SHA-gepinnte `revision_sha256`, geprüfte `license`, und bewusst bestätigte boolesche Freigaben `approved_for_training`, `privacy_reviewed`, `benchmark_screened`. Diese JSON-Flags sind keine juristische oder wissenschaftliche Prüfung.

Ein leerer oder fingierter Testbestand bzw. Freigabereport darf nicht zum Trainingsstart führen. Keine Rohdaten, Modelle, eigenen Checkpoints, Testfragen oder Tokenpacks nach GitHub/Hugging Face hochladen.

## EOS-supervised Pack lokal bauen (erst nach Freigabe)

Mit **funktionierender lokaler Python-Umgebung** und installiertem `tokenizers` aus dem Repo-Stamm (die bislang geprüfte Self-hosted-Runner-Umgebung kennt `python.exe` nicht auf `PATH`):

```text
python frontier350/nextgen_v1/sft_contract.py --manifest PATH_TO_MANIFEST.json --train PATH_TO_TRAIN.jsonl --validation PATH_TO_VALIDATION.jsonl --tokenizer PATH_TO_TOKENIZER.json --holdout-hashes PATH_TO_HOLDOUT_HASHES.json --screening PATH_TO_SCREEN.json --out LOCAL_NEW_PACK_DIR --seq 512
```

Das Pack benutzt `User: …\nAssistant:` und ausschließlich supervisierte Antworttokens samt echtem EOS; der Prompt wird maskiert. Ursprünglicher Tokenizer, BOS/EOS, kausale Verschiebung, ID-Eindeutigkeit, Hashes, getrennte Splits und exakte Holdout-Kollisionen werden überprüft. Keine Antwort wird trunciert; Ausgabeverzeichnis muss neu sein, maximal 20.000 Zeilen pro Split.

## Opt-in GPU-Pilot (erst wenn alle Daten-/Umgebungs-Gates bestanden sind)

```text
python frontier350/nextgen_v1/train_sft_pilot.py --execute --trusted-local-checkpoint --checkpoint PATH_TO_STABLE_LOCAL_COPY.pt --checkpoint-sha256 EXACT_64_HEX_SHA256 --pack LOCAL_NEW_PACK_DIR --tokenizer PATH_TO_TOKENIZER.json --out NEW_ISOLATED_RESULT_DIR --train-blocks 4 --batch 2 --steps 20
```

SHA-Check der **eigenen** Gewichte vor dem unsicheren Pickle-Laden; passende 349.825.152-Parameter-Architektur und Tokenizer, freie BF16-fähige RTX 4070, anfänglich mindestens 9.000 MiB VRAM frei, Starttemperatur höchstens 78 °C, während des Pilots höchstens 82 °C, reservierter VRAM höchstens 9 GiB. Nur obere 4 oder 8 Blöcke, BF16-Autocast mit Optimizer auf explizit trainierbaren Parametern, Antwort/EOS-Maskierung, Gradienten-Clipping; 1–50 Schritte; keine automatische Wiederaufnahme/lange 2–3B-Fortsetzung. Niemals parallel zu anderem GPU-Training ausführen.

**Korrigiertes, aktuelles Validierungsprotokoll:** Die Funktion wählt deterministisch bis zu **16 über den ganzen Validierungssplit verteilte Zeilen** (bei weniger Zeilen alle) **vor und nach** dem SFT. Ihre Cross-Entropy wird **pro supervisiertem Antworttoken gewichtet**, nicht fälschlich zeilenweise gemittelt. Dynamische Batch-Länge vermeidet Padding bis zu einer unnötig vollen Kontextlänge. Diese Stichprobe ist weiterhin weder vollständige Validierung noch ein unabhängiger Antwort-/NIS2-Benchmark; ein geringerer Loss führt allenfalls zu einem `production_eligible:false` isolierten Kandidaten.

## Beweisgrenzen

[Hosted-Codeprüfung #35140959328](https://github.com/Cola968/neral-helios/actions/runs/35140959328): 74 Phoenix- plus 30 SFT-Vertragstests auf synthetischen Eingaben bestanden. [Echte SAFE-64-Token-Baseline #35130560673](https://github.com/Cola968/neral-helios/actions/runs/35130560673) ist reine Engineering-Diagnose: 8/8 Antworten nichtleer, 5/8 EOS, 47 % wiederholte Wort-Trigramme und Rollenmarker in 2/8; weder Korrektheit noch NIS2. Auch das neue echte Metadaten-Audit trainiert kein Modell. Erst nach einem echten, unabhängigen v2-Holdout und freigegebenen Quelldaten kann ein Kandidat fair und blind gegen die unveränderte SAFE-Baseline mit identischen Prompts/Decoder-Einstellungen verglichen werden. Ein kürzeres Pilotexperiment belegt keine Produktionsreife oder 60–70k Tokens/s.
