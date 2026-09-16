# NERAL Phoenix – isolierter Entwicklungsstand und 7-Tage-Plan

**Stand: 16.09.2026. Entwurf, keine Produktionsfreigabe.** Eigene 350M-Gewichte und Tokenizer bleiben erhalten. Keine Phoenix-SFT-Gewichte trainiert, keine nachgewiesene Chatverbesserung, keine nachgewiesenen 60–70k akzeptierten Trainingstokens/s. Kein langes Training automatisch starten, keine privaten Dialogdaten, Berichte oder Gewichte nach GitHub/Hugging Face hochladen.

## Nachweisstand – strikt unterscheiden

- **Tatsächliche eigene SAFE-Inferenz bestanden:** [Windows-Self-Hosted-Run 35130240477](https://github.com/Cola968/neral-helios/actions/runs/35130240477) verifizierte den fest gepinnten SAFE-Checkpoint und generierte auf acht öffentlichen Engineering-Eingaben jeweils ein Token. Das belegt nur Ladbarkeit und Decoder-Ausführung, nicht Chatqualität.
- **Echte längere SAFE-Baseline liegt jetzt vor:** [Run 35130560673, erfolgreicher zweiter Jobversuch](https://github.com/Cola968/neral-helios/actions/runs/35130560673) führte nach dem anfänglichen Abbruch durch den GPU-Leerlaufschutz acht öffentliche, vorab im Code festgelegte Prompts mit jeweils höchstens 64 generierten Tokens auf den eigenen SAFE-Gewichten aus: 8/8 nichtleere Antworten, 5/8 echte EOS-Enden, 3/8 Budgetende, insgesamt 233 Tokens, 19,77 s innerhalb der Diagnoseroutine. `phoenix_gate.metrics` zählte **47 % wiederholte Wort-Trigramme** und Rollenkennzeichen in **25 % der Antworten (2/8)**. Diese Werte sind diagnostische Warnsignale und keine Antwortkorrektheit, NIS2-Wertung oder Trainings-Tokens/s. Rohantworten blieben aus den GitHub-Logs heraus; Gewichte unverändert, kein Training.
- **Synthetische Offline-CPU-CI:** [Run 35136782879](https://github.com/Cola968/neral-helios/actions/runs/35136782879): 47 Phoenix- plus 30 SFT-Vertragstests bestanden. [Separater Syntaxlauf 35136783104](https://github.com/Cola968/neral-helios/actions/runs/35136783104) ebenfalls erfolgreich. Vertragstests sind keine Modellqualität.
- **SFT-Validierung verbessert, real noch nicht vermessen:** bis zu 16 deterministisch über den gesamten Validierungssplit verteilte Zeilen statt der ersten 8, Loss pro beaufsichtigtem Token gewichtet statt pro Zeile, dynamische Batch-Länge statt fester Kontext-Paddinglänge. Stichprobe und neue Pad-Strategie ergeben weder eine vollständige noch unabhängige Evaluation; Geschwindigkeitssparen muss auf GPU separat gemessen werden.
- **Keine real geprüften SFT-Daten und kein Kandidatenmodell:** `sft_contract.py` und `train_sft_pilot.py` sind ausschließlich opt-in Vorbereitung/Experiment; Daten-/GPU-Gates sind noch offen.

## 7-Tage-Plan mit Stop/Go

| Tag | Umsetzung | Nur weiter nach belegtem Gate |
| --- | --- | --- |
| 1 | Eigene SAFE-Baseline zunächst technisch prüfen, dann Antworten **lokal** mit vorab fixiertem Promptset inhaltlich auswerten. | Acht öffentliche 64-Token-Entwicklungsdiagnosen liegen vor; blinde Korrektheit und repräsentative Evaluation fehlen noch. |
| 2 | Menschliche DE/EN-Dialogdaten nach Herkunft, Lizenz, PII, exakter und semantischer Überlappung prüfen. | Unabhängige Quellen-/Lizenzfreigabe, train/validation/holdout disjunkt; erst dann lokal SFT-Pack erzeugen. |
| 3 | 1–50 SFT-Schritte mit geprüften Daten, isoliertem Ergebnis und **Kopie** des stabilen Checkpoints, nur auf freier GPU. | Echte GPU-/Loss-Logs, Temperatur/VRAM, valide gespeicherter Kandidat; SAFE bleibt unverändert. |
| 4 | Unverändertes Promptset mit identischem Decoder für SAFE und Kandidat ausführen, blind Antwortkorrektheit, EOS, Wiederholungen, DE/EN, Mathe, Code und volles NIS2 prüfen. | Korrektheit nachweislich besser ohne unzulässige Regression. Niedrigerer Loss genügt nicht. |
| 5 | Fehler gezielt korrigieren, Entwicklungsdaten statt Release-Holdout zur Auswahl nutzen. | Eingefrorene Tests unangetastet; keine verdeckte Überanpassung. |
| 6 | Nur nach bestandenem Vergleich opt-in Kandidat in experimentelle Helios-Oberfläche einbinden. | Lokalstart/HTTP/Origin/Chatverlauf/Leistungsaufnahme geprüft; Modell- und Tool-Ergebnisse getrennt. |
| 7 | Mehrturn- und Sicherheits-/Stabilitätsprüfung plus experimentelle Abnahme. | Nachvollziehbare Qualität und SAFE-Rückfalloption; sonst kein Release. |

Zeitplan = Arbeitshypothese, keine Fertigstellungs- oder Qualitätsgarantie. Scheitert ein Gate, nicht die nächste Stufe erzwingen.

## Vorabprüfung lokaler Dialogdaten – kein automatisches Approve

`phoenix_data_audit.py` prüft ausschließlich **lokal** die beiden JSONL-Splits (`id`, `source_id`, `user`, `assistant`) und die bestehende SHA-256-Liste der eingefrorenen Holdout-Prompts. Es gibt Zählwerte für exakte Überlappung, doppelte IDs und Hinweise auf nahe Dubletten oder E-Mail-/Telefonmuster aus. Es prüft Ähnlichkeit nur näherungsweise (max. 512 Zeichen/32 Sample-Shingles je Prompt) und erkennt **weder alle Dubletten noch alle privaten Inhalte**. Ergebnis enthält immer `approved: false`; es ist kein Lizenz-, Datenschutz- oder Benchmark-Audit und kann die zwingende unabhängige `sft_contract.py`-Freigabe nicht ersetzen.

PowerShell aus dem Repo-Stamm (nach Prüfung deiner **eigenen** bestehenden Pfade; neue Ausgabedatei):

```powershell
$root = $env:NERAL_ASSETS_ROOT
python frontier350/nextgen_v1/phoenix_data_audit.py --assets-root "$root" --train "$root/phoenix/train.jsonl" --validation "$root/phoenix/validation.jsonl" --holdout-hashes "$root/phoenix/holdout_hashes.json" --out "$root/phoenix/prescreen-new.json"
```

Nicht vorhandene Pfade sind Platzhalter – das Skript legt weder ein künstliches Dataset noch eine scheinbar positive Prüfbestätigung an. Erst nach unabhängig überprüfter Datenherkunft, Lizenzen, Datenschutz, tatsächlichen Naheduplikaten und Holdout-Kontamination darf ein Mensch einen separaten, auf den Datei-Hashes beruhenden Freigabebericht ausstellen. `test_phoenix_audit_interlock.py` bestätigt mit künstlichen Daten, dass der Prescreen nicht als Freigabe durch `sft_contract.check_manifest()` durchgeht.

## Echte Rohantworten lokal erzeugen – nur wenn der Rechner frei ist

`phoenix_capture.py` ist opt-in, read-only, SHA-gepinnt und zählt wirklich erzeugte EOS-Tokens inklusive ihrer Tokenposition. Ausgabe enthält private Antworten **nur** in einer lokalen neuen Berichtdatei, nicht im Terminal. Erst selbst verifizieren, dass keine anderen GPU-/Trainingsjobs laufen und genügend RAM/Disk vorhanden sind. `--device cpu` beansprucht keine CUDA-Inferenz, aber dennoch erheblichen RAM/CPU; nicht parallel zu wichtigem laufendem Training.

Benötigt: NERAL_ASSETS_ROOT, die dort liegende selbst geprüfte SAFE-Kopie und eigener Tokenizer sowie 8–128 unterschiedliche **vorab fixierte** Prompts als JSON-Array mit genau `id` und `prompt` in `$root/phoenix/prompts.json`. Der Ordner muss existieren; Beispiele sind keine echten Release-/NIS2-Holdouts. Nie eingefrorene NIS2-Antworten zum Trainieren verwenden.

```powershell
$root = $env:NERAL_ASSETS_ROOT
$safe = Join-Path $root 'runs/neral350_autorecovery_neral-autorecovery-v1-20260912-r1_selected/resume.pt'
$prompts = Join-Path $root 'phoenix/prompts.json'
$before = Join-Path $root 'phoenix/baseline-new.json'
python frontier350/nextgen_v1/phoenix_capture.py --execute --trusted-local-checkpoint --assets-root "$root" --checkpoint "$safe" --checkpoint-sha256 ef3ee045203f1bfb7b9f501db800408b095aa5ef29da5d22a8a9486e115e9f7c --kind safe --prompts "$prompts" --out "$before" --device cpu --max-new-tokens 64
```

`phoenix_capture.py` kann die Unabhängigkeit/Testherkunft nicht selber authentifizieren; vor dem Start muss der Operator eindeutige, verschiedene Fragen bestätigen. Die Ausgangsdateien/Reports nicht wiederverwenden oder überschreiben. Für SFT-Kandidat: dasselbe Promptset, identisches Decoderprotokoll, neue Berichtsdatei, `--kind sft-pilot` und unabhängig geprüfter Kandidaten-SHA. Unbekannte Torch-Pickle-Dateien niemals laden; `--trusted-local-checkpoint` ist eine Bestätigung, keine Sicherheitsprüfung.

## A/B-Gesundheitsprüfung nach wirklichem SFT

```powershell
python frontier350/nextgen_v1/phoenix_gate.py --baseline PFAD_ZUR_BASELINE.json --candidate PFAD_ZUM_KANDIDATEN.json --out NEUER_GESUNDHEITSBERICHT.json
```

Identische Fall-IDs, **eindeutige** originale UTF-8-Prompt-Hashes, Protokoll und Tokenbudget, verschiedene Checkpointidentitäten sowie `raw_model=true` und `tool_calls=0` sind erforderlich. Wirklich erzeugtes EOS zählt als EOS; sofortiges EOS zählt zugleich als leere Antwort. Abgebrochene, künstlich gestoppte oder unvollständige Budget-Runs sind ungültig. Nur aggregierte Raten/Hashes dürfen GitHub erreichen; auch ein bestandener Health-Test sagt **nichts** über tatsächliche Antwortkorrektheit, völlige NIS2-Qualität oder Produktionsreife.

## Tests und Grenzen

```powershell
python -m unittest discover -v -s frontier350/nextgen_v1 -p 'test_phoenix_*.py'
python -m unittest discover -v -s frontier350/nextgen_v1 -p 'test_sft_*.py'
```

GitHub Actions: `.github/workflows/phoenix-gate-ci.yml` ist gehostete synthetische CPU-Prüfung. `.github/workflows/phoenix-own-weight-smoke.yml` ist read-only, eigens isolierter Self-Hosted-Engineering-Lauf mit festem öffentlichen Promptset und GPU-Leerlaufschutz. Keine automatische lange SFT-Kampagne und keinerlei Modellpromotion. Der bestehende `SFT_PILOT.md` enthält die noch ausstehenden realen SFT-Voraussetzungen.
