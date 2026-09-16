# NERAL Phoenix — sicherer 7-Tage-Entwicklungsplan

**Status 16.09.2026:** Opt-in Rohgenerierungs-Exporter und Gesundheitsvergleich sind implementiert und werden ausschließlich durch synthetische CPU-Tests geprüft. **Kein neuer Phoenix-Checkpoint, keine eigene GPU-Inferenz in diesem Zweig, keine nachgewiesene Qualitätssteigerung.** Bestehende SAFE- und 2B-Gewichte bleiben unverändert; PR bleibt Draft. Ein Zeitplan ist keine garantierte Fertigstellung.

## Fahrplan mit Stop/Go-Kriterien

| Tag | Umsetzung | Nachweis vor dem nächsten Schritt |
| --- | --- | --- |
| 1 | SAFE-Checkpoint/Tokenizer und ruhenden GPU-Zustand verifizieren; Rohantworten mit unverändertem Promptset erfassen. | SHA-256, echte Rohantworten, EOS/Abbruch-Protokoll, keine Änderungen an Gewichten. |
| 2 | Menschliche Dialogdaten unabhängig auf Lizenz, PII, Naheduplikate, Prompt-Gruppen und Benchmark-Leakage prüfen; SFT-Pack erzeugen. | Nachvollziehbare Quellen, bestätigte Prüfbelege, train/validation/holdout disjunkt. Keine erfundenen Freigaben. |
| 3 | Bestehenden `train_sft_pilot.py` als begrenzten 1–50-Schritte-Versuch auf einer **Kopie** eines stabilen Checkpoints ausführen, nur wenn GPU frei. | Tatsächliche Trainingslogs, Temperatur/VRAM, Antwort- und EOS-Validierung; nie den SAFE-Checkpoint überschreiben. |
| 4 | Echte identische Baseline-/Kandidaten-Generierungen und unabhängigen Blind-Korrektheitsvergleich plus NIS2 durchführen. | Gesundheitsprüfung ohne Regression **und** nachgewiesene Antwortverbesserung. Niedrigerer Loss genügt nicht. |
| 5 | Nur durch Tests belegte Fehler nachbessern; erneute unveränderte Prüfung. | Keine unbemerkte Änderung des eingefrorenen Holdouts; kein blindes Mehrtraining. |
| 6 | Falls Kandidat alle Gates besteht: zunächst experimentell und explizit in Helios integrieren; Rohmodell und Werkzeuge getrennt anzeigen. | Lokaler Start, Gesprächsverlauf, HTTP-Sicherheit, Laufzeit- und Speichertests. |
| 7 | Längere Dialoge, Robustheit, Sicherheit, Fehlerfälle und Freigabeentscheidung. | Vorher festgelegte Mindestkriterien und unveränderte SAFE-Rückfalloption. |

Ein fehlgeschlagener Gate verlängert den Zeitplan; er darf nicht durch Umbenennung eines Checkpoints umgangen werden. Langfristige 60–70k Tokens/s und ein guter allgemeiner Chatbot sind separate, bisher unbelegte Ziele.

## Rohantworten auf deinem eigenen PC erfassen (Tag 1)

`phoenix_capture.py` benutzt die echten lokalen Neral-350M-Gewichte, native Tokenizer-/Modellarchitektur und `kv_decode.py`; EOS wird als **wirklich generiertes Token** inklusive Tokenzähler erfasst. Das Skript führt **keinen Trainingslauf** aus, lädt nichts herunter und schreibt nicht in das Repository. Ein GPU-Aufruf wird bei zu wenig VRAM oder belegter/heißer RTX 4070 verweigert; Standard ist CPU. Prüfe selbst zusätzlich, dass kein Training parallel startet.

Benötigt: eine lokale, vertrauenswürdige, SHA-geprüfte Kopie des eigenen Checkpoints, den passenden eigenen Tokenizer und 8–128 **vorher festgelegte** private Prompts innerhalb des Neral-Assets-Verzeichnisses. Die Prompts stehen in einer JSON-Datei, die ein Array von Objekten mit genau `id` und `prompt` enthält. Beispiel-Eintrag: `{"id":"dialog-01","prompt":"Hallo, wer bist du?"}`. Ein Beispielprompt ist kein unabhängiger Benchmark. Keine eingefrorenen NIS2-Aufgaben in Trainingsdaten verwenden.

Im Repo-Stammverzeichnis, PowerShell, erst nach lokaler Prüfung der Pfade:

```powershell
$root = $env:NERAL_ASSETS_ROOT
$safe = Join-Path $root 'runs/neral350_autorecovery_neral-autorecovery-v1-20260912-r1_selected/resume.pt'
$prompts = Join-Path $root 'phoenix/prompts.json'
$before = Join-Path $root 'phoenix/baseline.json'
python frontier350/nextgen_v1/phoenix_capture.py --execute --trusted-local-checkpoint --assets-root "$root" --checkpoint "$safe" --checkpoint-sha256 ef3ee045203f1bfb7b9f501db800408b095aa5ef29da5d22a8a9486e115e9f7c --kind safe --prompts "$prompts" --out "$before" --device cpu --max-new-tokens 64
```

`$root` muss tatsächlich gesetzt sein und der Ordner `phoenix` muss bereits existieren. Die Ausgabedatei darf nicht existieren. Diesen Vorgang **nicht** erneut auf denselben Ausgabepfad ausführen. Für einen neu trainierten Kandidaten verwende denselben unveränderten `prompts.json`, `--kind sft-pilot`, dessen **unabhängig überprüften** SHA-256 und einen neuen Ausgabepfad; alte Ausgaben niemals überschreiben. Nur vertrauenswürdige lokale Torch-Pickle-Checkpoints laden (`--trusted-local-checkpoint` ist eine Erklärung, keine Sicherheitsprüfung).

Während der Erfassung erscheinen echte Fortschrittszeilen mit Fallnummer, Tokenanzahl und `finish_reason`, aber ohne Frage oder Antwort. Der private Bericht enthält die Antworten und bleibt lokal. Keine Reports oder Gewichte nach GitHub/Hugging Face hochladen.

## Auswertung nach einem tatsächlichen GPU-SFT und zweitem Export

```powershell
python frontier350/nextgen_v1/phoenix_gate.py --baseline PFAD_ZUR_BASELINE.json --candidate PFAD_ZUM_KANDIDATEN.json --out NEUER_GESUNDHEITSBERICHT.json
```

`phoenix_gate.py` verlangt identische Fall-IDs, SHA-256 der **ursprünglichen UTF-8-Prompts**, identisches Decoderprotokoll und Tokenbudgets, verschiedene Checkpoint-Identitäten, `raw_model=true`, `tool_calls=0` und ein korrektes EOS- oder Budget-Ende. Ein sofortiges echtes EOS zählt als EOS **und gleichzeitig als leere Antwort**. Ein unvollständiger Budgetabbruch oder eine künstliche Stoppsequenz wird abgewiesen. Ergebnisse enthalten nur Aggregatwerte und Hashes. Exit-Code 2 bedeutet Gesundheitsregression. Ein bestandenes Gesundheits-Gate ist **kein Nachweis korrekter Antworten** und keine Produktionsfreigabe. Die Authentizität der Eingabeberichte muss unabhängig geprüft werden.

## Tests und echte Logs

```powershell
python -m unittest discover -v -s frontier350/nextgen_v1 -p 'test_phoenix_*.py'
```

GitHub Actions: `.github/workflows/phoenix-gate-ci.yml`. Diese CI prüft nur Syntax und künstliche Offline-Vertragstests. Ohne ausdrücklich gestarteten, lokal verifizierten GPU-Lauf gibt es **keine Trainingslogs** und keine gemessene Phoenix-Chatqualität. Der separate bestehende `SFT_PILOT.md` beschreibt die begrenzte Trainingskomponente und ihre nicht erfüllten Voraussetzungen; seine Ausführung darf nicht durch den bloßen Wunsch nach schnellen Resultaten erzwungen werden.
