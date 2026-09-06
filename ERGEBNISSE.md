# Neral / Helios – ausgeführter Entwicklungsstand

Grundlage ist ausschließlich dein übergebener Neral-Checkpoint. Keine fremden Modellgewichte wurden geladen. Modell, Konfiguration und Byte-Tokenizer sind gegenüber dem Archiv unverändert.

## Prüfung des Originals
- Exakt **20.587.008** trainierbare Parameter; gebundene Embedding-/Ausgabegewichte.
- Original-SHA-256: `d026525f19515a5fd64ae4e35b84fce4b1a5eaabe6f317e2b4c8223e7c460670`. Striktes Laden mit `weights_only=True`; alle Gewichte endlich.
- Alle fünf Dateien des äußeren Manifests stimmen mit Größe und SHA-256 überein. Beide enthaltenen Checkpoints sind identisch.
- Daten: 97,830 Zeilen, nur 1,021 exakt verschiedene Texte. 98.96% sind zusätzliche Wiederholungen.
- Keine exakte Train/Val-Überschneidung, aber 3 gemeinsame normalisierte Texte; dadurch 4,348 von 6,435 Validierungszeilen auch im Training enthalten.

## Tatsächlich ausgeführtes Training
- RTX 4070, PyTorch 2.8.0+cu128, bf16, unveränderte Architektur.
- 976 deduplizierte Trainingsbeispiele; 115 neue Validierungs- und 78 Testbeispiele. Eigenständig verfasste Beispiele plus überprüfte Rechen-/Code-Generatoren. Originaldaten nur als dedupliziertes Replay.
- Domänenbalanciertes Sampling; Loss nur auf Antwort-Token. Diese Metrik ist nicht direkt mit dem alten All-Token-Loss vergleichbar.
- 160 Updates, 24.82 Sekunden gemessene Trainingsschleife, 737.7 MiB maximal von PyTorch allokierter VRAM (nicht gesamter GPU-Speicher).
- Bester Checkpoint: Schritt 80. Vier spätere Prüfungen ohne Verbesserung. Stop-Regel und maximales Budget wurden beide bei Schritt 160 erreicht.
- Validierung (Macro-Antwort-Loss): **4.2595 → 2.1246**.

## Eingefrorener Testvergleich

| Bereich | Original Loss | Pilot Loss |
|---|---:|---:|
| chat | 4.1873 | 2.5991 |
| coding | 4.1815 | 2.4056 |
| math | 4.9273 | 1.0483 |
| security | 4.0905 | 2.2229 |

Test-Macro-Loss: **4.3467 → 2.0690**. Referenzantworten wurden mit identischen Verfahren ausgewertet. Niedrigerer Loss ist kein Nachweis funktionierender freier Antworten.

**Reine Modell-Mathematik: 0/12 → 0/12 korrekt.** Die gespeicherten freien Chat- und Coding-Antworten bleiben häufig unverständlich oder sachlich unpassend.
Der alte Verteilungs-Loss verschlechterte sich von **0.7899 auf 0.9741**. Deshalb ersetzt der Pilot den Originalcheckpoint nicht automatisch.

## Umgesetzte Verbesserungen
- Neu: reproduzierbare Daten-Splits, Antwortmaskierung, Domänenmetriken, Baseline vor Training, Checkpoint-Auswahl mit Guardrails, Early-Stopping, atomare Checkpoints und gespeicherte RNG-/Optimizer-/Scaler-Zustände.
- Unsichere `weights_only=False`-Aufrufe in Eval-/Generate-Helfern beseitigt. Checkpoint-Prüfsumme in der Browser-App verpflichtend.
- API: JSON-Typen, endliche Zahlen, Kontext-/Antwortbudget, Cross-Site-Schutz, Anfragelimits, exklusiver GPU-Zugriff, Abbruch, striktere CSP ohne Inline-Skripte.
- Erweiterte Security-Regression: **26/36 → 36/36**; API-Probes **4/8 → 8/8**. Die neue Suite wurde auch zur Fehlerbehebung verwendet und ist kein unabhängiger Red-Team-Test.
- Gesamte automatisierte Suite: **101 Tests, 0 Fehler**. Zusätzlich enthält ein SOLVE-Test 840 exakt geprüfte Rechenfälle.
- Browser: lokale Chats, temporärer Modus, Verlaufssuche, Export, erneutes Testen, Abbruch, Farbschema, Parameter-/Hash-Anzeige, Original/Pilot-Umschaltung und echte Messwerttabelle.
- SOLVE: begrenzter AST-Parser und exakte Bruchrechnung, kein eval/exec. Klar als Werkzeug markiert; keine behauptete Modellfähigkeit.
- Codeprüfung: Python-Syntax und wenige typische statische Muster, ohne den eingereichten Code auszuführen. Keine allgemeine Korrektheits- oder Sicherheitsgarantie.

## Grenzen und nächste sinnvolle Arbeit
Dieser Pilot verbessert die Mess- und Sicherheitsgrundlage, macht Neral aber noch nicht zu einem allgemein nützlichen Sprachmodell. Mehr Schritte auf diesen wenigen Beispielen sind aufgrund der gemessenen Überanpassung nicht sinnvoll. Der nächste substanzielle Schritt ist ein deutlich breiterer, lizenzgeprüfter Sprach-/Code-Korpus mit unabhängigen Aufgabenfamilien, bevor ein weiterer Lauf angesetzt wird. Architekturvergrößerung und fremde Gewichte sind nicht Bestandteil dieser Änderung.

## Start
Doppelklicke `START_NERAL.bat`. Die lokale Testoberfläche ist unter `http://127.0.0.1:9680` erreichbar. Standardmäßig wird das Original geladen; den Pilot wählst du bewusst unter „Modell & Messwerte“.

## Belege
`reports/source_inventory.json`, `reports/data_audit.json`, `reports/baseline.json`, `reports/after_pilot_01.json`, `reports/security_baseline.json`, `reports/security_after.json`, `reports/pytest.xml`, `runs/pilot_01/run.json`, `runs/pilot_01/metrics.jsonl`, `runs/pilot_01/summary.json`.
