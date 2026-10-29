# Arctus350 — semantische Mathematik-Datenprüfung und v1-Reparatur

**29.10.2026 | Research Issue #179 | Draft PR #182 | keine GPU / keine neuen Kosten**

## Ergebnis (tatsächlich auf GitHub-hosted CPU ausgeführt)

Der alte deterministisch erzeugte v1-Mathematikkorpus wurde früher nur auf gleiche Texte geprüft. Dass `5+6` und `6+5` semantisch dieselbe Addition sind, erkannte die lexikalische Holdout-Prüfung nicht. Die neue Prüfung arbeitet mit nachprüfbaren, kommutativen Operandenpaaren und zählt Daten-Split-Überschneidungen **ohne Rohfragen, Wertepaare oder Datensatz-IDs zu exportieren**.

| Generation | Einträge insgesamt | Operandenpaare in mehr als einem Split | Interpretation |
| --- | ---: | ---: | --- |
| v1 `arctus350_exact_math_curriculum` | 782 | **61** | **Nicht** als mathematisch strikt getrennte Entwicklungsdaten behandeln |
| v2 `arctus350_balanced_arithmetic_v2` | 9.616 | **0** | Verifizierte, im Rahmen der erkannten Vorlagen disjunkte Operandenpaare |
| v3 `arctus350_carry_curriculum_v3` | 5.100 | **0** | Verifizierte, im Rahmen der erkannten Vorlagen disjunkte Operandenpaare |

**Quelle:** [Actions-Lauf #40785868220](https://github.com/Cola968/neral-helios/actions/runs/40785868220). **Konkreter Reparaturbeleg:** [Actions #40786246597](https://github.com/Cola968/neral-helios/actions/runs/40786246597) erzeugte lokal den abgeleiteten v1-Datensatz mit **721 statt 782 Zeilen**: 58 aus TRAIN und 3 aus VAL entfernt, **TEST unverändert mit 69 Zeilen**, danach **0** arithmetische Cross-Split-Konflikte, ursprünglicher und abgeleiteter Manifest-SHA verifiziert; **15 neue CPU-Tests plus 14 Nachbarregressionen erfolgreich**. Die Zahlen wurden aus den tatsächlich vorhandenen nativen Generatoren neu berechnet, **nicht** aus Modellgenerierungen. „0“ bedeutet nicht semantische Freiheit von allen möglichen Trainingsdaten oder öffentlich versiegelten Benchmarks. Bekannte generierte Zeilen mit falschen Antworten/Metadaten werden **abgewiesen**, nicht still übersprungen.

## Zusätzlich implementiert: v1 ohne Veränderung der Originaldaten reparieren

`sanitize_math_splits(rows)` erstellt eine **neue Kopie** aller drei Splits, reserviert zuerst **TEST**, dann VAL, danach TRAIN und entfernt komplette arithmetische Fakten-Familien aus dem jeweils niedrigeren Split. Vorhandene zweisprachige und formelverschiedene Varianten innerhalb desselben Splits bleiben erhalten. Die Funktion gibt einen aggregierten Löschbericht aus und verlangt anschließend erneut 0 Überschneidungen. **Nie den Original-TEST-Split verändern.**

`write_sanitized_generated_v1(destination)` erstellt aus dem fest eingecheckten, synthetischen v1-Generator eine **neue lokale JSONL-Kopie**, prüft mit dem existierenden NeralTalk-Manifest-/Holdout-Validator und schreibt `metadata_report.json` mit SHA-256-Manifest. Bestehende Zielordner werden strikt abgelehnt; vorhandene private Datensätze werden nicht geladen. Keine automatisch gestartete Trainingsausführung.

## Befehle

```bash
# Nur CPU/Standardbibliothek, kein Modell oder Korpusdownload:
python -m unittest tests.test_arctus350_math_semantic_audit tests.test_arctus350_math_sanitization tests.test_arctus350_sanitized_export -v

# Bestehende v1/v2/v3 Generatoren auditieren – nur Aggregat-Ausgabe:
python -m frontier350.arctus350_math_semantic_audit --generated-audit

# Eigenständige bereinigte v1-Kopie – Ziel darf NICHT existieren:
python -m frontier350.arctus350_math_semantic_audit --sanitize-generated-v1-to /pfad/zu/neuem-verzeichnis

# Optional lokalen, bereits geprüften NeralTalk-Korpus überprüfen; erstellt keinen neuen Trainingssatz:
python -m frontier350.arctus350_math_semantic_audit --data /pfad/zum/korporus --strict
```

Bei echten externen Mathe-Konversationen, die das Prüfmodul nicht semantisch validieren kann, bleibt der Zustand **OPAQUE** und `--strict` verweigert die Freigabe. Das ist ein bewusstes Sicherheits-Gate, keine Aussage, dass solche Beispiele falsch seien.

## Wahrheit, Grenzen und nächster Experimentablauf

- Die 61 Konflikte sind **Entwicklungskorpus-Konflikte**, keine bewiesene Kontamination von NIS2 oder externen GPT-3-Benchmarks.
- Es wurde **kein Arctus-Modell trainiert**, und die gemessene Intelligenz/Genauigkeit hat sich dadurch **nicht** verändert.
- Neben diesem Draft existieren bereits [PR #180](https://github.com/Cola968/neral-helios/pull/180) (gepaarte Checkpoint-Evidenz) und [PR #181](https://github.com/Cola968/neral-helios/pull/181) (optionale SFT-Paddingoptimierung). Dieser Branch verändert weder deren Module noch Modellgewichte.
- Erst **nach Code-Review und überprüftem lokalen Quellmanifest** dürfen abgeleitete, separate Trainingsdaten für eine kurze, beaufsichtigte Arctus350-Ablation geprüft werden. Die bereinigten Entwicklungsdaten sind **keine unabhängige Benchmark-Suite**.
- Zusätzliche Projektkosten: **0 €**; vorhandene GitHub Actions-Nutzung nur innerhalb freier/verfügbarer Kontingente. Kein Cloud-GPU-Job, kein RTX-4070-Start außerhalb der Nutzerfreigabe, keine privaten Daten oder Checkpoints hochgeladen.

**Gute nächste Wirkung:** Bessere Auswahl von SFT-Daten und saubere Generalisierungsberichte, nicht vermeintliche Leistungssteigerung durch Wiedersehen derselben Aufgaben.
