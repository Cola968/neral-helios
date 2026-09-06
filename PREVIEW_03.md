# Neral Preview 03 – eigenes 40M-Modell und erste Features

## Modell und Abstammung
**40.051.200 Parameter**, ausschließlich aus deinem eigenen Neral weiterentwickelt. Sechs Transformer-Blöcke, Modellbreite 512 und Byte-Tokenizer bleiben erhalten. Die FFN-Breite wächst von 1536 auf 3648. Kein fremder Checkpoint und keine Cloud-Inferenz. Modellcode und ursprünglicher Checkpoint bleiben erhalten.

Die vorhandenen Gewichte wurden kopiert. Neue Gate-/Up-Gewichte wurden lokal initialisiert, neue Down-Projektionen zunächst auf null gesetzt. Damit beginnt die größere Architektur mit praktisch derselben Funktion. Vor der Änderung wurde die 20M-Baseline erneut gemessen.
- Maximale FP32-Logit-Abweichung auf dem dokumentierten Probe-Tensor: 3.33786011e-06.
- BF16-Validation vor Erweiterung: 2.1298665; danach, vor Training: 2.1297359.
- Die erste BF16-Toleranz von 0,0001 war für die geänderte Matrixform geringfügig zu eng (Differenz rund 0,000131). Nach Prüfung des FP32-Vergleichs wurde die BF16-Grenze auf 0,01 gesetzt. Das ist ein numerischer Unterschied, kein gemessener Lerngewinn.

## Tatsächlich ausgeführtes Training
Unveränderte Curriculum-v2-Splits: 1515 Training / 177 Validation / 155 Regressionstest. Antwortmaskierung, sechs Domänen, BF16 auf RTX 4070, effektive Batchgröße 32. Keine zusätzlichen Daten in diesem Durchlauf.

| Versuch | Updates | Stop | Letzte Validation | Zeit Trainingsschleife |
|---|---:|---|---:|---:|
| neral40m_01 | 100 | early_stopping | 2.18750 | 42.27 s |
| neral40m_capacity | 100 | early_stopping | 2.16133 | 37.12 s |

Erster Lauf: alle Gewichte trainiert, LR 1e-5. Zweiter Lauf: nur die 19.464.192 neuen Parameter, LR 2e-5, kopierte Parameter durch Einfrieren/Gradientenmasken und deaktivierten Weight Decay unverändert. Beide hatten maximal 400 Updates, Evaluation alle 25 Updates und Geduld für vier Prüfungen ohne ausreichende Verbesserung.

**Keiner der beiden Läufe verbessert den Macro-Validierungs-Loss.** Early-Stopping greift jeweils bei 100 Updates; die beste Bewertung bleibt beim erweiterten Ausgangszustand, Schritt 0. Deshalb bleibt Neral-20M Preview 02 der Standard. Der tatsächlich trainierte 40M-Kandidat aus dem zweiten Lauf ist bewusst als experimentelle Vergleichsoption verfügbar.

Die übernommenen Parameter des zweiten Kandidaten wurden nach dem Training bitgenau gegen Preview 02 geprüft. Neue Down-Gewichte sind tatsächlich verändert. Modellwachstum allein ersetzt keinen deutlich breiteren, geprüften Sprachkorpus.

## Regressionstest
| Bereich | 20M Preview 02 | 40M Trainingsversuch |
|---|---:|---:|
| chat | 2.2814 | 2.3582 |
| coding | 2.1670 | 2.2110 |
| human_de | 2.3021 | 2.3269 |
| human_en | 2.5399 | 2.4496 |
| math | 1.0022 | 0.9998 |
| security | 2.0301 | 2.0409 |

Macro-Test-Loss: 2.0538 → 2.0644. Derselbe bereits betrachtete Testsatz; kein neuer unabhängiger Benchmark. Reine Modellrechnung: 0/12 korrekt. Freie Chatantworten sind weiterhin oft unverständlich. Rohantworten bleiben in reports/neral40m_candidate_test.json erhalten.

## Features V1 ausprobieren
Öffne http://127.0.0.1:9680 und klicke auf **Projekte → Beispiel ausprobieren**. Das Beispiel ist ausdrücklich eine lokale Budgetplanung, keine aus deinen privaten Dateien abgeleitete Analyse.

1. **CASCADE:** Budget 1200 → 1600 als Vorschau anzeigen. Reserve, verfügbares Budget, Live-Antwort und Einkaufsliste werden als Folgen sichtbar. Erst „Änderungen bestätigen & anwenden“ speichert. Undo setzt Projektwerte zurück, ohne Versionsgeschichte zu löschen. Veraltete Vorschauen und doppeltes Anwenden werden blockiert.
2. **RIPPLE:** Verfolgt alle hinterlegten Verbindungen über das gesamte gewählte Projekt. Zeigt betroffene Einträge, einen Pfad dorthin und ihre direkten Abhängigkeiten. V1 erkennt keine unangemeldeten Dateien oder impliziten Beziehungen.
3. **SOLVE:** Eigener Modus für exakte Bruchrechnung, Potenzen und nun lineare Gleichungen: `2*x + 3 = 11` liefert `x = 4` mit Umformung und Probe. Nichtlineare Gleichungen werden klar abgewiesen; keine allgemeine Problemlösefähigkeit behauptet.
4. **LIVING Answers:** Formeln und Antwortvorlagen als „Live“ markieren. Bestätigte Änderungen ihrer lokalen Quellen berechnen sie neu und speichern Version, Vorgängertext, Zeitpunkt und Änderungsgrund. Auch Undo erzeugt eine neue Version. Unveränderte Prüfungen erzeugen keine falsche neue Version. Daten und Historie bleiben in SQLite über Neustarts erhalten.

Eigene Projekte lassen sich anlegen; Annahmen, Formeln, Antwortvorlagen und Aufgaben hinzufügen. Formeln wie `budget * 0.1` und Vorlagen wie `Verfügbar: {{budget}} EUR` erzeugen Abhängigkeiten. Betroffene Aufgaben erhalten einen Prüfhinweis und können als geprüft markiert werden.

**Umfang:** Diese Features sind lokale, deterministische Werkzeuge. Sie sind keine vom Neral-Modell zuverlässig erkannten oder gelernten Fähigkeiten. Live-Antworten reagieren auf Änderungen im gespeicherten Projekt; externe Webseiten werden nicht überwacht. Undo ist nur unmittelbar nach der letzten CASCADE-Transaktion möglich, damit spätere Arbeit nicht überschrieben wird.

## Prüfung und Start
- 137 automatisierte Tests, 0 Fehler. Darunter atomare Vorschauen, Konflikte, projektübergreifende Trennung, Undo, Persistenz, Live-Versionen, 756 konstruierte lineare Gleichungen sowie bestehende Sicherheitsprüfungen.
- JavaScript-Syntaxprüfungen durchgeführt. Browser-/HTTP-Prüfungen separat in reports/browser_qa_03.json und reports/live_http_03.json.
- Start: START_NERAL.bat. Eigene 40M-Variante unter „Modell“ auswählbar; Warnung auf schlechtere Validierung sichtbar.
- Lokale Projektdatei: local_state/projects.sqlite3. Trainings- und Modellartefakte liegen getrennt unter runs/, seed/ und export/.

## Nachweise
- reports/growth_40m.json und baseline_before_growth.json: gemessener Architekturvergleich.
- runs/neral40m_01/ und runs/neral40m_capacity/: tatsächliche Messreihen, bester und letzter Checkpoint.
- reports/capacity_integrity.json: Abstammung, Hashes und unveränderte übernommene Gewichte.
- export/neral40m/neral40m.pt: tatsächlich trainierter experimenteller 40M-Kandidat.
- seed/neral40m.pt: funktionsbewahrend erweiterter Ausgangszustand.
- reports/comparison.json: aktuelle zusammengeführte Messwerte.
