# NERAL Release-Ratchet — ab 10.11.2026

Ziel: Experimente dürfen den funktionierenden Standardpfad nicht ohne nachgewiesenen Nutzen ersetzen. Ein gescheiterter Entwicklungsversuch ist erlaubt; ein unbelegter Qualitätsgewinn in `main` ist es nicht.

## 1. Was unverändert bleibt

- **Release-Standard:** die bereits geprüften lokalen Qwen/Ollama-Pfade. Experimentelles TIR nur ausdrücklich pro Aufruf mit `experimental_tir=true` **und** lokal gesetztem `NERAL_EXPERIMENTAL_TIR=1`. Keine automatische Aktivierung für reale Nutzer.
- **Release-Sicherheit:** Der bekannte grüne Windows-Release-Run gilt ausschließlich für seinen exakten Commit. Jede Änderung an Release-Code oder Assets verlangt die entsprechenden neuen Checks und einen Installations-Smoke für das endgültige Downloadpaket.
- **Nicht bewiesene Fähigkeiten:** Keine Freigabe eines LoRA-Modells, einer Router-Überlegenheit, einer lokal vollständig laufenden Sprachfunktion oder einer Smartphone-App nur wegen grüner Unit-Tests.

## 2. Messbarer Entwicklungs-Ratchet (Hard36)

**Nicht** jedes zukünftige Hard36 muss mindestens 20/36 erreichen: Die Fälle werden mit einem frischen Zufallsseed generiert. Die 20/36 von Qwen stammen aus einem einzelnen Lauf vom 09.11.2026; sie sind kein dauerhaft vergleichbarer Fixwert.

Für jeden neuen **gepaarten** 36-Fall-Lauf gilt stattdessen:

1. `raw_qwen35` und `neral_high_router` bearbeiten jeweils exakt denselben Satz (gleiche Aufgaben, dekodierendes Modell und Ausgabelimits). Die ursprüngliche Qwen-Antwort darf durch eine falsche Werkzeugausgabe nicht als korrekt gewertet werden.
2. Der neue Router darf in diesem *Entwicklungsdiagnostik*-Lauf nicht unter die gleichzeitige Qwen-Basis fallen. Der CPU-Prüfer `python -m scripts.neral_dev_ratchet <scalar-summary.json>` meldet bei Abfall einen Fehlerstatus, prüft die gepaarten Gewinn-/Verlustzahlen und lehnt falsche Provenienz ab.
3. Dieses Ergebnis bleibt **entwicklungsspezifisch**. Selbst 30/36 hier sind kein Beweis für einen Gewinn auf unbekannten Problemfamilien, keine Auslieferungsfreigabe und kein Nachweis verbesserter Modellgewichte.
4. Der Prüfer muss auf das originale, lokal erzeugte, skalare GitHub-Artefakt angewendet werden. Selbsterfundene Zahlen, ein künstliches Test-Fixture oder bloße Modellbehauptungen sind keine gültigen Belege. Die GitHub-CI prüft die Prüflogik; sie lässt **nicht** automatisch jede PR einen GPU-Benchmark ausführen.

## 3. Fünf-Aufgaben-Wächter

Vor einer manuell genehmigten großen GPU-Evaluation: **fünf vorab festgelegte und echte** GPU-Aufgaben, bei denen die Basis im selben Protokoll richtig liegt. Verliert der Kandidat eine davon, abbrechen und untersuchen. Keine harten Regeln anhand des Wortlauts dieser fünf Aufgaben trainieren oder nachträglich anpassen. Die Auswahl/Goldantworten und der Test auf echter GPU sind derzeit **nicht automatisiert**; der CPU-Test ist kein Ersatz dafür. Feste Canary-Aufgaben schützen bekannte Fehler, messen aber nicht allein Generalisierung.

## 4. Ein-Hebel-Experiment und sichere Rücknahme

Bei jeder qualitätsrelevanten PR eine primäre Hypothese mit Vorher-Nachher-Messung. Begleitende Tests und Dokumentation sind ausdrücklich erlaubt. Kein gleichzeitiges Umschreiben von Sampling, Prompt, Router und Scorer, wenn dadurch die Ursache eines Ergebnisses verschleiert würde. Modelländerungen bleiben auf dem Branch oder hinter einem deaktivierten Feature-Flag, bis die Nachweise vorliegen.

**Kein automatischer Qualitäts-Fallback pro Frage:** Ohne externes Oracle lässt sich eine falsche Werkzeugantwort oft nicht erkennen. Im Fehlerfall kann die Basisantwort genutzt werden, aber das garantiert weder Korrektheit noch Gleichstand mit der Baseline; auch ein durch zwei Modellaufrufe geänderter Kontext kann neue Fehler erzeugen.

## 5. Tatsächliche Freigabe-Hürde

Eine Verbesserung des Standardverhaltens braucht vor dem Merge: (a) grüne CPU-Regressionen, (b) gepaarte GPU-Qualität ohne negative Auswirkung auf den festgelegten Vergleich, (c) separate unbekannte Holdout-Aufgaben für Behauptungen über Qualitätsgewinn, (d) Windows-Release- und ZIP/EXE-Installationsabnahme auf dem **finalen** SHA, (e) explizite Freigabe und Rollbackmöglichkeit.

Branch-Schutz mit verpflichtenden Status-Checks muss zusätzlich in GitHub konfiguriert werden. Diese Datei und der neue CPU-Workflow allein können ohne Branch-Protection **keine Merges nach `main` technisch verbieten**.

**10.11.2026:** Kein Last-Minute-Merge des unbewiesenen TIR in den Standardpfad. Eine geprüfte lokale Beta kann veröffentlicht werden; vollständige 1.0 nur bei bestandenem GO/NO-GO aus [Issue #495](https://github.com/Cola968/neral-helios/issues/495).
