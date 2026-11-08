# Neral / Helios

## NERAL 1.0 Local Preview – aktueller Windows-Release-Kandidat

**Stand 8. November 2026:** Das neue lokale Preview läuft **ohne
historischen 50M-Checkpoint** auf dem bereits installierten
`qwen3:8b` via Ollama. Eigenständige Web-App auf
`http://127.0.0.1:9690/`, FAST-/THINK-Modus, lokale Prüfwerkzeuge,
Sitzungs-/CSRF-Schutz und sichtbare Modellherkunft.

- **Release-Candidate-Artefakt:** im jeweils vollständig grünen
  [NERAL-1.0-Release-Workflow](.github/workflows/neral-1-0-release-rc.yml)
  unter GitHub Actions, inklusive ZIP und SHA256SUMS.
- **Installationsanleitung und reale Releasegrenzen:**
  [NERAL 1.0 Local Preview](docs/NERAL_1_0_LOCAL_PREVIEW_RC.md).
- **Start:** Entpacken, Ollama mit `qwen3:8b` starten,
  `Start NERAL 1.0 Local.cmd` doppelklicken. Beim ersten Start werden
  Flask/Waitress **nur mit Einwilligung** in einer eigenen Python-Umgebung
  eingerichtet. Die QLoRA-Trainingsumgebung bleibt unverändert.
- **Evidenz:** [Windows-PC-Inferenz plus Windows/Linux-Sicherheitsabnahme
  erfolgreich](https://github.com/Cola968/neral-helios/actions/runs/42490217241).

**Keine allgemeine 1.0-Modellfreigabe:** Die aktuelle Laufzeit verwendet
ein unverändertes Qwen-Grundmodell. Ein eigenständiger, dem Basismodell
überlegener NERAL-LoRA-Adapter wurde noch **nicht** unabhängig validiert
oder in dieses Preview übernommen. Das ältere 50M-Projekt bleibt
historisch unten dokumentiert.


> **Aktueller experimenteller Entwicklungszweig (November 2026):**
> Neben den historischen 20M/50M-Eigenmodellen existiert ein lokales
> **Qwen2.5-Coder-7B-Instruct + NERAL-v5-LoRA**-System.
> Das 320er-v5-Ergebnis ist ein interner In-Distribution-Test, **kein**
> Nachweis allgemeiner Überlegenheit. Der unabhängige CPU-Auswerteweg
> **NERAL-EXT** wird in [docs/NERAL_EXT.md](docs/NERAL_EXT.md) entwickelt.
> Die Repository-Sichtbarkeit ist aktuell öffentlich; die ältere
> Bezeichnung „privat“ im historischen Abschnitt ist nicht aktuell.
> GPU-Adapter und private Chatdaten werden nicht im Repository veröffentlicht.


Eigenes lokales Sprachmodell mit CASCADE, RIPPLE, SOLVE und LIVING Answers.

**Aktuell:** Neral-50M Sprachpilot mit **50.012.976 Parametern**. Ausschließlich eigene Neral-Gewichte, keine fremden Checkpoints und keine Cloud-Inferenz. Der Pilot besteht den freien Sprachtest noch nicht; **Neral-20M Preview 02 bleibt Standard**. Das ist ein Entwicklungsprojekt, kein zuverlässiger allgemeiner Chatbot.

## Start

Installation und Checkpoint-Pfade: [REPRODUCE.md](REPRODUCE.md). Danach `START_NERAL.bat` starten und http://127.0.0.1:9680 öffnen.

Ohne eigene installierte Gewichte laufen die lokalen Projektwerkzeuge, SOLVE und Codeprüfung. Chatgenerierung meldet fehlende Gewichte; sie wird nicht heimlich ersetzt. Eigene Modellartefakte gehören in die separat bereitgestellten Checkpoint-Pfade, nicht in normale Git-Dateien.

## Erste Features

- **CASCADE:** Folgen einer lokalen Projektänderung vorab berechnen, bestätigen, anwenden und rückgängig machen.
- **RIPPLE:** hinterlegte Abhängigkeiten über das ganze gewählte Projekt verfolgen.
- **SOLVE:** exakte Arithmetik und lineare Gleichungen mit Probe; reine Recheneingaben werden automatisch zugeordnet.
- **LIVING Answers:** Antworten aus lokalen Projektwerten neu berechnen und Versionen dauerhaft behalten.
- **Code prüfen:** statische Python-Syntaxprüfung; eingereichter Code wird nicht ausgeführt.

Unter **Projekte → Beispiel ausprobieren** lässt sich eine Budgetänderung testen. Daten bleiben lokal in SQLite. Externe Webseiten, Kalender oder Dateien werden durch diese erste Version nicht überwacht oder verändert. Werkzeuge sind keine behaupteten Modellfähigkeiten.

## Tatsächliche Ergebnisse

[50M-Pilot und Effizienzvergleich](PREVIEW_50M.md) · [Model Card](MODEL_CARD.md) · [Datenherkunft](DATA_PROVENANCE.md) · [Trainingslog](TRAINING_LOG.md) · [Attribution](ATTRIBUTION.md)

- 142 lokale Tests bestanden (Stand des dokumentierten Piloten).
- 999.424 Pretraining-Tokens, 0 Padding, RTX 4070.
- Gemessene Attention-Optimierung: bester Durchsatz etwa 53.700 → 76.600 Tokens/s.
- Trotz sinkendem Loss wiederholt sich der 50M-Pilot in freien Antworten stark. Keine automatische Promotion und kein großes Folgetraining gestartet.

Historie: [Originalaudit und Pilot 01](ERGEBNISSE.md), [20M Preview 02](PREVIEW_02.md), [40M und Projektfeatures](PREVIEW_03.md).

## Repository-Umfang

Quellcode, Tests, Datenskripte, ausgewählte Messberichte und Herkunftsnachweise. Ausgeschlossen: virtuelle Umgebung, Rohtrainingsdaten, lokale Projekt-Datenbank, persönliche Chats und vollständige Optimizer-Checkpoints. Große eigene Inferenzgewichte werden getrennt von Git geführt.

Dieses private Repository legt keine neue pauschale Open-Source-Lizenz für übergebenen Code oder Modellgewichte fest. Datenquellen und Softwareabhängigkeiten behalten ihre eigenen Lizenzbedingungen.
