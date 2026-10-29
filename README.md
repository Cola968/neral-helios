# Neral / Helios

Eigenes lokales Sprachmodell mit CASCADE, RIPPLE, SOLVE und LIVING Answers.

**Aktuell:** Eigene Neral-Byte-Modelle und ein experimenteller Arctus350-Pfad; Neral-20M Preview 02 bleibt, wenn installiert, die Standardauswahl. Optional lässt sich der NERAL Assistant ausdrücklich aktivieren: Er verwendet Qwen3.5 als fremdes Open-Weight-Modell über lokales Ollama. Der eigene Neral-50M-Sprachpilot mit **50.012.976 Parametern** besteht den freien Sprachtest noch nicht. Das Projekt ist experimentell; die tatsächliche Qwen-Anbindung über die NERAL-Oberfläche wurde noch nicht mit echten Gewichten bestätigt.

## Start

Installation und Checkpoint-Pfade: [REPRODUCE.md](REPRODUCE.md). Danach `START_NERAL.bat` starten und http://127.0.0.1:9680 öffnen.

[Qwen-Assistant lokal aktivieren und Modellherkunft](docs/NERAL_QWEN_INTEGRATION.md) · [Arctus350 konfigurieren](docs/ARCTUS350_FLASK.md). Beide Modellpfade sind optional und verwenden unterschiedliche Tokenizer und Gewichte. Die App lädt keine Modelle automatisch herunter.

Für Qwen speichert der Chat zusätzlich den von Ollama gemeldeten Modell-Digest. Ändert sich der Modellstand unter demselben Namen, wird älterer Verlauf nicht unbemerkt weitergereicht.

Ohne eigene installierte Gewichte laufen die lokalen Projektwerkzeuge, SOLVE und Codeprüfung. Chatgenerierung meldet fehlende Gewichte; sie wird nicht heimlich ersetzt. Eigene Modellartefakte gehören in die separat bereitgestellten Checkpoint-Pfade, nicht in normale Git-Dateien.

## Chat Experience & Stability Update

Der echte Einstieg `START_NERAL.bat` → `launcher.py` → `app.py` bietet jetzt
auch im normalen Chat ohne geladenes Modell vollständige DE/EN-Rechenanfragen:
`7+8`, `57+68`, `What is 57 plus 68? Reply only with the number.` und `(7+8)*2`.
Diese Antworten sind ausdrücklich **Rechenwerkzeuge**, keine Modellleistung.
Zusätzliche, nicht unterstützte Arbeitsschritte werden nicht still übergangen.

Die Oberfläche hat eine überarbeitete Desktop-/Mobilansicht, eine wachsende
Eingabeleiste, sichtbare Zustände und Herkunftslabels. Fehlgeschlagene Nachrichten
und Entwürfe bleiben erhalten; unvollständige oder importierte Antworten werden
nicht als eigener Modellkontext verwendet. Streaming und lokale Synchronisierung
bleiben integriert. Importierte externe Antworten werden als solche angezeigt.
Arctus350 kann experimentell und Qwen als ausdrücklich aktivierter lokaler
Assistant verwendet werden; ihre Antworten und Gesprächskontexte bleiben getrennt.

[Architekturprüfung, Startanleitung, Tests und Screenshots](docs/CHAT_EXPERIENCE_UPDATE.md).

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

