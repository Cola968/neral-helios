# Neral Preview 02 – tatsächlich ausgeführte Arbeit

## Eigenes Modell bleibt erhalten
Ausschließlich die übergebenen Neral-Gewichte wurden weitertrainiert. Keine Gewichte von Qwen, Llama, ChatGPT oder anderen Modellen. Modellcode, Byte-Tokenizer und Architektur sind unverändert: **20.587.008 trainierbare Parameter**.

Checkpoint-Abstammung: übergebenes Original → Pilot 01 (ausgewählt: Schritt 80) → Preview 02 (ausgewählt: Schritt 100). Das Original ist unverändert verfügbar.

## Daten und Training
1515 Trainingsbeispiele: bisherige 976 plus 59 deutsche und 480 englische Dialogpaare. Validation: 177 Beispiele, Test: 155. Die Ergänzung stammt aus OpenAssistant/oasst1, laut Metadaten menschlich verfasst, Apache-2.0. Es wurden ausschließlich Daten, kein Modell geladen. Revision, Prüfsummen, Lizenz und Filterregeln sind lokal gespeichert.

Nur bewertete Wurzelfragen und bestplatzierte Antworten, Ausschluss markierter synthetischer Inhalte, bestimmter Qualitäts-/PII-/Sprachprobleme, erkannter sachlicher Probleme und zu langer Texte. Vollständige Antworten bis 1024 Byte-Token; keine stille Kürzung. Gesprächsbäume bleiben innerhalb eines Splits. Bewertungen garantieren keine sachliche Fehlerfreiheit.

RTX 4070, BF16, AdamW, Lernrate 2e-5, effektive Batchgröße 32. Budget 400 Updates, Validierung alle 25 Updates, Early-Stopping nach vier Prüfungen ohne ausreichende Verbesserung. Tatsächlich **200 Updates in 65.25 Sekunden** Trainingsschleife. Bester Stand **Schritt 100**, spätere Updates wurden verworfen. Peak von PyTorch allokierter Speicher: 1144.8 MiB; nicht gesamter GPU-Speicher.

Validation-Macro-Loss über sechs Domänen: **2.5873 → 2.1299**. Auswahl ausschließlich anhand Validation, mit begrenzter zulässiger Verschlechterung je Domäne.

## Vergleich auf Testdaten
| Domäne | Pilot 01 | Preview 02 |
|---|---:|---:|
| chat | 2.5991 | 2.2814 |
| coding | 2.4056 | 2.1670 |
| human_de | 2.9482 | 2.3021 |
| human_en | 3.7964 | 2.5399 |
| math | 1.0483 | 1.0022 |
| security | 2.2229 | 2.0301 |

Antwort-Token-Cross-Entropy, niedriger ist besser. Die vier alten Domänen sind inzwischen wiederholt betrachtete Regressionstests. Die neuen menschlichen Testdaten wurden nach dem Training verglichen; nur 7 deutsche und 70 englische Paare. Kein belastbarer allgemeiner Qualitätsbenchmark.

**Freie Chat- und Codeantworten bleiben oft unverständlich. Reine Modellrechnung: weiterhin 0/12 exakt korrekt.** Der niedrigere Loss bedeutet, dass Referenzantworten wahrscheinlicher werden; er bedeutet nicht, dass Neral diese Antworten selbst zuverlässig erzeugt.
Alter, durch normalisierte Duplikate belasteter Verteilungs-Loss: 0.9741 → 0.9919. Historische Baseline und erste Prüfung bleiben in ERGEBNISSE.md erhalten.

## Preview und Prüfung
ChatGPT-ähnliche Bedienstruktur mit eigener Neral-Identität: schmale Gesprächsseitenleiste, neutraler dunkler/heller Hintergrund, zentrierter Startbereich, großes abgerundetes Eingabefeld, Nutzerblasen, Kopieren, Verlauf, temporäre Chats und getrennt öffnende Modellinformationen. Preview 02 ist beim Start aktiv; Original und Pilot 01 bleiben auswählbar.

103 automatisierte Tests bestanden; dazu JavaScript-Syntaxprüfung. Die HTTP-Integration prüft echte Generierung aus allen drei eigenen Checkpoints sowie SOLVE, Codeprüfung und Sicherheitsregeln. Browser-Sichtprüfung ist separat in reports/browser_qa_02.json dokumentiert.

SOLVE rechnet deterministisch; Codeprüfung analysiert Python statisch. Beides ist ausdrücklich Werkzeugleistung, keine behauptete Modellfähigkeit. Prompt-Guard und kleine Security-Suite bieten keine umfassende Sicherheitsgarantie.

## Start und Nachweise
Doppelklick auf START_NERAL.bat oder http://127.0.0.1:9680 öffnen. Alles zur Laufzeit lokal, keine Cloud-Inferenz.

- runs/pilot_02/: vollständige Checkpoints, Optimizer-/RNG-Zustände, tatsächlich ausgeführter Trainer, Konfiguration und Messwerte.
- export/pilot_02/: Inferenzcheckpoint und SHA-256.
- curriculum_v2/manifest.json und ATTRIBUTION.md: Datenquellen, Lizenz, Filter und Split-Hashes.
- reports/human_comparison_02.json und after_pilot_02.json: Verluste und unveränderte rohe Modellantworten.
- reports/pytest.xml und live_http_02.json: ausgeführte Prüfungen.

Weitere Schritte auf demselben kleinen Korpus sind aufgrund des Early-Stoppings nicht die nächste sinnvolle Verbesserung. Neral benötigt erheblich breitere, geprüfte Sprachdaten; ein nützlicher allgemeiner Chatbot ist mit diesem Stand noch nicht erreicht.
