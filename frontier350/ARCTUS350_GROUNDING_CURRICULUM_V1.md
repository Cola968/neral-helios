# Arctus350 Grounded Intelligence – eigene bilingual SFT-Daten v1

**29.10.2026 · Forschungsauftrag [#179](https://github.com/Cola968/neral-helios/issues/179) · Draft [#184](https://github.com/Cola968/neral-helios/pull/184)**

## Warum dieses Experiment?

Der aktuelle eigene Arctus350-Checkpoint (349.825.152 Parameter, originale 16K-BPE) kann noch nicht zuverlässig als universeller Chatbot antworten. Zuletzt waren 0/32 ausgewählte neue Additionsbeispiele auf v3 VAL korrekt. Die vorhandenen SFT-Experimente verbesserten teils EOS und Loss, nicht aber robustes Schlussfolgern. **Weitere unstrukturierte Mathematikbeispiele allein sind daher kein hinreichend belegter nächster Schritt.**

Die neue Hypothese ist bewusst klein und falsifizierbar: *Ein strukturierter, balancierter Anteil aus realitätsfreien, eindeutig lösbaren DE/EN-Aufgaben kann die Fähigkeit fördern, Informationen im Prompt zu lesen, eine fehlende Information zuzugeben und vorgegebene Messwerte zu vergleichen – ohne einfach immer eine Zahl zu raten.*

Das ist ein **Trainingsexperiment**, keine bereits erzielte Modellverbesserung. GPT-3-/Giotto-Gleichwertigkeit wird nicht behauptet.

## Was tatsächlich im Repository existiert

- `frontier350/arctus350_grounding_curriculum_v1.py`: deterministisch generierte fiktive Sensor-Datensätze, keine Internetdaten, kein LLM-Teacher.
- `tests/test_arctus350_grounding_curriculum_v1.py`: überprüft Sprach-/Aufgabendeckung, Konsistenz jedes Labels gegen neu rekonstruierte ID-Referenz, Manipulationsschutz, Disjunktheit aller sechs Aufgabenvarianten eines Objekts zwischen Splits, lokale Manifest-Integrität und Nichtüberschreiben.
- `.github/workflows/arctus350-grounding-curriculum.yml`: isolierte CPU-Tests und Python-Syntax ohne Modellgewichte.
- Umfang: **360 fiktive Entitäten × zwei Sprachen × drei Aufgabengruppen = 2.160 Datensätze**.
- Einheiten und Daten sind ausdrücklich *fiktiv*, nicht reale Physikmessungen. Sprache: `de`, `en`; Aufgaben: `temperature`, `mass_known`/`mass_missing`, `compare`. Antworten jeweils exakt rekonstruiert; **bekannt UND nicht angegeben** vorhanden.
- Entity-ID (z.B. `K-0123`) bildet den Split-Schlüssel. Alle Varianten einer Entität bleiben vollständig in einem Split; TEST wird nicht als Trainingsdatum verwendet.
- Datensatzformat folgt vorhandener `frontier350/chat350_data.py`-Konvention mit `train.jsonl`, `val.jsonl`, `test.jsonl`, `source` und `license`.
- Die exportierte Metadaten-Datei enthält **nur aggregierte Anzahlen und SHA-256**, keine Rohbeispiele.
- Der Export verweigert die Überschreibung eines bestehenden Ausgabeverzeichnisses.

## Daten lokal erzeugen – **ohne** Training

```bash
python -m unittest tests.test_arctus350_grounding_curriculum_v1 -v
python -m frontier350.arctus350_grounding_curriculum_v1 --out ./local_grounding_v1
```

**Nicht** automatisch ausführen oder mit einem SFT-Lauf verbinden, solange die vorhandenen Dataset- und Checkpoint-Gates nicht geprüft sind. Die lokalen Original-SFT-Daten und bestehenden Gewichte bleiben unangetastet.

## Als nächstes vorgeschlagener wissenschaftlicher A/B-Test (noch NICHT durchgeführt)

| Lauf | Rezept | Fragestellung |
| --- | --- | --- |
| A (Kontrolle) | Kontrollierter, lizenzgeprüfter SFT-Mix ohne diese Zusatzdaten | Wie gut ist das bestehende Modell auf neuen Prompt-Datensätzen? |
| B (Experiment) | Exakt dasselbe Token- und Optimierungsbudget; **kleiner Anteil** der neuen synthetischen Entitäten an Stelle eines vorab definierten Anteils vorhandener SFT-Trainingszeilen | Steigt held-out Grounding/Abstention auf unbekannten Entitäten? |

- **Vorab Registrieren:** identische Parent-SHA, Original-Tokenizer-SHA, feste Seeds, identisches Eval-Verfahren, effektive Assistant-Target-Token und Replay-Budget. Im Test keine Vermischung von Werkzeug- oder Qwen-Antworten mit nativer Modellleistung.
- **Schützen:** bestehende versiegelte Benchmarkfragen, die 30er-Suite, NIS2 und diese `test.jsonl` niemals für Hyperparameterwahl verwenden. Die generierte `val.jsonl` ist nur internes DEV; `test.jsonl` bleibt unangetastet bis zur echten einmaligen Schlussauswertung.
- **Abnahmetest:** absolute exakte Trefferzahlen DE/EN je Aufgabengruppe, Antwortlänge und EOS, Wiederholungen, freie DE/EN-Dialoge, Mathematik, Code und Überabstention (`nicht angegeben` obwohl Antwort existiert).
- **Datenrisiko:** Bei zu vielen „nicht angegeben“-Beispielen kann ein Modell pauschal zu oft ablehnen. Deshalb gibt es bewusst bekannte und unbekannte Massewerte. Ein positiver Grounding-Wert allein reicht nicht für Checkpoint-Promotion.
- **Wissenschaftliche Grenzen:** Splits schützen synthetische Entitäten, **nicht** jedes ähnliche Rechen- oder Sprachmuster in fremden Daten. Synthetische Trainingsfälle sind kein Beweis für allgemeine Weltkenntnis oder GPT-3-Niveau. Der 16K-BPE-Tokenizer und die tatsächliche Sequenzlängen-Eignung müssen lokal mit dem echten Tokenizer geprüft werden.
- **Integration mit laufender Forschung:** #180 prüft gepaarte native Evidenz, #181 reduziert optional Padding, #182 auditiert mathematische Split-Leaks, #183 zählt echtes Assistant-Supervisionssignal. Dieses PR ergänzt *neue Lernaufgaben*, ersetzt keine der vier Arbeiten.

## Grenzen und Entscheidung

**Keine GPU, kein HF-Job, keine Modellgewichte geladen oder hochgeladen, kein direktes `main`-Schreiben, kein Merge, keine automatisierte SFT-Fortsetzung.** Direkte zusätzliche Projektausgaben wurden nicht autorisiert; wir benutzen nur vorhandene GitHub-CI-Ressourcen (deren Nutzungs-/Rechnungslimits separat gelten). Sollten später Tests bei eingeschaltetem PC laufen, braucht es die ausdrücklich erlaubten kurzen RTX-4070-Läufe und sichere Temperatur-/VRAM-Grenzen.

**Fertigkriterium dieses PR:** überprüfbare SFT-Datenpipeline + bestandene CPU-Evidenz, **nicht** „KI wurde intelligenter“.
