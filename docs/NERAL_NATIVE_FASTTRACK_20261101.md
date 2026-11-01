# NERAL Native Fast-Track – eigenständiges Modell, nicht Qwen-Fine-Tuning
**Beginn:** 1. November 2026  
**Status:** Forschungs-/Trainingsvorbereitung. Keine Gewichtsänderung, kein Modell-Qualitätsgewinn behauptet. Draft PR #209.

## Ziel und Grenze

Ein NERAL-Modell, das **mit den bereits vorhandenen eigenen Arctus350-Gewichten und dem eigenen 16K-BPE-Tokenizer** läuft, ohne Qwen-Gewichte oder Ollama bei der Inferenz. Qwen bleibt optional als Hilfsassistent und – nur mit korrekter Datenherkunft – als Teacher für zusätzliche Beispiele. **Keine Umschreibung von Qwen-Gewichten oder Fine-Tuning von Qwen als angeblich eigenständiges NERAL.**

Die eigene ~350M-Architektur, Train-/SFT-Skripte, Datensatzvalidierung, Evaluatoren und experimentelle Flask-Anbindung existieren bereits. Frühere unabhängige V3-Rechenversuche ergaben **0/32**. Der bestehende Test gilt nach mehrfacher Auswertung als Entwicklungsdiagnostik, nicht als unberührter Finaltest.

## Was jetzt neu ausführbar ist – ohne GPU und Qwen

**1.024 zusätzliche originäre bilinguale Instruktionsbeispiele** aus `frontier350/arctus350_instruction_curriculum_v1.py`: zwei nachrechenbare Aufgabenarten (*zweites Wort wählen*, *drei Zahlen aufsteigend sortieren*), 256 semantische Gruppen je Art, beide Sprachen im gleichen Split. Labels werden aus den eigenen Aufgaben deterministisch erzeugt, nicht von Qwen kopiert. Neue private Texte, Nutzer-Chats, Webdaten, Gewichte oder API-Schlüssel werden nicht gelesen.

Lokal, ohne Training:
```sh
python -m unittest -v tests.test_arctus350_instruction_curriculum_v1 tests.test_arctus350_teacher_gate
python -m frontier350.arctus350_instruction_curriculum_v1 --out /PFAD/ZU/EINEM/NEUEN/ORDNER/native-instruction-v1
```
Die Ausgabe enthält `train.jsonl`, `val.jsonl`, `test.jsonl` im überprüften NeralTalk350-Format und `metadata_report.json` mit SHA-256/Anzahlen. Existierende Ausgabepfade werden nicht überschrieben. Die generierten VAL/TEST-Splits sind **nur interne Entwicklungs-Holdouts**, keine externe Blind-Evaluation und kein Nachweis allgemeinen Weltwissens.

**Tests / nachgewiesene Umsetzung:** [TDD-RED Teacher 41337082797](https://github.com/Cola968/neral-helios/actions/runs/41337082797) – neue unabhängige DE/EN-Aufgaben zunächst nicht akzeptiert. **GREEN** [Teacher 41337160145](https://github.com/Cola968/neral-helios/actions/runs/41337160145) 14/14, [Research 41337160053](https://github.com/Cola968/neral-helios/actions/runs/41337160053) 60/60, Syntax [41337160071](https://github.com/Cola968/neral-helios/actions/runs/41337160071). Zusätzlicher TDD-RED [41337388168](https://github.com/Cola968/neral-helios/actions/runs/41337388168) vor eigenem Generator; **GREEN** [Teacher/Native 41337552007](https://github.com/Cola968/neral-helios/actions/runs/41337552007) 14+3 Tests und [Research 41337551973](https://github.com/Cola968/neral-helios/actions/runs/41337551973) 60 Tests, Syntax [41337552024](https://github.com/Cola968/neral-helios/actions/runs/41337552024). Die Ergebnisse belegen Daten-/Codeintegrität, **nicht** bessere Arctus-Gewichte.

## Beschleunigter, bedingter Sprint

| Zieltermin | Konkrete Lieferung | Harte Abnahme und Voraussetzung |
|---|---|---|
| **01.11.2026 – erledigt** | SFT-Originaldatensatz mit 1.024 Zeilen, DE/EN-Referenzprüfungen und TDD-CI | 17/17 Teacher-/Generator-Tests + 60 Forschungsintegrations-Tests bestanden |
| **03.11.2026** | Inventur des ORIGINALEN Arctus350-Checkpoints/16K-Tokenizers; Metadaten und lokale Datenpfade sicher prüfen | Quelle/Tokenizer SHA, Basismodell nachweislich unverändert, >1B ursprüngliche Pretrain-Tokens für `train_chat350.py` erforderlich. Wenn Originaldateien fehlen: Stop |
| **07.11.2026** | Neue, gegen das geplante Training abgeschirmte **unveröffentlichte** Native-Eval mit DE/EN, Instruktion, OOD-Mathe, EOS/Repetitionen, Grounding | Mindestens 64 getrennte Fälle, fixer Bewertungsvertrag, keine Tests im Teacher-/Train-Split; kein Tool-/Qwen-Erfolg als native Leistung |
| **10.–14.11.2026** | Maximal ein kurzer SFT-A/B-Pilot `baseline` gegen `baseline + geprüfte Instruktionen` auf **bereits vorhandener lokaler GPU** | **Nur nach gesonderter ausdrücklicher Erlaubnis** für GPU-/Gewichtsänderungen; vorher VRAM-Check, Checkpoint-Backup, Budget/Abbruchbedingungen; keine Cloud-GPU |
| **15.–21.11.2026** | Gefrorene A/B-Auswertung: OOD-Instruktionsgenauigkeit, Mathe, EOS, repetitive Ausgaben und Wissens-/Replay-Regressionsfälle | Kandidat ausschließlich bei tatsächlichem, breitem Vorteil ohne kritische Regression zur manuellen Promotion vorschlagen |
| **bis 30.11.2026 – konditional** | **NERAL Native 0.1** mit eigenem Modell als experimentellem Modus in Helios | Nur wenn echte Native-Tests/Hardware-Abnahme passen; sonst funktionierendes, transparent schwaches Arctus-Prototypmodell beibehalten |

**Nicht versprochen:** GPT-3-/Qwen3.5-Qualität im November, kostenloses 1B-Pretraining in wenigen Wochen oder ein automatischer Übergang von 350M auf 9B-Niveau. Eine RTX4070/12GB eignet sich für begrenzte Versuche; Vortraining im Milliardenparametermaßstab erfordert eine eigenständige Rechen- und Datenkalkulation.

## Technische nächste Schritte

1. **Keine Neuentwicklung der Architektur.** Vorhandene `frontier350/model350.py`, `train_chat350.py`, `arctus350_paired_evidence.py`, `arctus350_supervision_audit.py` und `chat350_data.py` weiterverwenden.
2. **Daten nicht blind mischen.** Alle Trainingsbeispiele müssen richtige Lizenz/Herkunft, unabhängige Labelprüfung, Split-/semantische Deduplizierung, passende native BPE-Länge und Assistant-Targetmasken haben. Das 1.024-Zeilen-Set ist nur ein Teilbereich; kein Ersatz für hochwertige dialogische Daten.
3. **SFT-Experiment begrenzen.** Für die vorhandene RTX4070 keine automatische Vollpretrain-Schleife; vor Training VRAM/Gradienten-/Durchsatz-Baseline. Original-Optimizer/Checkpoint niemals überschreiben. Reale Kosten neuer Cloud-Dienste bleiben null ohne Nutzerfreigabe.
4. **A/B-Gate vor Promotion.** Keine Verbesserung anhand Loss, Train-Gememorisierung oder SOLVE-Werkzeugtreffern behaupten. Nur unabhängige *native* Generationsmessung zählt; bei Versagen Kandidaten verwerfen.
5. **Qwen-Kern weiter nutzbar lassen.** Existierender Helios+Qwen-Release-Draft #215 bleibt unangetastet. Native Forschung getrennt in #209; keine automatischen Merges oder Gewichtsänderungen.
