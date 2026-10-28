# Arctus 350M – Bewertungs- und Freigabesicherheit (28. Oktober 2026)

**Status: Codeänderungen und Offline-Tests; KEIN Modelltraining und KEIN Qualitätszuwachs durch neue Gewichte.** Native Arctus-350M-Forschung auf Branch `arctus350-chat-cloud-prep`, keine Qwen-Modellgewichte.

## Warum dieser Patch wichtig ist

Die frühere Arctus-SFT-Forschung meldete, dass ein **1.200-Schritt-SFT-Pilot** die Validierungs-Answer-NLL von 3,163 auf 1,736 senkte, **während der eingefrorene 30-Fragen-Capability-Score von 3 auf 2 sank**. Der Checkpoint wurde zurecht zurückgewiesen. Das zeigt: niedrigere Trainingsverluste ≠ besserer Chatbot.

Im bisherigen Code gab es zusätzliche Auswertungsrisiken:

| Schwachstelle | Neuer Schutz |
|---|---|
| Doppelte Report-IDs überschrieben zuvor stillschweigend eine Antwort | Duplikat oder fehlende Antwort → `ValueError`, keine Score-Datei |
| Ungültige/fehlende Felder und leere `contains_all`-Listen konnten unzuverlässig gewertet werden | Strikte Schema- und Antworttypen, keine Vakuum-Treffer |
| Zwei abweichende eingefrorene Testsätze konnten bei gleicher Kategorienanzahl verglichen werden | SHA-256-Fingerprint der **exakten Case-Inhalte**, nicht nur des Dateinamens |
| Kategorierückgang um beispielsweise 10 Prozentpunkte durfte als `capability_no_regression` passieren | **Jeder negative Kategorien-Delta** blockiert jetzt `capability_no_regression` |
| `bool("false")`/Integer 1 konnten als wahrer Release-Gate-Wert durchgehen | Freigabechecks akzeptieren ausschließlich das echte JSON-Boolean `true` |
| fehlende exakte Instruktionsmetrik galt als `exact_instruction_no_regression` | fehlende Baseline oder Kandidatenmetrik → Gate `false` |
| NaN, Infinity oder widersprüchliche Scores konnten unbemerkt weitergegeben werden | endliche Werte, 0–100 %, Kategorie- und Gesamtzahlen müssen passen |
| beliebige positive Modellberichte mit falscher Fallzahl konnten zur Review-Freigabe führen | Release-Gate fordert exakt 30 Fälle sowie gültigen Suite-Fingerprint |

**ACHTUNG:** Das sind Software-Sicherheits- und Validitätsverbesserungen, kein kryptografischer Beleg, dass das Modell die richtige GPU durchlaufen hat. Ein von Dritten vollständig gefälschter Report bleibt ohne signierte Ausführungsprovenienz möglich. Der Checker autorisiert auch nach bestandenem Gate weiterhin **keinen automatischen Merge, kein Deployment und keine Modell-Promotion**: `automatic_release=false`.

## Kompatibilitätsgrenze

Die Vergleichssemantik ist jetzt **strenger**:
- Alte Score-Dateien **ohne `suite_sha256`** dürfen nicht in neue Model-Promotion-Entscheidungen übernommen werden. Baseline und Kandidat müssen unter derselben eingefrorenen Suite erneut ausgewertet werden.
- Vorher tolerierte kleine negative Kategorien-Deltas sind jetzt explizite Ablehnungsgründe.
- Tests ohne wirklich gemessene `exact_instruction_pct` dürfen keinen **unbedingt positiven** Generation-Health-Vergleich mehr ergeben.
- Die 30-Fragen-Suite ist nur ein kleines internes Diagnostikum, keine komplette Intelligenzbewertung; bei acht oder weniger Fällen pro Kategorie ist die statistische Unsicherheit hoch.
- Für NIS2 werden separate, unveränderte Benchmark- und Checkpoint-Nachweise benötigt. **Hier wird kein NIS2-Wert erzeugt.**

## Reproduktion – funktioniert ohne deine RTX 4070

```bash
python -m unittest tests.test_chat350_capabilities tests.test_arctus350_release_integrity tests.test_arctus350_generation_integrity -v
python -m py_compile frontier350/score_chat350_capabilities.py frontier350/chat350_generation_metrics.py frontier350/release_check350.py
```

Zusätzlich wird im neuen CI-Workflow die vollständige vorhandene native 350M-Chat-Unittest-Sammlung erneut ausgeführt. Die Tests enthalten **nur simulierte Mini-Daten**, keine echte 2B-Checkpoint-Inferenz und keine Trainingsergebnisse.

## Sinnvolle nächste Arbeit bei eingeschaltetem RTX-4070-PC

1. **Checkpoint-Inventur**: zuletzt tatsächlich vorhandene 2B-Gewichte, genauer `processed_tokens`-Zähler, `model_state`, Optimizer- und RNG-State, kompatibler eigener BPE-Tokenizer samt SHA-256, gesperrte Train-/Val-Splits, aktuelle Run-IDs. Bestehenden 1B-SAFE-Checkpoint unbedingt unberührt lassen.
2. **Echte Baseline** auf identischer, noch nicht zum Training verwendeter Suite: Val-Loss, EOS-Abschluss, Wiederholungen, freie deutsche Antworten, Instruktionen und 30 Fähigkeiten; Bericht mit exaktem Checkpoint und Tokenizer verknüpfen.
3. **Abgleichen**: die bekannten Probleme (EOS 0 %, starke Trigram-Wiederholung, Instruktionen 0 %) stammen aus historischen 1B-Reports, nicht aus einer heute erneut gemessenen 2B-Evaluation. Ohne aktuellen Messlauf keine Verbesserung behaupten.
4. **Gezielter Posttraining-Pilot** auf abgetrennter Kopie des 2B-Checkpoints, mit nachvollziehbar zugelassenen Konversationsdaten; kein Überschreiben der Foundation-/SAFE-Linie, wenn Validation-NLL fällt, aber Fähigkeiten abnehmen.
5. **Promotion nur bei grünen Originalmodell- und Sicherheitstests** plus menschlicher Freigabe. Qwen3.5 als deutlich ausgewiesene Übergangs-Chat-Software nicht mit Arctus-Benchmarks vermischen.

## Aktuelle Quellen-/Datensichtung auf Hugging Face (keine Downloads)

- [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) ist ein **externes, unabhängig trainiertes Referenzmodell** vergleichbarer Größe. Als Literatur-/Methodenvergleich geeignet, NICHT als Quelle vortrainierter Gewichte für Arctus.
- [Smol-SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smol-smoltalk) nennt Apache-2.0-Metadaten und wurde für Unter-1B-Instruktionsmodelle verwendet. **Noch nicht zur Verwendung freigegeben**: einzelne Unterquellen, Dialogformate, PII/Urheberrecht und Benchmark-Kontamination prüfen.
- [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) führt laut Metadaten auch deutsche menschliche Konversationen und Apache-2.0-Metadaten. **Noch nicht importiert oder admitted**; Quellen-, Text- und Split-Audit bleibt nötig.
- [SmolLM-Corpus](https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus) besitzt ODC-BY-Metadaten auf Sammlungsebene, inklusive synthetischer Bestandteile; kein automatischer Datenimport, jedes Subset separat prüfen.

## Budget und Betrieb

Zusätzliche Geldkosten **0 €**; Obergrenze **25 € insgesamt**, ohne Stromkosten, keine Freigabe für neue Ausgaben. Alle GitHub-Änderungen bleiben im **Draft-PR**; lokale RTX-4070-Gewichte sind unverändert.
