# Arctus350 – native SFT label integrity (29. Oktober 2026)

**Scope:** Diese Ergänzung verbessert die Qualitätssicherung vor dem nächsten eigenen 350M-SFT-Lauf. Sie verändert **keine** Gewichte, führt **keine** Inferenz aus und belegt **keinen** Fortschritt bei GPT-3, Giotto oder Arctus350. Modell- und Datenfamilie: Original Arctus350, 349.825.152 Parameter, gesperrter Original-16K-BPE; nicht Qwen.

## Welches reale Problem wird gelöst?

Der bestehende `frontier350/chat350_data.py` prüfte bereits strukturelle Integrität und Cross-Split-Prompts. Die Trennung fehlender Antworten innerhalb eines Splits von echten und unterschiedlichen, aber **widersprüchlichen numerischen Referenzen**, sowie die Prüfung direkt beweisbarer Additionen und anderer kleiner arithmetischer Aufgaben fehlten bisher.

**Neu:** `frontier350/arctus350_label_integrity.py` prüft eng definierte vollständige Direktaufgaben in Deutsch/Englisch: Addition, Subtraktion, Multiplikation und Division (nur kleine, nicht negative Operanden), begrenzte Schreibweisen mit Dezimalkomma und rein numerischen Antworten. Berechnung über Python `Fraction`, **ohne `eval` / `exec` / externes Modell**.

- `Wie viel ist 57 plus 68? Gib nur die Zahl aus.` → nur `125` als numerisch korrekt akzeptiert; `126` macht die Daten **ungültig**.
- Andere numerische Antworten für denselben vollständigen Gesprächskontext → `conflicting_numeric_contexts`.
- `4,50` und `4.5` bei `9/2` sind mathematisch gleichwertig.
- Erklärende / freie Textantworten, komplizierte Textaufgaben und nicht abgedeckte Mathematikformate bleiben explizit **ungescort**; sie werden **nicht** als richtig gezählt.
- Die Prüfung bearbeitet `train` und `val`; das **TEST**-Split bleibt für die eigenständige Auditfunktion **ungeöffnet**. Der bisherige Trainer behält seine eigenen Split-Integritätsprüfungen und erst nach Kandidatenselektion vorgesehene Test-Evaluation.
- Ausgabe ausschließlich aggregierte Zählwerte, Eingabedatei-SHA-256 und eindeutiger Status. **Keine User-Prompts, Antworten, Modelloutputs oder private Zeilen-IDs**.

## Direkt nutzbar auf deinem PC (wenn lokale Daten verfügbar sind)

```powershell
python -m frontier350.arctus350_label_integrity --data "C:\pfad\zu\neraltalk350" --out "C:\pfad\zu\neuem-label-audit.json"
```

Voraussetzung sind lokale `train.jsonl` und `val.jsonl` im bestehenden NeralTalk350-Schema. Kein Checkpoint, keine GPU und keine Hugging-Face-Downloads nötig. `test.jsonl` wird von diesem Offline-CLI **nicht geöffnet**. Der Ausgabepfad muss neu sein – alte Nachweise werden nicht überschrieben. Bei widersprüchlichen oder falsch berechneten direkten numerischen Labels beendet sich das CLI mit Exitcode **2**; sonst **0**. Auch bei 0 überprüfbaren Mathebeispielen lautet der Status nur `PRETRAIN_GATE_PASS_LIMITED_SCOPE`, **nicht** `DATASET_CERTIFIED`.

Im echten `frontier350/train_chat350.py` wird derselbe Fail-Closed-Check jetzt direkt nach der bisherigen Prompt-Holdout-Prüfung und **vor dem Laden eigener Gewichte, Tokenizer und Start des Optimierers** aufgerufen. Der aggregierte Report wird im SFT-Runmanifest erfasst. Die GPU-Nutzung bleibt unverändert und startet nicht allein durch diese Codeänderung.

## Messgrenzen und Risiken

- Keine Prüfung allgemeiner Weltfakten, unstrukturierter Erklärungen oder fortgeschrittener Mathematik. Auch lizenzrechtliche Eignung, Hidden-Duplicates und semantische Kontamination sind **nicht** damit bewiesen.
- Ein grüner Fixture-Test ist **kein gemessener Gewinn am echten 350M-Modell**. Die vorhandenen realen GPU-Daten aus Issue [#141](https://github.com/Cola968/neral-helios/issues/141) bleiben unverändert.
- Die neuen CPU-Tests verwenden nur erfundene Daten und zählen Testabdeckung, nicht wissenschaftliche Modellintelligenz.
- **Keine** Testsatz-Promotion und keine neuen SFT-Trainingsläufe ohne explizite Freigabe auf lokaler RTX 4070.

## Quellenrecherche auf Hugging Face (nur Metadaten)

- [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M): Referenzmodell ~362M Parameter, Apache-2.0, **keine** Übernahme fremder Modellgewichte.
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath): deklarierte Dataset-Lizenz `odc-by`, mathematisches Vortraining; bereits unter `frontier350/source_lock.json` mit Revision erfasst. Eine Dataset-Metadatenlizenz ersetzt nicht die Detailprüfung bei Weiterverwendung.
- [OpenAssistant OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2): deklarierte Dataset-Lizenz `apache-2.0`, deutsch/englische mehrsprachige Gespräche, bereits separat im Repo verarbeitet. Kein Download in diesem PR.

## Echter TDD-Nachweis auf GitHub

1. RED: neuer Test schlug wegen fehlender Implementierung fehl: [40862074669](https://github.com/Cola968/neral-helios/actions/runs/40862074669).
2. GREEN: 12/12 gezielte ursprüngliche Tests: [40862237060](https://github.com/Cola968/neral-helios/actions/runs/40862237060).
3. CLI-RED: zwei neue Tests schlugen bis zur Implementierung fehl: [40862514876](https://github.com/Cola968/neral-helios/actions/runs/40862514876).
4. CLI-GREEN und vorliegender 15/15-Satz: [40862619473](https://github.com/Cola968/neral-helios/actions/runs/40862619473).

**Review-Gates:** Dieser PR hängt gestapelt von PR #189 und den vorherigen Forschungsgates (#188, #187, #186) ab; nicht separat in `main` mergen. Der private 2B-Checkpoint bleibt offline. Direkte Zusatzkosten: **0 €**; keine Cloud-GPU-Aufträge oder Abos, kein automatischer Merge/Release.
