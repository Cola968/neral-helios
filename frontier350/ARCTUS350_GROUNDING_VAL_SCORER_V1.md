# Arctus350 Grounding VAL Score v1 – überprüfbarer, aber **nicht attestierter** Modelloutput

**Datum:** 29.10.2026 · Erweiterung des [Grounding-Datensatz-PR #184](https://github.com/Cola968/neral-helios/pull/184) · Forschungsvorhaben [#179](https://github.com/Cola968/neral-helios/issues/179)

## Warum dieses Modul?

Neue Daten und ein sinkender SFT-Loss beweisen nicht, dass Arctus350 tatsächlich besser antwortet. Dieser **VAL-only**-Scorer ist das kostenlose, offline auswertbare Gegenstück zum synthetischen DE/EN-Grounding-Curriculum: Er misst Genauigkeit **pro Sprache und Fähigkeit** sowie Fehler beim Nichtwissen.

**Erfolg eines CPU-Unit-Tests beweist nur die Korrektheit des Auswerters, nicht die Qualität des Modells.** Alle aktuellen Beispielantworten der Tests stammen aus Fixtures, nicht aus einer 350M-Gewichtedatei.

## Format, bewusst klein

`manifest.json` – Angaben, die der tatsächliche spätere lokale Inferenzlauf belegen müsste:

```json
{
  "schema": "arctus350-reported-inference-v1",
  "claimed_model": "native-arctus350",
  "checkpoint_sha256": "<64 lowercase hex characters of an independently verified local checkpoint>",
  "tokenizer_sha256": "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd",
  "decoding": {"temperature": 0, "max_new_tokens": 48}
}
```

`val-observations.jsonl`: eine Zeile je **tatsächlich beobachteter** VAL-Inferenz, jeweils genau `{"id":"cw-...","output":"..."}`. Alle im synthetischen `make_splits()["val"]` enthaltenen IDs müssen genau einmal vorkommen; **TEST-IDs werden abgewiesen**. Das Ausgabeformat enthält keine Prompts, Rohantworten oder Pfade und vermeidet stille Datenausleitung.

```bash
python -m unittest tests.test_arctus350_grounding_val_scorer tests.test_arctus350_grounding_curriculum_v1 -v
python -m frontier350.arctus350_grounding_val_scorer --val-predictions val-observations.jsonl --manifest manifest.json --out aggregate.json
```

Der CLI-Befehl führt **keine Inferenz** aus. Er wertet nur vorhandene, menschlich belegbare Beobachtungen aus. Es gibt **keine** Option, das versiegelte TEST-Set zu evaluieren. Ein vorhandener Bericht wird niemals überschrieben. Output: Gesamt- und Kategorien-Ergebnisse, bekannte Fakten fälschlich verweigert (`false_abstentions`), bei fehlender Masse erfundene Angaben (`guessed_missing_mass`).

## Entscheidende Vertrauensgrenze

- Ein SHA-256-String im von Menschen gelieferten Manifest ist **kein** Beweis für einen echten Modellaufruf. Dieser Scorer überprüft weder Gewichte noch Tokenizer-Dateien noch Inferenzlogs, obwohl er das **vorgegebene** Format strikt validiert.
- Der Bericht ist deshalb **immer** `UNVERIFIED_REPORTED_OUTPUTS`, `native_generation_verified=false`, `release_eligible=false`, `model_intelligence_improved=false`. Selbst 100 % aus synthetischen Test-Fixtures ist kein Arctus-Erfolg.
- Diese closed-world-Aufgaben benutzen wenige Schablonen und künstliche Objekte. Eine bessere VAL-Accuracy ist ein **eng definierter Grounding-Erfolg**, kein GPT-3-, Giotto- oder Allgemeinwissensnachweis.
- Keine automatischen Modelluploads, Downloads, HF-Jobs, GPU-Starts, Gewichtsänderungen, `resume`-Trainings oder Veröffentlichungen. Zusätzliche direkte Kosten 0 € (nur GitHub-CI im verfügbaren Kontingent).

## Nächste echte Evidenz nach Rückkehr des PC

1. Auf der lokalen RTX4070 einen ohnehin erlaubten und kontrollierten **read-only** Vorher-Lauf mit dem echten 350M-Checkpoint, seinem originalen 16K-BPE und diesem VAL-Set durchführen; Tokenizer und Checkpoint separat kryptografisch prüfen.
2. Erst bei genehmigtem A/B-Training: dieselben VAL-IDs **unter identischen** Einstellungen auswerten, dazu Regression auf normale Deutsch-/Englisch-Dialoge, Mathematik, Code, EOS, Wiederholungen und übermäßige Verweigerung.
3. Unabhängigen TEST nicht bei der Optimierung ansehen. Die bestehenden [Paired-Model-Gates aus PR #180](https://github.com/Cola968/neral-helios/pull/180) bleiben die Pflicht vor jeder möglichen Promotion. **Keine** Parameter- oder Modellgüteversprechen ohne reale Ergebnisse.

**Integration:** Neuer Draft-PR baut auf PR #184 auf und erweitert nur dessen Research-Dataset, ohne parallel laufende PRs #180–#183 oder die NERAL-Orbit-/Chat-Zweige zu überschreiben.
