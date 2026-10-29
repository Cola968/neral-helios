# Arctus350 – gemeinsamer Forschungskandidat (29. Oktober 2026)

**Status:** `INTEGRATED_CPU_VERIFIED`, **NICHT** `NATIVE_MODEL_IMPROVED` oder `READY_FOR_RELEASE`.

**Hauptziel:** schneller zu einer wirklich besseren eigenen 349.825.152-Parameter-KI kommen, indem wir vorhandene sichere Forschungsteile gemeinsam prüfen, statt sechs unverbundene SFT-Experimente zu starten.

## Ein gemeinsamer Kandidat statt sechs getrennte Arbeiten

| Herkunft | Verpflichtend eingefrorener Head | Was übernommen wurde | Was nach wie vor fehlt |
| --- | --- | --- | --- |
| [#180](https://github.com/Cola968/neral-helios/pull/180) | `e531529e750e3d02dd93c9a32460b234542ba5a4` | Gepaarter, hashgebundener VAL-Vergleich und bestehende Tests | Echte lokale Rohmodell-Beobachtungen |
| [#181](https://github.com/Cola968/neral-helios/pull/181) | `424f04ca0d40bdb70cb0a863791eda7fe9ab57d8` | Optionaler Compact-SFT-Sampler und originale Train-CLI-Anbindung | Echte GPU-Zeiten, OOM-/Qualitäts-A/B |
| [#182](https://github.com/Cola968/neral-helios/pull/182) | `3ba8adef89fb0e17c8f0eef3b7c3cd8e72b147f2` | Semantischer Arithmetik-Leak-Audit und nicht überschreibender v1-Derivatexport | Alle sonstigen Train/VAL/TEST-Datensätze auf unbekannte Leaks prüfen |
| [#183](https://github.com/Cola968/neral-helios/pull/183) | `10824e43250014fe8a8eb416cb755e63886f5279` | Assistant-Only-Supervisions-/EOS-Audit | Echten lokalen 16K-Tokenizer und das vollständige SFT-Mischkorpus prüfen |
| [#184](https://github.com/Cola968/neral-helios/pull/184) | `1f1948fbe9a5753d8209512af38c50c3aa8274a0` | 360 fiktive Entitäten / 2.160 deterministische DE/EN-Grounding-Beispiele | Nachweis, dass das eigene Modell auf unbekannte Eingaben generalisiert |
| [#185](https://github.com/Cola968/neral-helios/pull/185) | `b0240b76421f3e873a34413db501175aa8961a08` | Vollständiger VAL-only-Scorer mit getrennten Unterkategorien | Direktes natives Inferenz-Adapterprotokoll |

**Integration:** [Draft PR #186](https://github.com/Cola968/neral-helios/pull/186) auf `audit/arctus350-20261028-fresh-carry-baseline`. Die ursprünglichen PRs und ihr Git-Verlauf bleiben unverändert. 22 ausgewählte disjunkte Git-Blob-Pfade wurden auf Basis eines festen Parent-Commits zusammengeführt; alle sechs separaten, potenziell redundanten CI-Workflow-Dateien wurden **nicht** übernommen. Stattdessen gibt es **eine** gemeinsame CPU-CI.

### Echte unabhängige Integrationsevidenz

- **TDD RED:** [GitHub Actions 40850429490](https://github.com/Cola968/neral-helios/actions/runs/40850429490), `60 Tests`, davon **1 erwarteter Regressionstest fehlgeschlagen**: Der vorherige VAL-Scorer klassifizierte sämtliche falschen fehlenden Masse-Antworten als »erfundene numerische Masse«.
- **TDD GREEN:** [GitHub Actions 40850758651](https://github.com/Cola968/neral-helios/actions/runs/40850758651), **60 von 60 CPU-Tests bestanden**, inklusive ursprünglicher Forschungs-Unit-Tests und neuer Score-Klassifikation.
- In derselben CI wurde `python -m frontier350.arctus350_grounding_curriculum_v1` wirklich ausgeführt: **360 Entitäten, 2.160 Zeilen, Entity-Splits disjunkt**.
- In derselben CI wurde `python -m frontier350.arctus350_math_semantic_audit --generated-audit` wirklich ausgeführt: v1-Legacy-Numerik-Overlap wird erkannt; v2/v3-Generatoren sind im geprüften Bereich disjunkt.
- Keine echten Modellgewichte geladen, keine neue native Generation, keine Datasets in öffentlicher Cloud gespeichert, keine bezahlten HF-Jobs.

**Korrektur der Metrik:** Für `mass_missing` ist ein inhaltlich falsches »weiß ich nicht« **kein erfundener Messwert**. Nur explizite numerische Masse-Antworten (Zahl mit optional `g/kg`) werden als solche gezählt. Ungenaue Angaben bleiben selbstverständlich im Exact-Match-Fehlerzähler; False-Abstention und Accuracy sind eigene Kennzahlen.

## Was NERAL tatsächlich schneller zur ersten guten KI führt

1. **Keine weitere ungezielte SFT-Runde.** Die eigenen 1B-, 2B- und besten Chat-SFT-Gewichte dürfen nicht durcheinandergeraten; tatsächliche lokale Hashes gemäß [Issue #141](https://github.com/Cola968/neral-helios/issues/141) stehen über jeder Namensbezeichnung.
2. **Vor GPU-Start:** mit dem originalen lokalen 16K-BPE den Supervisionsaudit auf der vorhandenen Train/VAL/TEST-Mischung ausführen; wenn zu wenige korrekte Assistant-Targets oder unerwartete EOS-Masken auftreten, Datenformat zuerst reparieren. Die Codeimplementierung allein ist noch **kein echter lokaler Audit**.
3. **Kleinster GPU-Entscheidungstest:** Eine eingefrorene native DEV-Suite mit gleichbleibender Prompt-ID, Sampling, Kontext und Tokenizer erst auf 1B, 2B und historischem SFT **read-only** ausführen. Die bisherige 0/32-v3-Arithmetik nicht durch künstliche Tool-Ergebnisse ersetzen.
4. **Erst danach:** kontrollierter kurzer SFT-A/B mit maximal einer neuen Variable (z. B. Compact-Batching, alternativ geschlossenes Grounding-Curriculum) und gleichem effektivem Token-/Optimizer-Budget. Keine Auto-Promotion: benötigt geparter Accuracy-Gewinn ohne kritische Kategorie-, EOS- oder Wiederholungsregression.
5. **Produkt und Modell getrennt:** NERAL Chat / Orbit kann schon ohne fertiges Arctus350 nutzbar werden. Tool-Rechenergebnisse und Qwen-Ausgaben gelten nicht als native Arctus350-Modellmetriken.

## Einfache kostenlose CPU-Kontrolle des Forschungsbündels

```sh
python -m unittest -v \
  tests.test_arctus350_paired_evidence \
  tests.test_arctus350_paired_evidence_bridge \
  tests.test_arctus350_carry_baseline_audit \
  tests.test_sft_batching350 \
  tests.test_arctus350_compact_sft_batches \
  tests.test_arctus350_math_semantic_audit \
  tests.test_arctus350_math_sanitization \
  tests.test_arctus350_sanitized_export \
  tests.test_arctus350_supervision_audit \
  tests.test_arctus350_grounding_curriculum_v1 \
  tests.test_arctus350_grounding_val_scorer \
  tests.test_arctus350_integrated_scoring_regression
```

Der echte lokale SFT-Audit erfordert den **bereits vorhandenen originalen** Tokenizer und zugelassenen bestehenden Korpus. Der 2B-Checkpoint bleibt lokal; die Cloud-Version enthält keine Originalgewichte.

## Risiken, Freigabe und Kosten

- Vollständige Regressionstests von `train_chat350.py` mit **echtem Training** stehen aus. Das kombinierte CPU-Gate beweist den Import/Algorithmus-Test, aber keine Trainingsstabilität auf der RTX4070.
- Eigenständiges 350M-Modell auf GPT-3-/Giotto-Niveau ist **nicht** belegt. Forschungs-Benchmarks sind keine Performance-Garantien.
- Keine eigenmächtigen GPU-/PC-Jobs, keine Uploads privater Gewichte, kein Merge/Deploy, keine bezahlten HF-Jobs.
- Projektbudget: vorzugsweise 0 € direkte Zusatzkosten; höchstens 25 € **gesamt**, Haushaltsstrom ausgenommen, Kosten nur mit separater ausdrücklicher Zustimmung.
- Weiteres Integrationsrisiko: PR #186 ist auf den bereits bestehenden Audit-Branch gestapelt. Ein grüner PR ist **keine Freigabe zum direkten Merge in main**. Vor produktiver Übernahme den vollständigen Branch-/PR-Abhängigkeitsgraphen prüfen.

**Bewusst ehrlich:** Ein erfolgreiches Bündel spart Review- und Integrationarbeit. Es verbessert die gemessene Intelligenz der **Gewichte** noch nicht. Die nächste echte Capability-Verbesserung muss aus sauber gemessener Modellinferenz bzw. kontrolliertem Training kommen.
