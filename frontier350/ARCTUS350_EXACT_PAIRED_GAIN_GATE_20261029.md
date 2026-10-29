# Arctus350 Research: Kleine Qualitätsgewinne statistisch absichern

**29.10.2026 · [Issue #179](https://github.com/Cola968/neral-helios/issues/179) · Draft PR #187 · CPU-only**

## Gefundener Auswertungsfehler

Der bisherige gepaarte native Checkpoint-Vergleich `arctus350_paired_evidence.py` konnte schon bei **einem einzigen zusätzlich richtigen Fall auf acht Beispielen** die Entscheidung `REVIEW_CANDIDATE` liefern. Das war keine automatische Modellveröffentlichung, aber auch keine überzeugende Evidenz einer echten Leistungssteigerung.

Das Problem ist besonders relevant, weil die lokalen Arctus350-Experimente stark variierende kleine DEV-Sets und wiederholte Trainingspiloten verwenden. Ein zufälliger Einzelfall-Treffer darf kein Forschungsprogramm bestimmen.

## Neue Entscheidung

`frontier350/arctus350_paired_stats.py` berechnet den **exakten zweiseitigen McNemar-/Binomial-p-Wert** aus den gepaarten discordanten Fällen (neu richtig = `gained`, vorher richtig und jetzt falsch = `lost`). Keine fehleranfällige asymptotische Approximation bei kleinen Stichproben.

Die **vor der Auswertung festgelegte Projektpolicy** (kein wissenschaftlich garantierter Schwellenwert) für einen `REVIEW_CANDIDATE` verlangt gleichzeitig:

1. **Mindestens 32** eindeutige, auf denselben Fragen gepaarte DEV-Fälle.
2. Mehr gewonnene als verlorene Fälle und **exakter zweiseitiger p-Wert ≤ 0,05**.
3. Unverändert **keinen Kategorien- oder EOS-Rückschritt** in `arctus350_paired_evidence.py`.
4. Selbst wenn die Bedingungen erfüllt sind: Das Ergebnis bleibt **nur ein Review-Kandidat**, kein PROMOTE oder Release, denn das Modul verifiziert **weder Modellgewichte noch Prompt-Identität oder Original-Inferenz**.

### Rechenbeispiele – ausdrücklich synthetische Test-Fixtures

| Veränderte richtige Antworten | Gesamt | Exakter zweiseitiger p-Wert | Review |
| --- | ---: | ---: | --- |
| 1 gewonnen, 0 verloren | 8 | 1,0 | `INSUFFICIENT_STATISTICAL_EVIDENCE` |
| 5 gewonnen, 0 verloren | 32 | 0,0625 | `INSUFFICIENT_STATISTICAL_EVIDENCE` |
| 6 gewonnen, 0 verloren | 32 | 0,03125 | `REVIEW_CANDIDATE`, falls alle anderen Gates bestehen |
| 8 gewonnen, 0 verloren | 8 | 0,0078125 | **Trotzdem zu wenige Cases** |
| 8 gewonnen, 2 verloren | 40 | 0,109375 | `INSUFFICIENT_STATISTICAL_EVIDENCE` |

**Alle Zahlen der Tabelle sind rein mathematische Beispiele aus synthetischen Test-Fixtures. Das sind keine Arctus350-Modell-Scores.**

## Statistische Grenzen

- Der exakte Test setzt unabhängige oder geeignet kontrollierte gepaarte Prüfbeobachtungen voraus. Schablonenfamilien, nahezu identische Zahlenaufgaben, Doppelungen und Leakage können die effektive Stichprobe massiv verkleinern.
- Wenn man viele Checkpoints wiederholt gegen dieselben DEV-Daten testet und nur den Sieger auswählt, verliert der nominelle p-Wert seine Fehlerkontrolle. Das **unabhängige versiegelte TEST** darf gerade deshalb nicht bei jeder Experimentauswahl genutzt werden.
- Ein p-Wert belegt **weder praktische Nützlichkeit noch Vergleichbarkeit mit GPT-3, Giotto.ai oder anderen Modellfamilien**.
- `REVIEW_CANDIDATE` erlaubt weder Gewichts-Promotion noch GPU-Training, Datenupload oder Veröffentlichung.
- Die 32er-Minimalgröße ist eine bewusste konservative **Projektschwelle**, keine statistische Powergarantie; viel größere sauber diversifizierte Testmengen sind für verlässliche Aussagen nötig.
- Für eine echte Modellbehauptung müssen Hashes realer Checkpoints lokal geprüft, native Greedy-Ausgaben protokolliert und zusätzlich unabhängige Fragevarianten evaluiert werden. Bis dahin `native_provenance_verified=False`.

## Tests & Reproduzierbarkeit

```bash
python -m unittest tests.test_arctus350_paired_stats tests.test_arctus350_paired_evidence tests.test_arctus350_paired_evidence_bridge -v
```

- **RED**: [GitHub Actions #40854317363](https://github.com/Cola968/neral-helios/actions/runs/40854317363) – absichtlich fehlendes Statistikmodul, Reproduktion.
- **GREEN**: [GitHub Actions #40854579159](https://github.com/Cola968/neral-helios/actions/runs/40854579159) – neuer exakter Paired-Gate-Vertrag mit bestehenden Regressionstests.
- Das bestehende integrierte Research-Workflow bleibt zusätzlicher Gatekeeper.
- Ausschließlich GitHub-gehostete CPU-Berechnungen, keine RTX4070-Auslastung, keine HF-/Cloud-GPU, keine Checkpoint-Änderung, keine zusätzlichen direkten Projektkosten.

## Nächster Schritt

Auf einem **tatsächlich verfügbaren und explizit freigegebenen lokalen Modell** zuerst eine unveränderte, strukturierte 32–100+ Entwicklungs-Fragensuite bei identischen Generationseinstellungen paarweise auswerten. Keine blind wiederholten SFT-Läufe aufgrund sinkender NLL oder einzelner Treffer.
