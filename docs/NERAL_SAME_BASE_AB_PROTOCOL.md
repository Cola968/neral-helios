# NERAL vs. Pure Qwen3.5: fairer Dreiarm-Vergleich

Stand: 08.11.2026. **Entwicklungsprotokoll, kein ausgeführter GPU-Benchmark.** Begleitend zu #454, #458, #459, #468 und Forschungs-PR #467.

## Definition der drei Arme

- **A – `raw_qwen35`:** lokales Qwen3.5:9B, festes `measured_nonthinking_2048`-Profil, direkte Modellantwort ohne NERAL-Werkzeug oder Review.
- **B – `neral_high`:** dieselben Gewichte und Inferenzparameter, existierender NERAL HIGH-Reviewpfad ohne neuen Router.
- **C – `neral_high_router`:** dieselben Gewichte mit verifizierter Werkzeugwahl und sicherem Preflight. **Dieser Arm darf erst gemessen werden, wenn die tatsächliche Routerintegration samt Provenienztest funktioniert.** Keine gefälschten Werkzeugaufrufe.

Die identische ursprüngliche Nutzerfrage muss in allen drei Armen verwendet werden; zusätzlich Modell-Digest, Sampling, Tokenbudget, Reihenfolge/Randomisierung und alle System-Prompts lokal protokollieren. Keine GPU-Parallelität auf der RTX 4070.

## Ist der Router nicht längst da?

Teilweise: `neral_high_verified_first.py` versucht bereits strikt begrenzte, lokale Mathematik-/Logik-/Python-Trace-Oracles. Aufgaben außerhalb des unterstützten Bereichs dürfen **nicht** blind an `solve.py` durchgereicht werden. Der 72er-Qwen3:8B-Lauf #42512932846 zeigte HIGH 59/72 statt LOW 32/72, berichtete jedoch 0 Werkzeugantworten. Diese Messung beweist keine Qwen3.5-Systemüberlegenheit.

## Neues CPU-Werkzeug in diesem PR

`scripts/neral_same_base_ab_score.py` wertet ein **privates, separat aufgenommenes** A/B/C-Ergebnisfile aus. Es kontrolliert identische Modell-Digests und Eingabe-Hashes, Paarvollständigkeit, den Status der Antworten, nur erlaubte Tool-Tags, exakte Integer-Endwerte, domänenspezifische Treffer, gepaarte Wins/Losses, McNemar-p, Mittelwert/Median/p95, API-Fehler. Es exportiert nur Summen, niemals private Texte.

```powershell
python -B -m unittest discover -s tests -p test_neral_same_base_ab_score.py -v
python -B -m scripts.neral_same_base_ab_score --private-input C:\private\ab-input.json --aggregate-output C:\private\ab-aggregate.json
```

Das Skript **ruft kein Modell auf** und kann keinen Test selbst starten. Es benötigt einen **separaten, vor GPU-Ausführung zu prüfenden** Live-Sammler und echte private Aufgaben mit unabhängigen Oracles. Seine Integer-Auswertung ist nicht als Code-Debugging-/Projekt-/Tool-Aufgaben-Grader geeignet.

## Zwei Stufen statt falscher Durchbruchsmeldung

**36 private Entwicklungsfälle:** 12 Mathe / 12 Reasoning / 12 Code-Trace, mindestens sechs Familien. Nur technischer Pilot; keine unabhängige Validierung.

**120 gesperrte Aufgaben:** 30 Mathe / 30 Reasoning / 40 Python-Code / 20 Tool-Aufgaben. Code und Zustandsänderungen benötigen unabhängige Sandbox-/Artefaktprüfer, nicht bloß Integer-Endwerte. Die gemischte Release-Auswertung muss entsprechend erweitert werden, bevor sie als Holdout gelten darf.

**Primärziel:** NERAL C schlägt Pure Qwen3.5 A auf bisher unbekannten Aufgaben um mindestens +10 Prozentpunkte, gepaarter p≤0,05, ohne kritische Regressionen; zusätzlich klarer C-vs-B-Mehrwert. Median und p95 separat optimieren, nicht nur Accuracy/Sekunde. Auch bei einem erfolgreichen Entwicklungs-Pilot bleiben alle Freigabe-Flags im Scorer **false**. Kein automatisches Training, kein Modellwechsel, kein Release-Override.

**Offen:** echter Windows-GPU-Sammler, Router-Feature-Gate, unabhängige Code-/Tool-Oracles, VRAM-/Cancel-Tests, erlaubter GPU-Slot und finale Freigabe.