# NERAL-EXT: CPU-Audit von Extraktion, Abbruch und Bewertungsdecke

**Stand: Diagnose des bereits verbrauchten 400er-Tests, keine neue
unabhängige Modellbewertung.** Der Vergleich
[42395097203](https://github.com/Cola968/neral-helios/actions/runs/42395097203)
ist abgeschlossen. Die Rohantworten bleiben unter Windows im privaten
Forschungsverzeichnis; nur Summen verlassen den lokalen Runner.

## Warum der ursprüngliche Score die Fähigkeiten nicht isoliert

Stratum A mit **200 Fällen** bestand aus 100 GSM8K, 50 numerisch
lösbaren MATH-500 und **50 MBPP-Python-Aufgaben ohne ausführbaren Grader**.
Daher darf 2/200 nicht als 1 % richtige Mathematikaufgaben interpretiert
werden. Im Original-400er-Lauf wurden außerdem **126/12/2**
unvollständige Antworten von Qwen / Qwen-Few-Shot / NERAL v5 als leere
Ausgaben gespeichert. Diese Antworten sind **zensiert** und können
nachträglich nicht auf Korrektheit geprüft werden.

Qwens veröffentlichter GSM8K-Score ist unter einer anderen Instruktions-
und Decode-Konfiguration entstanden. Der große Unterschied ist ein
**dringender Kalibrierungsbefund**, aber noch kein Beweis, dass eine
bestimmte Anzahl richtig gelöster Antworten von diesem Grader verworfen
wurde. Insbesondere **nur** eine letzte Zahl blind zu wählen erzeugt
auch gefährliche False Positives (Zwischenergebnisse, Beispiele,
Widerlegungen, Zitate).

## Was das neue CPU-Instrument misst

`scripts/neral_ext_ceiling_audit.py` zeigt pro Variante, Stratum
und Herkunftsdomäne getrennt:

- numerische Aufgaben versus Python ohne funktionale Auswertung
- unvollständige Ausgaben (bei denen die Rohantworten gelöscht wurden)
- vollständig erzeugte Antworten, die der alte Grader als
  richtig/falsch/unbewertbar kategorisiert
- nachvollziehbar extrahierte Endantworten des **konservativen**
  Diagnostik-Parsers: genau ein FINAL-/Antwort-Marker am Schluss,
  ein einzelner `\\boxed{Wert}`-Endwert oder eine eindeutige
  Formulierung wie `The answer is ...`
- exakte Integer-/Bruch-/EUR-Gleichwertigkeit, nachträglich
  zurückgewonnene korrekte Antworten, parsebare falsche und weiterhin
  nicht extrahierbare Antworten

Die neue Auswertung ist **explorativ**. Beide Graderstände bleiben
im Bericht unterscheidbar. Keine Modellausgabe wird für ein Training
verwendet. Die private **20-Beispiele-Stichprobe pro Variante aus A**
enthält Referenzantwort, ursprünglichen Prompt, gespeicherte
vollständige Rohantwort (sofern vorhanden) und Fehlerklasse.
Diese Stichprobe darf **niemals** in GitHub hochgeladen werden.

## Starten (explizit manuell, kein GPU-Modelllauf)

Workflow `.github/workflows/neral-ext-ceiling-cpu-manual.yml`
auf `main` manuell mit genau
`AUDIT_NERAL_EXT_400_CEILING_CPU_ONLY` als `authorize` starten.

Der Runner benutzt ausschließlich die bereits vorhandenen Dateien
des vorherigen erfolgreichen NERAL-EXT-Laufs
`ext-400-42395097203-1` und verlangt den unveränderten
400-Fälle-Hash
`47258a911ce0beb96635a34137a634556f364457487fb18a160a892302124cb1`,
die Original-Adapter-SHA sowie exakt 126/12/2 unvollständige
Antworten. Die Prüfung ist **reine CPU-Auswertung** und benötigt
keine weitere Qwen- oder NERAL-Inferenz. Es werden keine
Trainingsabhängigkeiten installiert.

Auf dem privaten Rechner entsteht ein **neues** Verzeichnis:
`C:\Users\<Windows-Benutzer>\NERAL-QLoRA-Research\v5-r16-480-42378021782-1\ext-400-ceiling-cpu-<RUN_ID>-1`

Darin liegen `scalar.json` und die lokale Reviewdatei
`20-per-variant-A-RAW-LOCAL-ONLY.jsonl`. Über GitHub Actions
wird ausschließlich `summary.json` (aggregierte Zählwerte ohne
Prompts, Referenzen, IDs oder Modellantworten) veröffentlicht.

## Manuelles Review und Schlussfolgerungen

Die private Reviewdatei auf dem **eigenen PC** öffnen und getrennt
dokumentieren:
- Steht in der gespeicherten Antwort eine eindeutige richtige Lösung?
- Welche Parsingform hat der alte Grader übersehen?
- Ist die Antwort bloß abgeschnitten? Falls ja: im alten Lauf
  **nicht rekonstruierbar**, ohne eine neue Vergleichs-Inferenz.
- Löst der Parser eine mehrdeutige Zwischenzahl fälschlich als
  Endantwort auf? Solche Fälle dürfen nicht in `lenient_correct`.

**Für zukünftige Benchmarks** erst auf separaten Entwicklungs-
und Kalibrierungsaufgaben Prompt, Decode, EOS-Kriterien, Tokenlimit
und Grader **vorregistrieren**, mit einem synthetischen positiven
und negativen Kalibrierungssatz prüfen und dann erst eine
**unbenutzte** Test-Suite öffnen. Das Nachjustieren des alten
400er-Graders auf denselben Antworten erzeugt keinen neuen
unabhängigen Qualitätssprung. Python-Code ist weiterhin
`functional_tests_executed=false`, bis die Sandbox fachlich und
sicherheitstechnisch abgenommen wurde.

## CPU-Tests

```powershell
python -m unittest discover -s tests -p test_neral_ext_ceiling_audit.py -v
```

Keine TRAIN- oder ORPO/DPO-Paare werden aus den 400er-Testantworten
gebaut. Die bereits existierende Trainingspipeline
`scripts/neral_e1_pair_miner.py` bleibt auf
separaten Train-/Val-Aufgaben.
