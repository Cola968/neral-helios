# NERAL dev72 – Oracle-Sperre, Testergebnis und Bewertung

Audit davor: `docs/NERAL_DEV72_ORACLE_AUDIT_20261107.md`, Commit `ad1d4500`.
Oracle-Sperre: Commit `8a1067ea`.
Zweig: `cursor/dev72-independent-oracle-f78c`.
Pull Request: https://github.com/Cola968/neral-helios/pull/365
Diese Änderung bewertet keine 7B-Gewichte. 21/72, 24/72 und 29/72 bleiben die
gesperrten Aggregatzahlen aus Lauf 42409458549.

## Was geändert wurde

`scripts/neral_dev_oracle.py` prüft jede Entwicklungsaufgabe ein zweites Mal,
bevor `build_cases()` oder `cases72()` sie zurückgeben.

- Code: genau ein Python-Fence, AST-Allowlist, dann ein kurzlebiger Prozess
  `python -I -B` mit Zeitlimit 2 Sekunden und, unter Linux, CPU-, Speicher- und
  Dateideskriptor-Limit. Die gedruckte ganze Zahl muss die Goldzahl sein.
  `import`, `open`, `exec` und `__import__` werden abgelehnt, bevor ein Prozess
  startet.
- Übrige Familien: die Zahl wird aus dem Prompttext neu berechnet (Euklid,
  wiederholtes Quadrieren, Münz-DP, Binomialprodukt, geschlossene Teilbarkeit,
  Vorwärtsverkettung für Horn, Breitensuche, drei explizite Wege). Nicht aus
  der Indexformel des Generators.
- Abweichung wirft `OracleMismatch`. Die Suite entsteht dann nicht.

Der AST-Tracer in `neral_agent_python_trace.py` ist unverändert. Er führt
weiter keinen Modellcode aus und unterstützt weiterhin kein `append`. Diese
Sperre gilt nur für Repository-Fixtures.

## Vorher / nachher

Zurückgezogene Formel auf `d7c33cc`, `scripts/neral_v61_new_dev_eval.py`:

`answer = (p + 3*i) + (q + i) + 3` ergibt für i = 0 den Wert 14.

Dieselbe Quelle druckt unter CPython 17, weil `sum(a)` die angehängte 3 enthält
und `len(b)` bei gemeinsamer Liste noch einmal 3 addiert. Die Serie der alten
Formel ist 14, 20, 26, 32, 38, 44. Die Ausführung ist jeweils +3.
`assert_independent_oracles` lehnt ein so gefälschtes Gold ab.
`cases72()` auf diesem Zweig liefert die Ausführungswerte 17, 23, 29, 35, 41, 47.

Das ist keine neue Modellmessung. Die historische Suite von 42409458549 enthält
diese Alias-Aufgaben nicht. Ihre 24 Code-Aufgaben stimmen mit der Ausführung
überein; ein um 1 verfälschtes Gold wird ebenfalls abgelehnt.

Absichtlich falsche `FINAL_ANSWER` über jede der 12 Familien: 66 richtig von 72,
0 unbewertet. Prosa, die die Goldzahl nur enthält: 66 richtig und 6 unbewertet.
Der Parser schreibt falsche Zahlen nicht in Formatfehler um und Prosa nicht in
Treffer.

## Befehl und Ergebnis

```text
python3 -m unittest tests.test_neral_v61_new_dev_eval \
  tests.test_neral_v6_fresh_dev_eval tests.test_neral_v6_regression_gate \
  tests.test_neral_v6_dev72_regression_gate \
  tests.test_neral_v6_dev72_regression_audit -v
```

36 Tests, 1.218 Sekunden, Ergebnis `OK`, keine Fehlschläge. Lauf auf CPython
3.12.3, CPU, ohne GPU und ohne Netz. Darunter explizit
`test_withdrawn_alias_gold_fails_execution_and_current_gold_matches` und
`test_every_family_wrong_integer_fails_and_prose_stays_ungraded`.

## Drei Ursachen, epistemischer Stand

1. **Messung und ungleiches Prompt-Protokoll.** Hoch, dass 29 gegen 21 keine
   Fähigkeitsdifferenz ist. Qwen allein sieht drei Beispiele, davon zwei keine
   ganzen Zahlen. v6 hat 19 unbewertete komplette Antworten, v5 sieben, Qwen
   null. Streng gepaart gegen v5 nur 3 Rückschritte und 0 Verbesserungen,
   p = 0.25. Mittel, welcher Teil der 19 Texte inhaltlich richtig war: die
   Rohantworten sind nicht im Repository. Alternative: Qwen wäre auch ohne
   Beispiele besser. Falsifikation: gekreuzte Prompts, gleiche Aufgaben, gleicher
   Decoder, unveränderte Gewichte; alte 21/72 nicht umschreiben.
2. **Strenger v5-zu-v6-Abfall unter gleicher Prompt-Bedingung.** Hoch für den
   Abfall im gesperrten Parser (Mathematik 7 zu 4, sonst flach). Niedrig dafür,
   dass die drei Fälle falsche Zahlen statt unlesbarer richtiger Rechnungen sind.
   Alternative: Rauschen bei korrelierten Familien. Falsifikation: nur die drei
   Partitionzähler des vorhandenen CPU-Audits aus dem privaten Laufordner.
3. **Schmales QLoRA-Rezept als Mechanismus, nicht als belegte Ursache.** v6:
   120 Schritte, Lernrate `1e-5`, keine 70/30-Mischung. v6.1 senkt auf `2e-6`
   und mischt 560/240, hat aber keinen abgeschlossenen Drei-Wege-Vergleich.
   Mittel, dass so ein Rezept Vergessen erzeugen kann. Niedrig, dass es diese
   drei Punkte erzeugt hat. Alternative: Spezialisierung auf die alten
   Oberflächen statt allgemeines Vergessen. Falsifikation erst auf neuen,
   familienweise zurückgehaltenen Aufgaben.

Der Alias-Goldfehler ist eine vierte, bereits im Generator korrigierte
Messlücke der **neuen** Suite. Er erklärt 21/72 nicht. Diese Sperre schließt
die Lücke, dass Generator und Handformel denselben Irrtum teilen können.

## Literatur, nur soweit sie hier trägt

Keine Prozentwerte aus den Papieren werden als Ergebnis dieses Repos übernommen.

- arXiv:2512.22337, Approximate Regularized Replay: Replay kann Vergessen nach
  schmalem LoRA-SFT auf Qwen2.5-Instruct-7B mindern. Nicht Coder, nicht 4-bit.
  Ein festes Token-Budget verdrängt neue Beispiele. 70/30 ist damit nicht belegt.
- arXiv:2602.04998: zuerst Lernrate, Schrittweite und Dauer prüfen. Die
  berichteten Peaks sind Llama-2-7B, nicht dieses QLoRA.
- arXiv:2405.09673: längeres LoRA-Training kann Erhalt verschlechtern. Rank,
  Alpha, Lernrate und Zielmodule wirken zusammen. Keine gewählte Konfiguration.
- arXiv:2305.14314 und arXiv:2405.14394: Antwort-only-Masken sind nicht Dogma.
  Die hiesige Maske bricht bei Präfixfehler oder Trunkierung ab. Ob sie optimal
  ist, bleibt eine spätere gepaarte Ablation bei gleichem Token-Budget.
- arXiv:2312.03732, rsLoRA: `alpha/sqrt(r)` ändert die Amplitude, garantiert
  weder Generalisierung noch weniger Vergessen.
- arXiv:2305.01210, EvalPlus: funktionale Tests legen falsche Referenzen offen
  und können Ranglisten ändern. Genau das motiviert die Ausführungssperre.
  Sie ändert hier keine Modellrangliste, weil kein Modell neu bewertet wurde.

## Spätere Bewertung, nicht dieser Lauf

1. Goldsperre bleibt Voraussetzung. Verbrauchtes dev72 nicht als Auswahlziel.
2. Optional und CPU-only: Partition der drei v6-gegen-v5-Rückschritte, nur Zähler.
3. Erst danach ein manueller GPU-Vergleich, und nur mit Freigabe auf `main`.
   Dieselbe Aufgabenliste, dieselbe Chat-Vorlage, gekreuzte Few-Shots, greedy,
   160 neue Tokens, für Qwen, v5 und den Kandidaten.
4. Getrennte Berichte für Transfer und Erhalt. Unsicherheit auf Familienebene,
   nicht nur Aufgaben-McNemar.
5. Eine unberührte externe Endsuiten-Prüfung erst nach der Kandidatenwahl.
   Nicht auf dieser Endsuite auswählen oder nachoptimieren.
6. Gleichstand oder Rückstand gegen v5: nicht promovieren. Ein Gewinn auf 72
   korrelierten Aufgaben ist kein Release.

## Risiken

- Die Allowlist kann eine spätere, harmlose Fixture-Form ablehnen. Das ist
  beabsichtigt fail-closed und leicht an den Tests sichtbar.
- Die Ausführung ist CPython auf dem jeweiligen Rechner, kein formaler
  Beweis für jede Python-Implementierung. Die Aufgaben nutzen nur den
  erlaubten Kern.
- Prompt-Muster sind absichtlich eng. Eine Umformulierung ohne Zahlenänderung
  lässt die Suite-Erzeugung scheitern, statt still ein falsches Gold zu liefern.
- CI: Pull Request startet die vorhandenen CPU-Workflows
  `neral-v61-dev72-ci` und `neral-v6-dev72-cpu-ci` (ubuntu-latest und
  windows-latest) sowie `python-syntax`. Die dev72-GPU-Workflows bleiben
  `workflow_dispatch` auf `main` mit Autorisierung. `neral-agent-live-pc.yml`
  ist der einzige Pull-Request-Pfad auf `neral-gpu` ohne Extra-Freigabe; diese
  Änderung berührt seine Pfade nicht. Kein `[skip ci]`, damit die CPU-Verträge
  laufen. Keine Workflow-Abschaltung.

## Nicht behauptet

Kein Fähigkeitsgewinn gegenüber Qwen oder v5. Kein neues 21/72. Kein Beleg,
dass v6.1 vergisst oder nicht vergisst. Training 42411765250 bleibt ein
Trainingslauf. PR 360 bleibt Werkzeugnachweis. PR 363 bleibt Receipt- und
Codec-Fix.
