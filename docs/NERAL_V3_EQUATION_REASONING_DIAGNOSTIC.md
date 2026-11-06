# NERAL v3: Gleichungsdiagnose vor jedem weiteren Training

Status: **reine GPU-Inferenz, keine Gewichtsänderung und keine automatische Freigabe**.

Ausgangspunkt: Die bereits bekannten acht Gleichungsaufgaben `novel-041` bis
`novel-048` im 80er Mathe-v2-Entwicklungstest. Sechs waren unter v2 falsch,
**alle acht behielten korrekt das verlangte reine Zahlenformat**.
Die sechs erwarteten Werte sind aus der versionierten Entwicklungsquelle
unabhängig nachrechenbar. Die tatsächlich ausgegebenen falschen Zahlen liegen
nur im lokalen 80er-Laufverzeichnis, nicht im GitHub-Artefakt.

Dieser neue Workflow `.github/workflows/neral-v3-equation-reasoning.yml`
prüft **dieselben Gewichte** für die Qwen-Basis, den ursprünglichen Rank-32
und den Rank-16-Mathe-v2-Adapter mit drei Aufgabenformulierungen:

1. `direct`: Wortlaut aus dem ursprünglichen Benchmark, nur Zahl.
2. `silent_steps`: ausdrückliche Aufforderung, Umformung und Division vor dem
   finalen *reinen Zahlenwert* gedanklich durchzuführen.
3. `two_step`: eine Zeile `ax = ...`, dann eine Zeile `x = ...`.
   Die erste Zeile wird unabhängig anhand der Gleichung auf Korrektheit geprüft.

Es werden pro Modell **8 × 3 = 24 Antworten**, insgesamt **72**, verglichen.
Die Auswertung trennt mathematische Korrektheit, Formatdisziplin und korrekte
Umformungszeile. Die Auswertung berücksichtigt auch mathematisch richtige
Ergebnisse in *formatwidrigen* Antworten, meldet beides aber getrennt.

Die Testfälle sind **bekannte Entwicklungsaufgaben**, kein neuer unabhängiger
Holdout. Weder eine Verbesserung in diesem Probe noch ein besserer Trainings-Loss
reichen allein, um den geplanten Core+Mathe-v3-Adapter freizugeben.

Sicherheitsbedingungen: nur manuelle `workflow_dispatch`-Freigabe,
`RUN_READONLY_V3_EQUATION_REASONING`, gepinnter v2-Lauf `42264245416`,
SHA-gesicherte Qwen-Basis und beide Adapter, lokal vorhandener BF16/CUDA-Stack,
mindestens 10.000 MiB freie GPU, gemeinsame GPU-Concurrency-Gruppe.
Der Workflow schreibt rohe Modellantworten ausschließlich in einen neuen
lokalen PC-Ordner, lädt nach GitHub nur Hashes/Zahlen hoch und startet keinen
einzigen Optimizerschritt.

**Entscheidungsregel für den nächsten Trainingstest:** Wenn `silent_steps`
bei stabiler Formatgültigkeit Gleichungen häufiger korrekt beantwortet, wird
zuerst Prompt-/Decoding-Design geprüft. Wenn nur `two_step` besser ist, wird
der Anteil an expliziten schrittweisen Gleichungen in der bereits erstellten,
384er-v3-Mischung gezielt neu bewertet. Wenn beide nicht helfen, müssen
Gleichungsdaten und Fehlerarten neu untersucht werden, statt blind Rank 32
oder die Schrittzahl zu erhöhen.

CPU-Tests:
```bash
python -m unittest discover -s tests -p 'test_neral_v3_equation_probe.py' -v
```
