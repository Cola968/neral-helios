# NERAL v6.1 – neuer struktureller Entwicklervergleich (72 Fälle)

Die neue v6.1-Replay-QLoRA startete am 07.11.2026 auf der RTX 4070 als
[GPU-Job #42411765250](https://github.com/Cola968/neral-helios/actions/runs/42411765250).
**Kein Erfolg/Intelligenzgewinn darf aus Trainings-Loss abgeleitet werden.**

Dieser Vergleich ist NUR ein Entwicklerdiagnostikum:
72 originell programmatisch erzeugte Fälle, 3 Domänen à 24 und insgesamt
12 Familien à 6, jeweils deutsche/englische Prompts. Keine Beispiele
oder Antworten aus dem v6.1-Training, v5/v6-Validation oder dem
alten bereits verbrauchten dev72. Der Generator prüft
normalisierte Promptgleichheit und 10-Gramm-Überlappungen.
Das beweist nicht, dass Qwen die Familien nicht bereits aus
dem Pretraining kennt.

Aufgaben:

- Mathematik: ganzzahlige GCD-Bestimmung, modulare Potenzen,
  Münzkombinationen mit ganzzahligen Nebenbedingungen,
  Ziffernsumme eines Quadrats.
- Logik/Reasoning: XOR-Parität, gerichtete Erreichbarkeit,
  Anzahl boolescher Belegungen, kürzeste Route in einem kleinen DAG.
- Codeverständnis (nicht Codegenerierung): Fibonacci-Rekursion,
  Mengenkomprehension, Listenaliasing und `enumerate`-Aggregation.

**Drei Vergleichsmodelle, nacheinander auf der lokalen GPU:**

1. Qwen2.5-Coder-7B-Instruct + exakt drei eingefrorene Few-Shots.
2. Derselbe Qwen-Backbone + ursprünglicher v5-Rank16-Adapter
   (SHA `2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5`).
3. Derselbe Backbone + neuer v6.1-Replay-Rank16-Adapter **nur nach**
   unabhängiger SHA-Prüfung des lokalen 120-Step-Trainingsberichts.

Erst wenn #42411765250 grün ist und der gewichtsgeprüfte Adapter vorliegt,
darf der manuelle Vergleichsworkflow
`.github/workflows/neral-v61-dev72-gpu.yml` auf `main` mit
`authorize=EVALUATE_NERAL_V61_NEW_DEV72_READONLY` ausgeführt werden.

Der abgesicherte vorhandene Inferenzpfad wird wiederverwendet:
4-Bit NF4, gleiche Vorlage, greedy, 160 neue Tokens, keine Werkzeuge,
eine eigenständige Modellinitialisierung pro Variante. Originalantworten
bleiben ausschließlich im privaten Windows-Verzeichnis
`NERAL-QLoRA-Research\\v61-new-dev72-<RUN_ID>-1`.
Auf GitHub erscheint nur eine aggregierte JSON mit Scores je Modell und
Bereich, korrekten Antworten vs nicht auswertbaren Formatantworten,
paarweisen Verbesserungen und Regressionen sowie McNemar-Explorationswert.

**Gleich gute Werte oder Werte unter v5 = Kandidat nicht promovieren.**
Auch ein positiver 72er-Entwicklungstest beweist keine unabhängige
Generalisierung; für den Release sind unberührte, fachlich geprüfte
600er-Dev/Release-Suiten und unabhängige Python-Codeausführung nötig.
