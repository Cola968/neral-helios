# NERAL v2: zusaetzliche, unabhaengig formulierte Entwicklungsaufgaben

Die 80 bestehenden Mathe-v2-Aufgaben verwenden verwandte Aufgabenformate zur
synthetischen Ausbildung. Ein guter Wert dort beweist keinen allgemeinen Fortschritt.

Dieser **read-only** GPU-Test ergaenzt 24 anders formulierte Rechenaufgaben
(u. a. mehrschrittiges Rechnen, Woerter, negative Zahlen, Brueche und
Sachkontexte) und 12 einfache Anweisungstreue-Aufgaben mit festem Text.

Testet dieselbe lokale Qwen-Basis, den alten NERAL Rank-32 und einen separat
trainierten Math-v2-Rank-16-Adapter. Keine Code-Ausfuehrung der Modellantworten.
Die Antworten werden nur lokal in `local_details.json` gespeichert, damit
Formatfehler analysiert werden koennen. GitHub erhaelt ausschliesslich
`summary.json` (Zaehler, Kategorien und Hashes, keine Texte oder Gewichte).

**Grenzen:** Dies ist eine oeffentliche Entwicklungssuite, keine verblindete
Release-Bewertung, kein Nachweis besserer Programmierfaehigkeiten. Die
Modellgewichte werden nie automatisch aktiviert, auch bei besseren Ergebnissen
nicht. Die GPU-Workflow-Ausfuehrung ist nur nach erfolgreichem Trainingslauf
sinnvoll und darf die laufende Entwicklung nicht stoeren (gleiche Concurrency-Gruppe).

CPU-Test:

```sh
python -m unittest discover -s tests -p test_neral_v2_generalization_probe.py -v
```