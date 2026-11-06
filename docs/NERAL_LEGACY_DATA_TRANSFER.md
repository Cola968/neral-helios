# NeralTalk350 → aktuelles Qwen-7B-NERAL: Kandidat, kein Training

**Status: Audit / Vorbereitung – nicht automatisch als Trainingsdaten freigegeben.**

Der bisherige NERAL-QLoRA-Trainingsvertrag akzeptiert JSONL-Zeilen mit genau
`system`, `user` und `assistant`. Das ältere NeralTalk350-v3 enthält überwiegend
Einzelturns, außerdem mehrturnige Beispiele. Dieses Werkzeug überprüft die
vorhandenen drei Splits lokal, wählt **ausschließlich** `train`-Einzelturns aus,
und schreibt einen neuen, höchstens 1 MiB großen QLoRA-kompatiblen Kandidaten.

Die Quoten für die Entwicklungsauswahl: Code max. 120, Mathematik 100,
Reasoning 60, Chat 50, Erklärungen 40, Wissenschaft 20, Format 20,
Zusammenfassung 20. Maximal 430 Zeilen; tatsächliche Anzahl hängt von den
vorhandenen, gefilterten Daten ab. Auswahl ist deterministisch.

Sicherheits- und Reproduktionsregeln:

- Quell-Dateien `train.jsonl`, `val.jsonl`, `test.jsonl` müssen existieren und
  einen konsistenten Chat-Schema-Vertrag erfüllen; Quelle und Ausgabe dürfen
  nicht überschrieben werden.
- Genau normalisierte Root-Prompts aus `val` und `test` werden als Ausschluss-
  Fingerprints verwendet. **Val-/Test-Antworten werden weder importiert noch trainiert.**
- Auch die sechs in `scripts/neral_r32_compare.py` festgelegten Aufgaben werden
  aus dem Export ausgeschlossen. Semantische Beinahe-Duplikate und andere
  Benchmark-Suites benötigen zusätzlich einen separaten Audit.
- Nur bekannte Quellen `neral-generated-*` mit Projekt-Lizenz oder
  `oasst2-human-*` mit Apache-2.0 werden zugelassen. Große/kurze Antworten,
  Mehrturn-Beispiele, erkennbare Geheimnisse und `<think>`-Tags werden nicht
  in den ersten Kandidaten übernommen. Dies ist **keine** umfassende PII-
  oder Faktenprüfung.
- `candidate.jsonl` wird ausschließlich auf dem Windows-PC gespeichert,
  in einem neuen Verzeichnis `~/NERAL-QLoRA-Research/legacy-candidate-<runid>-<attempt>`.
- GitHub erhält nur `summary.json`: Zählwerte, Hashes und `approved_for_training=false`.
  **Keine Rohdaten, Antworten, Adaptergewichte oder Zugangsgeheimnisse.**
- Kein GPU-Training, keine Downloads, kein Helios-Deployment, keine Änderung
  am ursprünglichen Arctus350-Modell oder dem vorhandenen Rank-32-Adapter.

Ausführung: Manuell über GitHub Actions auf `main`, nach Code-Review des
Workflows `NERAL NeralTalk350 to Qwen candidate (read-only audit)` mit
Bestätigung `AUDIT_LOCAL_NERALTALK_ONLY`.

Lokale CPU-Vertragstests: `python -m unittest discover -s tests -p 'test_neral_legacy_transfer.py' -v`.

**Nicht vor dem Training übersehen:** Der existierende `verify_review`-Gate des
NERAL-QLoRA-Trainers verlangt einen **zeilenweisen Quellen-/Wahrheitsreview**
mit überprüfbarem Hash-Manifest. `summary.json` ist **kein** solches
Freigabemanifest. Vor jedem Training zusätzlich Qwen-Tokenizerlängen prüfen
(max. 512/1024/2048), faktische Korrektheit/Antwortqualität prüfen und
Benchmarks/ältere QLoRA-Pilotbeispiele auf Nahe-Duplikate kontrollieren.

Die 2,073 Milliarden Arctus-**verarbeiteten** Tokens repräsentieren keine
2,073 Milliarden zwingend einzigartigen oder Qwen-kompatiblen Trainingszeilen.
Die alten nativen Gewichte bleiben eine getrennte Modelllinie.