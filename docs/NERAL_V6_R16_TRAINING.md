# NERAL v6: 120-Schritt-QLoRA-Experiment auf der RTX 4070

Ziel: **Tatsächlich geänderte Modellgewichte** mit neuen überprüften
Mehrschrittverfahren – statt nur Formatkontrakte. Dies ist ein **Experiment**,
kein Release, keine behauptete allgemeine KI-Überlegenheit.

## Daten

Die CPU reproduziert deterministisch zwei Quellen:

- **600** ursprüngliche SFT-Train-Aufgaben aus
  `scripts/neral_v6_compositional.py`: Prozent- und Gebührenfolgen,
  Flugtreibstoff, Zeitzonen, Lager und Mischungsverhältnisse, exakte
  Integer-/Fraction-Oracles.
- **128** original verifizierte kontrafaktische Rechen-Traces aus
  `scripts/neral_reasoning_train_candidate.py`; die
  `verify_case`-Funktion prüft jede Rechnung gegen unabhängige
  Decimal-Replays. Das Training erhält kurze, vollständige
  `CHECK`-/`FINAL_ANSWER`-Lösungsfolgen.
- **100** von v6 getrennte Validation-Aufgaben.
- **100 synthetische Holdout-Aufgaben** bleiben ausgeschlossen.
  Sie sind **kein** unabhängiger externer Test und dürfen für keine
  Releasefreigabe verwendet werden.

Die neuen Formate sind programmatisch und teilweise inhaltlich
verwandt. Kein menschlicher Qualitätsreview ist behauptet oder
automatisch freigegeben. Die bekannten v5- und 400er-Tests werden
**niemals** als neue Trainings-Labels benutzt.

## Modell und GPU

- Start: SHA-geprüfter, bereits vorhandener **NERAL v5 r16 480-Schritte**,
  SHA `2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5`
- Foundation: lokaler Qwen2.5-Coder-7B-Instruct auf unveränderter
  Revision `c03e6d358207e414f1eca0bb1891e29f1db0e242`
- 4-Bit NF4 / BF16, PEFT LoRA rank 16, **40.370.176**
  trainierbare LoRA-Parameter; Base-Gewichte eingefroren.
- **120 echte Optimizer-Schritte**; 1 Microbatch, vierfach akkumuliert,
  Lernrate 1e-5, paged AdamW 8-bit, Gradient Checkpointing und
  maximal 768 Tokens pro Aufgabe.
- Nur nach echten Qwen-Tokenizer-Checks: Chatgrenze unverändert,
  **nur Assistant-Completion im Loss**, keine stillen Trunkierungen.
- `nvidia-smi`: mindestens 10.000 MiB frei und höchstens
  20 % GPU-Auslastung; kein paralleler lokaler GPU-Job.
- Eigener neuer Ausgabeordner `v6-r16-120-<RUNID>-1` auf dem
  Windows-PC. Der v5-Adapter bleibt unverändert; eine neue
  `candidate-adapter/adapter_model.safetensors` wird separat gehasht.

## Start

Auf GitHub Actions in `main` ausschließlich
`.github/workflows/neral-v6-r16-120-train.yml` manuell ausführen,
mit `authorize=TRAIN_NERAL_V6_R16_120_EXPERIMENTAL`.

Ohne diese exakte Genehmigung findet **kein Training** statt. Der
Workflow hat eine Obergrenze von 100 Minuten. GPU-Prozesse und
Unterbrechungen können dennoch fehlschlagen; gerade die CUDA-/TRL-
Kompatibilität ist vor dem ersten echten Lauf nicht bereits belegt.

Die einzige öffentliche GitHub-Datei ist ein aggregierter
`summary.json`-Bericht mit Zeit, Loss, VRAM und Gewichtshash.
**Keine** Trainingsprompts, Holdout-Antworten, Rohgewichte, LoRA-Dateien
oder privaten GPU-Protokolle als Artefakt hochladen.

## Bewertung nach dem Experiment

Ein niedrigerer Trainings-/Validation-Loss und ein neuer
Adapter-SHA zeigen nur, dass der Optimizer echte Gewichte verändert
hat. Das ist **keine** bessere Generalisierung.

Vor einem produktiven v6 müssen folgende Punkte messbar erfüllt sein:

1. Ein neuer, vorher eingefrorener CPU-Grader und korrekt
   kalibriertes Antwortende.
2. Paare v5 vs v6 auf *frischen* Aufgaben und unabhängige
   Qwen+Few-Shot-Baseline mit gleicher Inferenzkonfiguration.
3. Versteckte Python-Tests in isolierter Sandbox und eigene
   Logikaufgaben, nicht nur Kompositionsgenerator-Familien.
4. Bestehende Stärken ohne bedeutende Regression; alle
   Promotionsregeln in `GATE.md`.

Der Code setzt `release_validated=false`,
`adapter_promoted=false`, `independent_transfer_proven=false`.
Ohne diesen Nachweis ist v6 **ein trainierter Kandidat**, nicht
nachweislich besser als Qwen.
