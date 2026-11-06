# NERAL v3: Rank-16 gegen Rank-32 – kontrollierter Versuch

**Status am 6. November 2026: Code vorbereitet und CPU-getestet; kein neuer
Rank-16-/Rank-32-Adapter trainiert und keinerlei produktive Aktivierung.**

## Was gleich bleibt

Beide Adapter trainieren mit demselben, nach dem erfolgreichen
Zweizeilen-Quell-/Qwen-Tokenizer-Audit erzeugten **320er-Datensatz**,
genau **80 Optimizerschritten**, LoRA-Alpha gleich Rank,
`lr=0.00005`, `seed=42`, per-device Batchgröße 1, Accumulation 4,
`max_length=512`, BF16/NF4 und **Completion-only-Loss**.
Die getrennten 64 Validierungsbeispiele werden nicht zum Fit verwendet.

Die neue Quelle `scripts/neral_v3_rank_ab_patcher.py` akzeptiert
ausschließlich den originalen QLoRA-Git-Blob
`f4b0f72440ce40653ca0ce2b171bec6721505e18` und erzeugt
*eine neue Scratch-Kopie* des Trainers. Die bisherige harte
Rank-32-3-Schritte-Schranke wird **nur für dieses neue separat
autorisierte, genau 80 Schritte umfassende Experiment** durch eine
harte 80-Schritte- und 512-Token-Prüfung ersetzt; die Originaldatei
bleibt unangetastet.

Die Anzahl trainierbarer Parameter muss vor jedem Optimizerschritt
exakt **40.370.176 (Rank 16)** bzw. **80.740.352 (Rank 32)** sein.
Beide alten Adapter und die Qwen-Basis werden nicht überschrieben.

## VRAM und Ausführung

Workflow: `.github/workflows/neral-v3-rank16-rank32-ab.yml`, nur
manueller, ausdrücklich autorisierter Windows-Self-hosted-Runner.
Die beiden Modelle werden **nacheinander** trainiert und in
separaten lokalen Ordnern gespeichert. Mindestens 10.500 MiB freier
GPU-Speicher und ein unbeschäftigter Runner werden vor dem Versuch
sowie vor dem Rank-32-Teil erneut nachgewiesen. Eine bereits
erstellte One-shot-Claim-Datei blockiert versehentliche zweite Starts.

**Achtung:** Kürzere Eingabesequenzen senken Aktivierungsspeicher,
nicht den gesamten Optimizer-/LoRA-Speicher. Der historische Rank-32-
Smoke-Test lag nahe an der Kapazitätsgrenze einer RTX 4070 mit 12 GB.
Es ist daher möglich, dass der Rank-32-Teil an der bestehenden VRAM-
Sicherheitsschranke scheitert. Ein Fehlschlag ist keine Freigabe, die
Schranke automatisch abzuschalten.

## Prüfplan

`scripts/neral_v3_rank_ab_eval.py` verwendet jeweils nur einen
NF4-Modellsnapshot mit einem Adapter im GPU-Speicher. Die Qwen-Basis
wird ausdrücklich innerhalb `PeftModel.disable_adapter()` geprüft,
danach Rank16 und schließlich Rank32.

Für alle drei werden identisch ausgeführt:

- 64 lokale Validierungsbeispiele (bei Core als **strenger Text-Exact-Match**,
  nicht als semantische Richtigkeitsprüfung);
- 80 bereits bekannte Mathematik-Entwicklungsaufgaben;
- 36 bereits verwendete Mathematik-/Anweisungsaufgaben;
- 20 zusätzlich handverfasste, nicht im 320er-Training enthaltene
  Aufgaben für Zahlen, Gleichungen und genaue Textanweisungen.

Die **36** sind keine unabhängigen Holdout-Aufgaben. Auch die 20
frisch geschriebenen Fälle sind *projektintern* und kein
externes standardisiertes oder verborgenes Benchmark. Coding,
Computer-Use und echte Agentenaufgaben sind damit nicht ausreichend
geprüft. Eine große allgemeine Verbesserung oder Release-Reife darf
nicht behauptet werden.

**Datenschutz:** Alle Trainingsdateien, Modellgewichte und Rohantworten
bleiben auf dem Windows-PC. GitHub erhält nach vollständigem Erfolg
ausschließlich numerische Trainings- und Evaluationsergebnisse sowie
SHA-Prüfsummen. Kein automatisches Adapter-Promoting.

## Externe Startblockade

Beim Versuch, den vorangehenden Zweizeilen-Konvertierungsworkflow
über die verfügbare GitHub-Anbindung direkt auszulösen, wurde der
Aufruf von einer Sicherheitsprüfung **vor GitHub** blockiert.
Es wurde daher kein neuer 320/64-Zweizeilen-Kandidat auf dem PC
bestätigt. Für das Rank-A/B-Experiment muss zuerst dieser Audit
regulär erfolgreich laufen; die neue Trainingspipeline selbst darf
nicht als bereits ausgeführt dargestellt werden.

Für den manuellen Start ist nach erfolgreichem Daten-Audit
`authorize=TRAIN_NERAL_V3_R16_R32_80_STEPS` und die numerische
`source_run_id` des abgeschlossenen Zweizeilen-Audits erforderlich.
