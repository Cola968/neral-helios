# NERAL – verbindliche Entwicklungs-Roadmap bis 31. Dezember 2026

Status: angenommener Arbeitsplan. Termine sind **Zieltermine**, kein Versprechen
einer positiven Abnahme oder eines Siegs gegen Frontier-Modelle.

## Referenz, Sperren und Hardware

- Die bestehende GPU-Referenz ist **main@29b29822c8005fd5605fafdd2a79747570d6e71c**.
  Der einmalige 3+3-GPU-Lauf auf genau diesem Commit bleibt bis zur
  **separaten ausdrücklichen Freigabe** gesperrt. Dieser PR ändert main nicht.
- Erst nach bestandenem 6/6 folgt ein ebenfalls freizugebender, kontrollierter
  **50-Aufgaben-A/B-Vergleich** gegen unveränderte Qwen3.5-Basismodelle.
- RTX 4070 mit **10,5 GiB = 10.752 MiB maximaler GPU-Belegung** für Modell,
  CUDA-Kontext und Training. Bei fehlender VRAM-Messung/Überschreitung abbrechen,
  niemals Schutzgrenzen erhöhen.
- **Z3, SymPy, Lean, Compiler, pytest und Hypothesis: CPU-only, GPU-Sichtbarkeit
  abgeschaltet, Ziel 0 MiB VRAM.** Netzwerksperre, Ressourcenlimits und
  Prozessisolation vor echter Codeausführung unabhängig nachweisen.
- **421 gesicherte Blind200-Ergebnisse inklusive SHA-256: unangetastet.**
  Nicht lesen, neu bewerten, kopieren, trainieren, überschreiben oder ausgeben.
- Normales Chat-Routing, Amber/Quartz, Quellenprüfungen und ~6-s-Alltagspfad
  dürfen durch Math/Codex nicht beeinträchtigt werden. Vor Integration eigene
  Routing- und p50/p95-Latenz-Regressionsgates.
- Formale Solver-Beweise bestätigen nur die eingespeisten Formalisierungen.
  Das korrekte Übersetzen der natürlichen Sprache bleibt ein eigener Prüfpunkt.

## Woche 1 – 23.–30. November: V4.2 / NERAL 1.0

3+3-Abnahme auf **exakt** main@29b29822 (6/6, HTTP 200, 10,5-GiB-Hülle).
Danach 50er-A/B-Messung als dokumentierte Null-Linie. Modell/Adapter/Tool-SHAs,
Tokenbudget, gleiche Aufgaben, genaue Testregeln, Erfolg, Latenz und GPU-Spitze
dokumentieren. Kein Start aus diesem PR, keine ungeprüfte 6/6-Behauptung.

## Woche 2 – 1.–7. Dezember: Dual-Verifier-Kernel

- **Obsidian Math**: Falsification-First mit Z3 und Grenzwert-Fuzzer;
  exakte SymPy-Algebra und Polynome. SAT = Gegenbeispiel; UNSAT = nur im
  geprüften Modell/unter Annahmen keine Widerlegung. Nicht untersuchte Bereiche
  ergeben unknown. Keine modelldiktierten Skripte außerhalb der CPU-Sandbox.
- **Obsidian Codex**: Tests vor der endgültigen Antwort; pytest und
  Hypothesis müssen echte Tests sammeln. Bei Ausfall realer Stacktrace,
  begrenzt zwei Patch-/Rerun-Schleifen; Fehler bleiben als ungeprüft markiert.
- **Routing**: alltägliche Gespräche, Erklärungen und gewöhnliche Recherche
  bleiben aus schweren Solver-/Compiler-Schleifen heraus.

## Woche 3 – 8.–14. Dezember: Lean 4 und Multi-File-Codex

- Lean 4, Mathlib und lokale CPU-only REPL vorbereiten; erst prüfen, ob
  eine kompatible REPL für `lake exe repl` tatsächlich installiert ist:
  dieser Aufruf gehört nicht automatisch zu Lean/Mathlib.
- Best-First-Top-K-Taktiken werden vom Modell auf GPU vorgeschlagen, jeder
  Beweisschritt im isolierten CPU-Prozess geprüft. Compilerdiagnosen für
  Backtracking. Keine `sorry`/`admit`-Beweise akzeptieren.
- AST-/Import-/Klassen-Slicing und transaktionale Diff-Patches mit
  Pfad-Allowlist, Rollback und unabhängigen Regressionstests.

## Woche 4 – 15.–21. Dezember: NERAL 1.1 RFT-QLoRA

Zielmix geprüfter Spuren: **35 % Math**, **35 % Codex**, **30 % Alltags-Replay**
(Dialog, Anweisungen, Stil/Humor, Belegsyntax, ehrliches Nichtwissen).
Maximal 3.072 Tokens pro Beispiel, NF4 4-bit, LoRA Rank 16–32, Gradient
Checkpointing und 8-bit paged AdamW. **Die Machbarkeit von 9B/Rank 32/3072
bei 10,5 GiB ist noch nicht bewiesen.** Erst GPU-preflight nach Freigabe;
bei Überlauf Rank, Microbatch oder Länge reduzieren oder abbrechen.
Der Replay-Mix garantiert nicht, dass kein Catastrophic Forgetting entsteht:
getrennte Alltags-Gates sind verpflichtend. Training und Holdout strikt trennen.

## Woche 5 – 22.–31. Dezember: Frontier-Showdown

Ein **neuer, unabhängiger und privat versiegelter 180-Aufgaben-Holdout**:
60 Alltag & Fakten, 60 schwere Mathematik, 60 Programmierung. In GitHub
liegen **nur Metadaten-Schema und synthetische Unit-Tests**, keine echten
Prompts, Antworten, Referenzen oder privaten Hashlisten.

A: unverändertes Qwen3.5, **nur Modell**; B: NERAL 1.1 Adapter **ohne Tools**;
C: NERAL 2.0 **mit Math-/Codex-Werkzeugen**; D: Frontier-Modelle in einem
fairen Vergleich mit **vergleichbarem Werkzeugzugang, Zeit- und Tokenbudget**
(oder explizit als asymmetrischer Vergleich gekennzeichnet).

Berichten: getrennte Erfolgsquoten, Pass@1, Unsicherheit, Fehlertypen,
Latenz, VRAM, Kontrollbedingungen und Datenkontamination. Frontier-Überlegenheit
nur bei tatsächlichem unabhängigem Blindtestbefund behaupten; einen
projektinternen Test **nicht als offiziellen Artificial-Analysis-Score** ausgeben.

## Was diese erste reine CPU-Vorbereitung tatsächlich liefert

- neral_frontier/codex.py: sicherer **Test-First-Controllervertrag** mit
  injiziertem Sandbox-Protokoll; noch **kein** echter ausführender Docker-Runner.
- neral_frontier/math_kernel.py: CPU-only Z3-Gegenbeispielsuche und
  exakte SymPy-Prüfungen für begrenzte univariate Polynome; noch
  **kein** Lean-REPL, Grenzwertfuzzer oder freier LLM-Parser.
- neral_frontier/blind180.py + schemas/neral-frontier-blind180.schema.json:
  strenger Validator und JSON Schema für 60/60/60-Metadaten.
  **Noch kein Blindtest-Datensatz, kein Grader, keine Auswertung.**
- tests/test_neral_frontier_cpu.py: synthetische CPU-Vertragstests.
- .github/workflows/neral-frontier-cpu-scaffold.yml: nur GitHub-gehostete
  CPU-Tests, keine selbstgehostete GPU.

Diese Module sind **nicht in den bestehenden Chat-/Modellpfad integriert**.
Eine Integration, Installation auf W39, ein GPU-Training oder ein Benchmark
bedarf jeweils einer späteren ausdrücklichen Freigabe.
