# NERAL 1.1 – eingefrorene Baseline und Qwen3.5-9B-QLoRA-Experiment

Stand: 12.11.2026. **Kein automatisches Training und kein automatischer Merge.**

## 1. Nicht veränderbare Ausgangs-Baseline

- Prüfkandidat: PR #533, SHA \`d280516e1a28a1cbdbc5b6b9b89d694769f6292f\`.
- A/B mit RTX 4070 + Docker: Python A \`-422\`, \`verified=true\` (3,469 s);
  Logik B \`12\`, \`repaired\` (18,328 s):
  https://github.com/Cola968/neral-helios/actions/runs/43189782646
- TIR12 (bekannte Entwicklungsfamilien): NERAL 9/12 vs. rohes Qwen3.5 7/12, Differenz
  +2 Aufgaben bzw. +16,7 Prozentpunkte, 4 Siege, 2 Niederlagen, 10 erfolgreiche Tools:
  https://github.com/Cola968/neral-helios/actions/runs/43192047994
- Vier synthetische Transferfälle (Entwicklung): **1/4**, keine Generalisierung nachgewiesen:
  https://github.com/Cola968/neral-helios/actions/runs/43192261645
- Kein Z3-Aufruf: NERAL benutzt zurzeit den eigenständigen, begrenzten
  Finite-Domain-Solver \`neral_formal_constraints.py\`. \`Z3\` ist nur
  als Zukunftsoption zu betrachten; Trainingslabels entsprechen dem
  tatsächlich unterstützten \`NERAL_CONSTRAINTS_JSON\`-Vertrag.

## 2. PR-Merge-Reihenfolge

1. PR #505 ist Draft gegen \`main\`; zuerst neue CI/Reviews,
   GitHub-Konfliktprüfung und nötigenfalls Rebase auf aktuelles \`main\`.
2. Nur nach Freigabe PR #505 separat in \`main\` mergen. Squash-Merge
   verändert die Ancestry; PR #533 danach mit \`git rebase --onto\`
   auf \`main\` aufsetzen statt die #505-Commits doppelt mitzunehmen.
3. PR #533 ist **Ready for Review**, bleibt jedoch bis dahin gestapelt
   auf #505. Nach #505-Merge: Ziel \`main\`, Review/CI inklusive realer
   GPU-Abnahme erneut kontrollieren, dann separat mergen.
4. Der Datensatz-/Trainingszweig ist ab \`d280516\` verzweigt und darf
   das Baseline-Commit nicht verändern. Neue LoRA-Adapter sind ein
   **Experiment**, keine stillschweigende Release-Änderung.

## 3. Dateistruktur

\`\`\`text
scripts/
  neral_qlora_dataset.py       # 500–1000 train rows; Default 720 (360 + 360)
  neral_qlora_holdout_eval.py  # 24 Fälle mit Solver-Orakel, Leak-Gate, Scorer
  neral_qlora_train.py         # lokales 4-Bit-QLoRA, assistant-only loss
  neral_qlora_export.py        # optionaler GGUF-/Modelfile-Export
datasets/
  neral_qwen35_holdout_v1.jsonl # 24 kuratierte, absichtlich ungesehene Aufgaben
tests/
  test_neral_qlora_curriculum_cpu.py
.github/workflows/
  neral-qlora-curriculum-cpu.yml
.data/neral_qlora/              # generiert, .gitignore; keine Repo-Trainingsdaten
  train.jsonl
  train_manifest.json
\`\`\`

Die JSONL-Zeilen enthalten \`messages\` mit den Rollen system/user/assistant
sowie Metadaten. Der Trainer erzeugt **erst beim Tokenisieren**
\`tokenizer.apply_chat_template(..., enable_thinking=False)\` des tatsächlich
geladenen Qwen3.5-Tokenizers. Keine von Hand gebauten ChatML-Endtokens!
Ein geändertes Template führt zum Abbruch; alle Nicht-Assistant-Tokens
haben Label \`-100\` und gehen nicht in den Loss ein.

Das SFT-Curriculum generiert pro Aufgabe ein präzises Turn-0-JSON und
ein Turn-1-Korrekturbeispiel mit ungültigem Ausgangsschema und
Solver-Diagnose (\`solver_type_error\` oder \`solver_unsat_fallback\`).
Alle Sollwerte werden mit dem **realen Solver** und dem
Source-Grounding-Parser geprüft. Dies beseitigt keine mögliche
Generalisierungslücke aufgrund wiederholter synthetischer Satzmuster.

Die 24 Holdout-Aufgaben sind **getrennt von der Generatorfamilie**
(12 boolesche \`at_least\`-Fälle, 12 mehrbedingte vollständige Ordnungen
mit anderen Namen). Prompt- und Formalschema-Gleichheit werden vor
Training ausgeschlossen; nur die Trainingszeilen werden an das Modell
übergeben. Da der Holdout hier im Repository liegt, ist er **nicht blind**
für die Entwickler. Er liefert keine unbeeinflusste finale Release-Abnahme.

## 4. Reproduzierbare CPU-Vorbereitung

Im Repo-Root mit Python 3.12:

\`\`\`bash
python -B -m unittest discover -s tests -p test_neral_qlora_curriculum_cpu.py -v

python -B -m scripts.neral_qlora_dataset \
  --out .data/neral_qlora/train.jsonl \
  --holdout datasets/neral_qwen35_holdout_v1.jsonl \
  --pairs 360

python -B -m scripts.neral_qlora_holdout_eval \
  --holdout datasets/neral_qwen35_holdout_v1.jsonl \
  --train .data/neral_qlora/train.jsonl

python -B -m scripts.neral_qlora_train \
  --train .data/neral_qlora/train.jsonl \
  --holdout datasets/neral_qwen35_holdout_v1.jsonl \
  --out .data/neral_qlora/check-only --check-data-only
\`\`\`

Der Generator überschreibt \`train.jsonl\` absichtlich nicht. Dies
schützt Versionen und SHA256-Manifest vor stiller Manipulation.

## 5. Manuelles GPU-Training und Export

Unsloth / PEFT / bitsandbytes, aktuelle Versionen für Qwen3.5, Torch
mit passendem CUDA-Treiber, echtes 9B-Checkpoint lokal verfügbar.
**Vorher Ollama entladen** und VRAM prüfen; 12 GiB sind knapp.
Default: NF4, Rank 8, 2.048 Token, Batch 1, Gradientenakkumulation 8,
Checkpointing, 60 Optimierungsschritte. Diese Konfiguration ist
**noch nicht** als erfolgreiches 9B-GPU-Training validiert.

\`\`\`bash
python -B -m scripts.neral_qlora_train \
  --train .data/neral_qlora/train.jsonl \
  --holdout datasets/neral_qwen35_holdout_v1.jsonl \
  --out .data/neral_qlora/run001 \
  --model unsloth/Qwen3.5-9B --max-steps 60
\`\`\`

Nur wenn Training und Adapter validiert sind:

\`\`\`bash
python -B -m scripts.neral_qlora_export \
  --adapter .data/neral_qlora/run001/adapters \
  --out .data/neral_qlora/run001/gguf --quant q4_k_m

# Nach erfolgreichem GGUF-Export und geprüfter Kompatibilität:
ollama create neral-qwen35-qlora:dev \
  -f .data/neral_qlora/run001/gguf/Modelfile.neral-qlora-dev
\`\`\`

Qwen3.5-9B ist multimodal; **GGUF-/Ollama-Support kann von Version,
mmproj und Unsloth-Konverter abhängen**. Das Exportskript verweigert
fehlende GGUF-Ergebnisse, überschreibt kein bestehendes Qwen-Modell
und führt \`ollama create\` nicht selbst aus. Nach erfolgreichem Import
zunächst das Modell identisch quantisiert **ohne** Adapter exportieren
und beide Modelle erneut unter gleichen Dekodier-/Tool-Bedingungen
vergleichen; danach Holdout, TIR12 und Regression messen.
Ein Adapter-Erfolg ist nicht durch niedrigeren Trainingsloss bewiesen.

## 6. Evaluationsvertrag

Ein reales Inferenzprogramm muss für jeden Holdout-Fall genau eine
JSONL-Zeile \`{"id":"holdout-bool-01","response":"{...}"}\` erzeugen.
Der Scorer prüft tatsächliche modellgenerierte Schemata, keine
synthetisch vorbefüllten Antworten:

\`\`\`bash
python -B -m scripts.neral_qlora_holdout_eval \
  --holdout datasets/neral_qwen35_holdout_v1.jsonl \
  --train .data/neral_qlora/train.jsonl \
  --predictions .data/neral_qlora/real_model_predictions.jsonl
\`\`\`

Die vier bekannten Transferaufgaben und TIR12 dürfen **niemals** als
Trainingsbeispiele aufgenommen werden. Ohne echten Modelllauf gibt der
Holdout-Scorer ausschließlich \`model_score=not_measured\` aus.
