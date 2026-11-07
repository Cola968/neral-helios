# NERAL Deep Research II: Warum v6/v6.1 nichts messbar Neues gelernt haben

**Präziser technischer Befund (07.11.2026):** Beide v6-Rank16-QLoRA-
Trainingsläufe waren reale GPU-Optimierungen, aber weder v6 (21/72 statt
v5 24/72) noch v6.1 (16/72 exakt wie v5) zeigten einen unabhängigen
Transfergewinn.

## Sechs nachprüfbare Befunde

1. **Tatsächliche Datenabdeckung**. `scripts/neral_v6_r16_gpu.py`
   verwendet bei `per_device_train_batch_size=1`,
   `gradient_accumulation_steps=4` und `max_steps=120` nur
   **480 Trainingsbeispiel-Besuche** (ohne Gradient Accumulation zu
   verwechseln). Bei v6.1 sind das nominal 0,6 Epochen des
   800-Beispiel-Datensatzes; bei v6 0,659 Epochen der 728 Beispiele.
   Ein SHA beweist neue Gewichte, nicht die Verarbeitung *aller*
   Beispiele und schon gar keinen Transfer.
2. **Baseline-Promptkonfounder**. Im vorigen 72er-Benchmark sehen
   Qwen+Fewshot drei Beispiele im Prompt; NERAL v5/v6.1 dagegen
   keine. Der praktische Systemvergleich ist legitim, aber er ist
   **kein** sauberer Zero-Shot-Gewichtsvergleich. Für die nächste
   Untersuchung Qwen-0shot, v5-0shot, v6.1-0shot und separat
   alle drei mit denselben eingefrorenen Beispielen. Identische
   Generation-Settings und unabhängig geprüfte Orakel.
3. **Gleiche korrekte Fallmenge**. Auf dem jüngsten neuen 72er-
   Entwicklungstest erzielte v6.1 exakt dieselben binären
   Richtig/Falsch-Zuordnungen wie v5 (0 gewonnene, 0 verlorene
   Fälle). Das heißt *nicht*, dass die Rohtexte identisch sind.
4. **Loss ist kein Generalisierungsbeweis**. Die 140er v6.1-Validierung
   ist generatorverwandt und ihr Score wurde nicht vor/nach
   Training mit exakt gleichen Daten/Prompts gegen die v5-
   Referenz verglichen. Ohne diesen A/B-Log-Loss und gehaltene
   Aufgabenfamilien ist 1,0073 isoliert wenig aussagekräftig.
5. **Template- und Formatbias**. v5-Traces enthalten je nach
   Stil `CHECK`, `FINAL_ANSWER`, manchmal nur Kurzantwort;
   v6 ebenfalls. Fehlende Finals dürfen nicht als erwiesene
   mathematische Fehler ausgegeben werden. Unklare Endantworten
   bleiben eine reale Format-/Systemschwäche.
6. **Adapterwirkung statt Dateihash**. Der neue
   `neral_lora_drift_audit.py` berechnet beim alten v5 und
   neuen v6.1 **exakt den Frobenius-Abstand der wirksamen
   Low-Rank-Updates `BA`**, ohne den 7B-Backbone zu laden,
   ohne Training und ohne GPU. Große Dateiunterschiede
   können in `BA` aufgehen (etwa A×2, B÷2)!
   Only scalar sums and the ten strongest group deltas
   leave the private Windows machine. A large/small BA norm
   alone is still **kein Intelligenzbeweis**.

## Wissenschaftlich begründete Prioritäten

- **Weniger Vergessen:** Lin et al. (ICLR 2026) messen
  kleineren Learning-Rate-Erhalt und Token-Adaptive Loss
  Reweighting (TALR); keine absolute Garantie.
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/4e447acb68f57e29234bc0eb19896f11-Abstract-Conference.html
- **QLoRA statt FullFT weiter möglich:** LoRA Without Regret
  (Thinking Machines, 2025) erklärt, weshalb LoRA bei
  geeigneten Parametern ähnlich effizient sein kann und
  höhere Ranks nicht automatisch helfen.
  https://thinkingmachines.ai/blog/lora/
- **Qwen2.5-nahe Forschung:** SDC-LoRA (Findings of ACL 2026)
  erprobt die Eindämmung von Unterraumdrift explizit
  auf Qwen2.5-7B-Chat, MetaMathQA und CodeFeedback.
  Das ist eine verwandte, **nicht identische**
  Foundation (NERAL nutzt Qwen2.5-Coder-7B-Instruct).
  https://aclanthology.org/2026.findings-acl.1207/
- **Posthoc-Schritt als risikoarmes Forschungsprogramm:**
  SLoRA-Post (ACL 2026) filtert LoRA-Störkomponenten
  nach dem Training und kann theoretisch nach
  trainierten Gewichten angewendet werden. Die publizierten
  Verbesserungen gelten *nur für deren Aufgaben/Modelle*.
  Auf der 4070 hängt die Praktikabilität
  von tatsächlichem RAM/SVD/Quantisierungsschema ab.
  https://aclanthology.org/2026.acl-long.247/
- **Vorsicht bei selbst generierten Lösungen:**
  Microsoft/Kim et al. (COLM 2026) finden bis zu 40%
  Performanceeinbußen durch unpassende
  Reasoning-Selbstdestillation und OOD-Unsicherheitsverlust.
  https://www.microsoft.com/en-us/research/publication/why-does-self-distillation-sometimes-degrade-the-reasoning-capability-of-llms/
- **Daten-Auswahl wichtiger als rohe Menge:**
  Quagmires (ICLR 2026) bestätigt den Unterschied
  zwischen in-domain SFT-Loss und RL-Folgeleistung;
  manche 2-Epochen/klein-Datensatz-Experimente schlagen
  einen Durchgang über mehr Beispiele.
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d6ae8ba43ecb378030753c4408ef9bd-Abstract-Conference.html
- **Modelle fair vergleichen:** Prompt-Sensitivität und
  Heuristik-Grading können Scheineffekte verursachen.
  https://aclanthology.org/2025.emnlp-main.1006/
- **Coding = funktionale Tests:** EvalPlus ergänzt viele
  stärkere versteckte Tests; NERAL darf keine
  Coding-Superiorität aus Python-Tracing ableiten.
  https://evalplus.github.io/

## Entscheidungssystem – in genau dieser Reihenfolge

1. Lock den aktuellen explizit ausgewählten Laufzeit-Adapter,
   keine automatische v6/v6.1-Promotion. Bestehende runtime
   quarantine blockiert die bekannten SHA-Kandidaten.
2. Berichte den exakten neuen BA-Low-Rank-Abstand und
   480/800-Abdeckung. **Kein** Zusatztraining nur für
   einen kleineren Loss.
3. Erstelle einen vollständigen, identischen Nullshot-/Fewshot-
   Referenzvergleich Qwen-Foundation vs v5 vs v6.1 auf
   einem neuen Generator-/Promptfamilien-shift mit
   unabhängig replayverifizierten Orakeln; präregistriert.
4. Echte NERAL-v5 **TRAIN-only** Fehlantworten sammeln und
   extern verifizieren, begrenzte, diverse
   Code+Math+Logic-Korrekturpaare.
5. Kandidaten einzeln und aus gleichem v5-SHA trainieren:
   wenige gezielte SFT-Updates mit gemessenem Eval-Profil,
   optional DPO mit *korrekt gepinnter v5-Referenz*
   (TRL ermöglicht `precompute_ref_log_probs`,
   kein automatisch garantiertes 12-GB-Fit), danach
   optional spektrale LoRA-Störfilter als eigener
   separater Versuch. Keine fragwürdigen 120 Blindschritte.
6. Zum Schluss großes extern überprüftes
   600er-Release-Gate, keine relevanten Regressionen
   und manuelle Freigabe nach Canary-Test.

**Forschungsfazit:** Wir haben kein allgemeines Modellgesetz
gefunden, das Vergessen vollständig ausschließt.
Wir haben jedoch die Entwicklungsbehauptungen,
die strukturellen Schwächen und die konkreten
prüfbaren Gegenmaßnahmen identifiziert.
