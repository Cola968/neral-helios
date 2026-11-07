# NERAL – Deep Research: Keine ungeprüfte Verschlechterung, schnellerer echter Fortschritt
_Stand: 07.11.2026 · Forschung + GitHub-/GPU-Befunde. Keine erfundenen Leistungsversprechen._

## A. Befund und Präzisierung der Anforderung

NERAL basiert auf der lokal SHA-fixierten Qwen2.5-Coder-7B-Instruct
Foundation (RTX 4070 12 GiB), mit separaten LoRA-Adaptern. Heute gemessen:

- Numerische EXT400-Teilmenge A/B n=250: v5 **69/250**, Qwen+Fewshot
  **66/250** (+1,2 Prozentpunkte; paired McNemar p≈0,664).
- Erster 72er-Entwicklungstest: v6 **21/72**, v5 **24/72**,
  Qwen+Fewshot **29/72**. Bei v6 19/72 statt v5 7/72
  vollständige, aber unparsebare Antworten.
- Neuer, disjunkter 72er-Entwicklungstest: v6.1 **16/72**, v5
  **16/72**, Qwen+Fewshot **19/72**. v6.1 vs v5: null
  Verbesserungen / null Regressionen. Kein Leistungsgewinn.
- Beide QLoRA-Rank16-Experimente liefen tatsächlich je 120
  Optimizer-Schritte auf der RTX4070. Neue Gewichtshashes bestätigen
  *Training*, nicht Transfer. In den jeweils ≈0,6 Epochen wurden
  120×4=480 Trainingsinstanzen pro Experiment verarbeitet.

**Training ohne jegliche Leistungsverluste ist nicht garantierbar.**
Verhindern lässt sich *Deployment* ohne positive, reproduzierbare,
unabhängig geprüfte Messung. Die alten SHA-gepinnten Gewichte bleiben
unverändert, Experimente werden separat abgelegt und per Laufzeitgate
blockiert. Evaluation ist kein Schutz, wenn ihre Musterlösungen falsch
sind: Beim neuen dev72 mussten sechs Python-Listenaliasing-Referenzen
vor Nutzung korrigiert werden (PR #364).

**Aktueller Default ist in `agent_auto_api.resolve_local_adapter_paths`
`legacy`** (alter r32), NICHT zwingend v5. Der v5-Kanal ist ein
explizites `NERAL_AGENT_ADAPTER_CHANNEL=v5`. Niemals behaupten, v5
sei flächendeckend bereits ausgerollt; den aktiven SHA im Helios-Status
anzeigen und die Stabilitätsentscheidung separat treffen.

## B. Vorrang: reale Qualität absichern, nicht einen Loss schönrechnen

1. Foundation und bewährten Adapter mit SHA-256 fixieren; kein
   `merge_and_unload()` oder Überschreiben des vorigen Adapters.
2. `neral_hf_identity.validate_adapter`, HF-Agent und Mini verweigern
   nach negativem/verbleibend unvalidiertem Entwicklergebnis die
   bekannten v6 **8bdc878a...** und v6.1 **33e939f6...**
   Gewichte in normalen Laufzeitpfaden. Forschung darf isoliert
   inferieren und nachmessen; weder eine umbenannte Datei noch ein
   neues gültiges Manifest darf diesen Block umgehen.
3. Jede neue Modellversion bekommt einen separaten Experiment-SHA,
   Daten-/Tokenizer-/Base-Revision, konkrete Codecommit-ID,
   Decoder-Konfiguration und einen reproduzierbaren Testbericht.
   **Trainings-Loss ist niemals das Promotion-Kriterium.**
4. Entwicklungssuiten dürfen wiederholt zur Diagnose benutzt werden,
   aber niemals als frische unabhängige Release-Beweise oder
   nachträglich als Trainingsdaten. Verifikation der Aufgaben-Oracles
   muss unabhängig vom Aufgabengenerator geschehen.
5. Bewertung von MODELLEISTUNG (gleiche Prompts, ohne Tools) strikt von
   SYSTEMLEISTUNG (mit THINK, Python-Tool, Code-Sandbox) trennen;
   beide messen und entsprechend benennen.

## C. Welche Forschung hilft konkret?

**LoRA Vergessen / Adapter-Bewahrung:**
- Shuttleworth et al., NeurIPS 2025: LoRA und vollständiges
  Finetuning ändern Gewichte grundlegend verschieden, Vergessen kann
  in einzelnen LoRA-Richtungen konzentriert sein.
  https://www.microsoft.com/en-us/research/publication/lora-vs-full-fine-tuning-an-illusion-of-equivalence/
- Yang et al., ACL 2026, SLoRA: störende Parameterkomponenten
  durch Subraumfilterung beim kontinuierlichen Lernen vermindern;
  in ihren Experimenten bis zu 29% weniger Vergessen (kein
  NERAL-Garantiewert). https://aclanthology.org/2026.acl-long.247/
- Lu et al., ACL 2025, CLoRA: Subraumregularisierung
  mindert Vergessen, erhöht aber Integrationskomplexität.
  https://aclanthology.org/2025.acl-long.940/

**Warum unsere synthetischen Traces riskant sind:**
- Kim et al., COLM 2026: Selbst-Destillation kann Unsicherheit und
  OOD-Reasoning unterdrücken, sogar bei sinkender Antwortlänge;
  auf einigen Modellen beträchtliche Leistungseinbrüche.
  https://www.microsoft.com/en-us/research/publication/why-does-self-distillation-sometimes-degrade-the-reasoning-capability-of-llms/
- Jin et al., ICML 2026: Datenvielfalt und Schwierigkeit
  sind wesentliche Auswahlgrößen; hohe reine SFT-Scores
  auf homogenen Beispielen sind kein Transfer-Nachweis.
  https://proceedings.mlr.press/v306/jin26o.html
- Kang et al., ICLR 2026, Quagmires: sogar verbesserte SFT-Ergebnisse
  garantieren keinen RL-Nutzen.
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/5d6ae8ba43ecb378030753c4408ef9bd-Abstract-Conference.html

**Der schnellere Weg bei 12 GiB: Test-Time-Compute + Verifikator:**
- Kang et al., ICLR 2026, T1: kleine Modelle profitieren stark
  davon, Rechnungen an Codeinterpreter auszulagern, statt ihre
  eigene Selbstkontrolle als Wahrheit zu betrachten.
  https://proceedings.iclr.cc/paper_files/paper/2026/hash/776a5f2c7d6dd4b0d83145fc044e2726-Abstract-Conference.html
- Ren et al., TMLR 2025, SETS: begrenzte Kandidaten-Generierung
  kombiniert mit Verifikation und gezielten Korrekturen.
  https://research.google/pubs/sets-leveraging-self-verification-and-self-correction-for-improved-test-time-scaling/
- EvalPlus (HumanEval+/MBPP+): Mehr funktionale Hidden Tests
  statt Syntax-/Codeverständnis-Stellvertreter.
  https://github.com/evalplus/evalplus

**Echte Präferenz-Optimierung als spätere, nicht sofortige Option:**
- TRL/Hugging Face DPO unterstützt PEFT, getrennte Ref-Logprobs
  und `precompute_ref_log_probs` zur Speicherreduktion.
  `ref_model=None` kann bei PEFT jedoch die *Basis ohne Adapter*
  bedeuten; für NERAL muss explizit der korrekte **v5-Referenzzustand**
  geprüft werden. Nicht blind die bestehende DPO-CPU-Pipeline als
  funktionsfähigen CUDA-Smoke bezeichnen.
  https://huggingface.co/docs/trl/dpo_trainer

## D. Konkrete Versuchsreihenfolge (nicht alles gleichzeitig)

**P0 – jetzt / vor neuem Training:**
- v5 oder andere bislang freigegebene Laufzeit nicht still austauschen;
  aktive SHA auf jeder Antwort sichtbar und Rückrollweg testen.
- Orakel prüfen, vollständige Rohantwort auf dem privaten Runner
  speichern, genauer Text-Diagnosecheck statt Format-Fake-Heuristik.
- Zwei Baselines fixieren: v5 und Qwen+Fewshot.
- Den bestehenden THINK-Parser um *nur voll verifizierbare* Mathematik
  und Code-TRACE-Fälle ergänzen; keine Ausführung fremden Codes
  außerhalb einer echten Sandbox. Ablationsvergleich:
  Modell pur / Modell + verifizierte Tools.

**P1 – nächste echte Gewichtsverbesserung:**
- Nur lokal generierte Antworten auf neuen TRAIN-only Aufgaben
  mit exakten Mathematikorakeln oder isoliert ausgeführten
  Python-Hiddentests als *wirklich korrekt/falsch* klassifizieren.
  Nicht parsebar ≠ mathematisch falsch; keine imaginären DPO-Negative.
- Erst nach unabhängiger Prüfung ausreichend viele diverse
  positive/negative Beispiele je Familie verwenden, kein reines
  Kopieren der Rechenfamilien; Fehlerklassen und Schwierigkeitsgrade
  ausbalancieren.
- Kleine, separat gespeicherte Adapterexperimente aus gleichem
  unverändertem Startpunkt: (A) wenig SFT mit altem Replay und
  echten Korrekturen, (B) 30-Step DPO mit korrekt gepinnter v5
  Referenz, sofern CUDA-Speicher-Preflight besteht; (C) optional
  SLoRA-Postprocessing nur nach eigenem CPU+GPU Nachweis.
  Variiere *einen Faktor pro Experiment*.
- Unabhängige echte Coding-Tests sind Pflicht, bevor Coding-Gewinn
  behauptet werden darf.

**P2 – Freigabegate:**
- Benutzte dev72 nur zur Diagnose. Neue, vor der Ausführung
  eingefrorene größere Testmenge, externe Provenienz und von
  Aufgabengenerierung getrennte Orakel.
- Paarweise gleiche Decoding-Settings Qwen/v5/Kandidat,
  exact McNemar + Familiencluster-Unsicherheit, Sprache und
  Domains getrennt, Format/Abbrüche separat.
- Kein relevanter Kategorieverlust; spürbarer Gesamtgewinn über
  v5 **und** Qwen. Für Release eigene Regeln in `GATE.md`
  und **immer explizite menschliche Freigabe**.
- Nach Kandidatenfreigabe kanarisch in Helios testen; automatisches
  Rollback auf den SHA-gepinnten vorigen Adapter bei Fehlern.
  Keine Selbstpromotion eines Trainingsworkflows.

## E. Ressourcengrenzen und realistische Erwartung

Die RTX4070 (12 GiB) kann echte 7B-QLoRA-Fortsetzungen auf
120 Stufen nachweislich ausführen, aber parallele 7B
Policy+Referenz beim DPO kann VRAM sprengen. Referenzlogprobs
*vorberechnen* oder zwei LoRA-Namen auf derselben 4-Bit-Foundation
prüfen, bevor irgendein DPO trainiert. Keine Cloud-Ausgaben.

Ein positiver A/B-Gewinn bei echten, neuen Aufgaben ist plausibel,
aber nicht im Voraus kalkulierbar. In wenigen Tagen sind belastbare
Messung, robuste Modellselektion und Verbesserung des
Tool-verifizierten Systems realistischer als ein stark gesteigerter
allgemeiner Wissens-/Reasoning-Score des nackten 7B-Modells.

**Kurz:** Keine weiteren blind aufsynthetisierten 120 Schritte.
Erst Orakel, Rückrollschutz, Tool-Verifikation und echte
On-Policy-Fehler. Dann ein isolierter, sehr kleiner und
kontrollierter Trainingsversuch mit unabhängiger Freigabe.
