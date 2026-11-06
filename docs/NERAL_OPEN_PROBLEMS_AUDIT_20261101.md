# NERAL – Vollständige Übersicht aller aktuell *erfassten* offenen GitHub-Arbeiten

**Erfasst am 01.11.2026** über die privaten GitHub-API-Seiten 1 und 2 des Repositories `Cola968/neral-helios`. Momentaufnahme: **26 offene Issues und 110 offene PRs**, davon **97 Drafts**, **20 direkt gegen `main`** und **90 auf andere Entwicklungsbranches gestapelt**.

> Diese Liste erfasst alle derzeit offenen **GitHub-Einträge** und ausgewählte nachgewiesene Release-Blocker. Sie behauptet weder, dass jeder Issue ein Fehler ist, noch dass keine ungemeldeten Bugs existieren. Jeder PR kann geplant, unfertig, redundant oder bereits getestet sein. Keine ungeprüften automatisch geschlossenen Issues.

## Nachgewiesene technische Fehler und Änderungen

1. **`code_check` HTTP 400 (geschlossen im Draft):** Real Qwen gab `code_check` mit Argument-Dict und inerten Metadaten zurück. Eingangsvalidierung zu strikt. In #215/Diagnose-#216 eng begrenzt behoben und **echter Windows-Qwen-Test 41329964479 3/3 grün**. Nicht auf `main` gemergt.
2. **Codeprüfer-Antwort durch künstliches 128-Token-Budget abgeschnitten (Testproblem):** Diagnose-#216 verwendet im synthetischen Codefall jetzt 320 Tokens. Realer Qwen-Test beendet Antwort nach 173 Tokens vollständig. Standard-App-Budget/Denkprofile getrennt betrachten.
3. **SOLVE kombiniert Gleichungsseiten ohne Bereichsprüfung (bestätigter Bug):** Die unabhängigen AST-Zwischenergebnisse sind begrenzt, aber `coefficient=a-c` und `constant=d-b` überschritten 10**24. Vier neue Tests **RED 41331184432**, einzelne Bereichsprüfung aus #217 in #215 übernommen. GREEN-CI 41331306141, Gesamtintegration 41331306135, offline Gate 41331306206, Syntax 41331306134 erfolgreich; `main` unverändert.
4. **One-shot GPU-Probe blieb erneut auslösbar (konservativ gehärtet):** Der einmalige Windows-Test war erfolgreich, aber sein Workflow ließ erneute auslösende Commits zu. In Diagnose-#216 ist der Job nun dauerhaft `if: "false"`; weitere GPU-Jobs benötigen eine gesonderte Freigabe und entsprechende Überarbeitung. Dieser Schutz ist ebenfalls noch Draft.

## Tatsächlich noch offene Release-Blocker

| Priorität | Problem / Erfordernis | Nachweis / was fehlt |
|---|---|---|
| P0 | **Integration nicht freigegeben** | #215 besteht CI als gemeinsamer Draft, aber kein manueller reviewter Merge nach `main`; 90 weitere PRs sind auf Nicht-main-Branches gestapelt. Viele sind Forschung und müssen nicht in die erste Beta. |
| P0 | **Reale Qualitätsbreite nicht nachgewiesen** | Live-Qwen-Test 41329964479 beweist genau 3 synthetische Fälle: Wissen, Rechnung, `code_check`, nicht GPT-3.5-Parität oder umfassende Zuverlässigkeit. Breiter unabhängiger DE/EN-/Coding-/Kontext-Test fehlt. |
| P0 | **Endgeräte/Installationsabnahme** | Windows-Launcher-CI und simulierte Browserflüsse grün; vollständige manuelle Nutzung in Helios auf Zielgerät, Speicher-/Neustart-/Fehlerfälle, Android/PWA, Accessibility und Betrieb mit dauerhaftem Ollama müssen abgenommen werden. |
| P0 | **Sicherheits-/Datenschutz-/Lizenzfreigabe** | Sechs Launchchecks wurden in anderem Draft #206 vorbereitet, aber noch nicht in den Core-Release integriert bzw. rechtlich geprüft; keine unbelegte öffentliche Freigabe. |
| P0 | **Kontrollierte PR-Konsolidierung** | 110 offene PRs (97 Drafts) mit vielen geschichteten und historischen experimentellen Varianten. Entscheidungsmatrix pro PR, Test/Review, bewusste Schließung redundanter PRs nach Zustimmung erforderlich. Keine Mass-Merges. |
| P1 | **Arctus350 native Antwortqualität** | Eigenes ~350M-Modell wurde nicht durch die Qwen-Tests verbessert. Unabhängige OOD-/Instruktions-/Generalisationstests und datensatzsaubere A/B-Trainingsevidenz nötig. Qwen ist externes Zusatzmodell. |
| P1 | **24/7 KI-Coding nicht durchgehend autonom** | ChatGPT-Automation arbeitet stündlich in Cloud; Cloudflare-Koordinator im anderen Stack ist nur Cron/Heartbeat, CAS-Aufgabenqueue liegt in Draft #214; kein grenzenlos durchlaufender kostenloser Programmieragent. |
| P1 | **Mehrwerkzeug-/Deep-Mode-E2E** | Offlinevertrag/Fixtures für maximal zwei freigegebene Tools grün; echter Qwen-Live-Test hat **nur** `code_check` verwendet; zwei Toolaufrufe nacheinander, Deep-Mode, lange Kontexte und OOM bleiben gesondert zu messen. |
| P1 | **Budget-/Quota-/Hosting-Nachweis** | Keine zusätzlichen GPU-/API-Aufträge beauftragt; GitHub Actions hat Kontingent. Cloudflare Free-/Billing-Status wurde nicht unabhängig verifiziert; keine garantierte unbegrenzte 24/7-Leistung. |

## Sämtliche 26 offenen Issues (nicht automatisch 26 Bugs)

### P0 – Produktauslieferung, Agenten und Sicherheit

- **[#195](https://github.com/Cola968/neral-helios/issues/195)** – NERAL: vorhandenen Qwen-Assistant aus PR #129 sicher in neuen Chat und Arctus350 (#173/#175) integrieren
- **[#137](https://github.com/Cola968/neral-helios/issues/137)** – NERAL Coding Agent · v0 technischer Vertrag und erster Testauftrag
- **[#136](https://github.com/Cola968/neral-helios/issues/136)** – NERAL W4 · Red Team, Leistungsnachweis, Release-Entscheid (18.11.–24.11.)
- **[#135](https://github.com/Cola968/neral-helios/issues/135)** – NERAL W3 · Weiterarbeiten und Persönlichkeit (11.11.–17.11.)
- **[#134](https://github.com/Cola968/neral-helios/issues/134)** – NERAL W2 · Coding Agent MVP (04.11.–10.11.)
- **[#133](https://github.com/Cola968/neral-helios/issues/133)** – NERAL W1 · Chat, Persona und erste Qualitätssicherung (28.10.–03.11.)
- **[#102](https://github.com/Cola968/neral-helios/issues/102)** – Neral Live Telemetry
- **[#42](https://github.com/Cola968/neral-helios/issues/42)** – Helios App v0.1 — adaptive Android + Desktop foundation

### P1 – Eigenes Arctus350, Evaluation und Qualität

- **[#179](https://github.com/Cola968/neral-helios/issues/179)** – Arctus350 Research: eigenes 350M-Modell gegen Qwen-0.5B/0.6B benchmarken und effizient verbessern
- **[#153](https://github.com/Cola968/neral-helios/issues/153)** – Arctus350 next: echte DE/EN Instruktionskompetenz statt nur NLL/EOS-Verbesserung
- **[#144](https://github.com/Cola968/neral-helios/issues/144)** – Gemini: unabhängiger Arctus350-Review zu Prompt-Echo, EOS und 1B/2B/SFT-Qualität (CPU-only)
- **[#142](https://github.com/Cola968/neral-helios/issues/142)** – Codex: Eigenständige Free-Generation-Diagnostik für Arctus350 (Echo, EOS, kurze Antworten, Verträge)
- **[#141](https://github.com/Cola968/neral-helios/issues/141)** – Arctus 350M: 2B-Checkpoint inventarisieren und echte RTX-4070-Baseline messen
- **[#121](https://github.com/Cola968/neral-helios/issues/121)** – Phoenix 350M: quality-first path to a usable Helios
- **[#119](https://github.com/Cola968/neral-helios/issues/119)** – Validate real 350M weight-only pilot, exact recovery and independent quality gates
- **[#114](https://github.com/Cola968/neral-helios/issues/114)** – Wissensbenchmark 200: 2B-Checkpoint gegen 2.5B prüfen
- **[#99](https://github.com/Cola968/neral-helios/issues/99)** – Roadmap: Arctus 350M first Helios model → Vega ~1B
- **[#39](https://github.com/Cola968/neral-helios/issues/39)** – Neral-50M I1: decisive win over 20M
- **[#36](https://github.com/Cola968/neral-helios/issues/36)** – Neral-50M G1: generation repair target after B2 100M

### P2 – Trainings-/Durchsatz-/Datensatzinfrastruktur

- **[#115](https://github.com/Cola968/neral-helios/issues/115)** – V8.1: wiederholte Stage-Rejections und effektiven Durchsatz untersuchen
- **[#108](https://github.com/Cola968/neral-helios/issues/108)** – V6 Fast2 optimization tracking
- **[#106](https://github.com/Cola968/neral-helios/issues/106)** – Scale V4 rollback cap
- **[#104](https://github.com/Cola968/neral-helios/issues/104)** – Scale V4 reliability: production-style promotion/eval improvements
- **[#103](https://github.com/Cola968/neral-helios/issues/103)** – Throughput V4: sustained 55k tok/s target
- **[#97](https://github.com/Cola968/neral-helios/issues/97)** – Arctus live training telemetry
- **[#72](https://github.com/Cola968/neral-helios/issues/72)** – HF data expansion for Neral-50M after M2.1

## Sämtliche 110 offenen Pull Requests (Integrationsinventar)

Diese PRs sind **nicht** jeweils ein eigenständiger Fehler. `main` als Basis bedeutet **noch nicht**, dass der PR gemergt ist. Die GitHub-Links lassen die Details prüfen. Ein massenhaftes Übernehmen wäre riskant.

| PR | Titel | Basisbranch | Zustand |
|---|---|---|---|
| [#217](https://github.com/Cola968/neral-helios/pull/217) | fix(SOLVE): enforce bounds when combining linear equation sides | `feature/neral-dual-tool-20261101` | Draft |
| [#216](https://github.com/Cola968/neral-helios/pull/216) | NERAL: echten lokalen Qwen-Helios-E2E-Test einmalig und sicher ausführen | `integration/neral-core-beta-candidate-20261101` | Draft |
| [#215](https://github.com/Cola968/neral-helios/pull/215) | NERAL Core Beta: gestapelte Qwen-/Helios-PRs erstmals gemeinsam gegen main testen | `main` | Draft |
| [#214](https://github.com/Cola968/neral-helios/pull/214) | NERAL Cloud: gemeinsame CAS-Aufgabenqueue mit begrenzten, ausfallsicheren Leases | `security/neral-dependabot-cloud-20261101` | Draft |
| [#213](https://github.com/Cola968/neral-helios/pull/213) | NERAL: zwei verschiedene sichere Werkzeuge pro Qwen-Antwort + Forschungsroadmap | `codex/neral-ollama-completion-integrity-20261101` | Draft |
| [#212](https://github.com/Cola968/neral-helios/pull/212) | NERAL: abgeschnittene Qwen-Antworten aus Kontext und Qualitäts-Passes ausschließen | `feature/neral-2022-capability-gate-20261101` | Draft |
| [#211](https://github.com/Cola968/neral-helios/pull/211) | NERAL: Dependabot als unabhängigen Cloud-Wartungshelfer vorbereiten | `main` | Draft |
| [#210](https://github.com/Cola968/neral-helios/pull/210) | NERAL 2022 Quality Gate: GPT-3.5-Ziel messbar statt behauptet | `codex/neral-quality-reasoning-20261029` | Draft |
| [#209](https://github.com/Cola968/neral-helios/pull/209) | Arctus350: Qwen-Teacher-Daten mit Lizenz-, Korrektheits- und TEST-Schutz prüfen | `integration/arctus350-research-gates-20261029` | Draft |
| [#208](https://github.com/Cola968/neral-helios/pull/208) | Arctus350: nahezu gleiche Trainingsdaten und Validierungs-Leaks offline erkennen | `data/arctus350-billion-token-streaming-20261029` | Draft |
| [#207](https://github.com/Cola968/neral-helios/pull/207) | Arctus350 für Vast.ai RTX 4090 sicher vorbereiten (ohne Miete oder Training) | `arctus350-chat-cloud-prep` | Draft |
| [#206](https://github.com/Cola968/neral-helios/pull/206) | NERAL Launch-Check: sechs Datenschutz- und Compliance-Risiken sichtbar prüfen | `data/arctus350-billion-token-streaming-20261029` | Draft |
| [#205](https://github.com/Cola968/neral-helios/pull/205) | Arctus350: Milliarden-BPE-Daten reproduzierbar und speicherschonend vorbereiten | `codex/neral-quality-reasoning-20261029` | Draft |
| [#204](https://github.com/Cola968/neral-helios/pull/204) | NERAL: optionales Qwen-Tiefdenken mit längeren Antworten und sicherem Kontext | `codex/neral-issue195-qwen-arctus-integration-20261029` | Draft |
| [#203](https://github.com/Cola968/neral-helios/pull/203) | NERAL: Qwen3.5 Assistant sicher mit Arctus350-Chat integrieren (#195) | `codex/arctus350-flask-experimental` | Draft |
| [#202](https://github.com/Cola968/neral-helios/pull/202) | Arctus350: Replay bei Trainingsvergleichen reproduzierbar halten | `fix/arctus350-per-example-invalid-label-20261029` | Draft |
| [#201](https://github.com/Cola968/neral-helios/pull/201) | NERAL: vorhandene Mehrschrittrechnung in den gemeinsamen Chat übernehmen | `codex/qwen-model-identity-20261029` | Draft |
| [#200](https://github.com/Cola968/neral-helios/pull/200) | Arctus350: paired-seed sampler risk reporting, not median-only evidence | `research/arctus350-padding-cost-audit-20261029-1300` | Draft |
| [#199](https://github.com/Cola968/neral-helios/pull/199) | Arctus350: PC-Jobs vor automatischen Pull-Request-Starts schützen | `research/arctus350-statistical-gain-gate-20261029` | Draft |
| [#198](https://github.com/Cola968/neral-helios/pull/198) | Arctus350: einmalige stochastische Vergleiche nur explorativ bewerten | `research/arctus350-statistical-gain-gate-20261029` | Draft |
| [#197](https://github.com/Cola968/neral-helios/pull/197) | NERAL: Ollama-Modellstand im Qwen-Verlauf abgleichen | `codex/neral-qwen-unified-20261029` | Draft |
| [#196](https://github.com/Cola968/neral-helios/pull/196) | NERAL: vorhandenen Qwen-Assistant in Arctus-Chat integrieren | `codex/arctus350-flask-experimental` | Draft |
| [#194](https://github.com/Cola968/neral-helios/pull/194) | Arctus350: length-only CPU audit of SFT padding and attention cost | `integration/arctus350-research-gates-20261029` | Draft |
| [#193](https://github.com/Cola968/neral-helios/pull/193) | Arctus350: theoretisches Trainingsbudget ehrlich quantifizieren statt 4T-Tokens blind anstreben | `integration/arctus350-research-gates-20261029` | Draft |
| [#192](https://github.com/Cola968/neral-helios/pull/192) | Arctus350 SFT: ungültige Assistant-Token-IDs vor Cross-Entropy sicher abfangen | `experiment/arctus350-equal-example-sft-loss-20261029` | Draft |
| [#191](https://github.com/Cola968/neral-helios/pull/191) | Arctus350: experimenteller SFT-Loss pro Beispiel statt dominierender langer Antworten | `research/arctus350-label-integrity-gate-20261029` | Draft |
| [#190](https://github.com/Cola968/neral-helios/pull/190) | Arctus350: falsche Mathe-Trainingsantworten und widersprüchliche Supervision vor SFT stoppen | `feature/arctus350-kv-cache-20261029` | Draft |
| [#189](https://github.com/Cola968/neral-helios/pull/189) | Arctus350: native grouped-query KV cache for efficient, exact-parity generation | `research/arctus350-frozen-ablation-contract-20261029` | Draft |
| [#188](https://github.com/Cola968/neral-helios/pull/188) | Arctus350: one-variable frozen A/B research protocol to prevent misleading training gains | `research/arctus350-statistical-gain-gate-20261029` | Draft |
| [#187](https://github.com/Cola968/neral-helios/pull/187) | Arctus350: Gewinne erst mit statistisch belastbaren gepaarten Messungen als Review-Kandidat markieren | `integration/arctus350-research-gates-20261029` | Draft |
| [#186](https://github.com/Cola968/neral-helios/pull/186) | Arctus350: sechs Forschungsbausteine gemeinsam integrieren und wissenschaftliche Metriken härten | `audit/arctus350-20261028-fresh-carry-baseline` | Draft |
| [#185](https://github.com/Cola968/neral-helios/pull/185) | Arctus350: VAL-only Grounding-Evaluator mit Überabstentions- und Halluzinationsprüfung | `research/arctus350-grounded-instruction-curriculum-v1-20261029` | Draft |
| [#184](https://github.com/Cola968/neral-helios/pull/184) | Arctus350: bilingual grounded QA and honest abstention curriculum – offline and heldout-safe | `audit/arctus350-20261028-fresh-carry-baseline` | Draft |
| [#183](https://github.com/Cola968/neral-helios/pull/183) | Arctus350: echte Assistant-Token-Ausbeute und EOS-Lernsignale vor dem nächsten Training messen | `audit/arctus350-20261028-fresh-carry-baseline` | Draft |
| [#182](https://github.com/Cola968/neral-helios/pull/182) | Arctus350: mathematische Train/VAL-Leaks trotz unterschiedlicher Fragen zuverlässig erkennen | `audit/arctus350-20261028-fresh-carry-baseline` | Draft |
| [#181](https://github.com/Cola968/neral-helios/pull/181) | Arctus350: reduce SFT padding compute with opt-in compact batches (CPU verified) | `audit/arctus350-20261028-fresh-carry-baseline` | Draft |
| [#180](https://github.com/Cola968/neral-helios/pull/180) | Arctus350 Research: echte gepaarte native Benchmark-Evidenz statt SFT-Blindflüge | `audit/arctus350-20261028-fresh-carry-baseline` | Draft |
| [#178](https://github.com/Cola968/neral-helios/pull/178) | Review #173: Herkunft und Stream-Abschluss sichern, Arctus350 optional vorbereiten | `codex/neral-chat-experience-stability` | Draft |
| [#177](https://github.com/Cola968/neral-helios/pull/177) | NERAL: messbar intelligentere Chat-Werkzeuge – sichere vollständige mehrstufige Mathematik | `codex/arctus350-flask-experimental` | Draft |
| [#176](https://github.com/Cola968/neral-helios/pull/176) | NERAL Orbit v0: persistente Agenten-Aufträge mit Freigabe und Status, ohne Fake-Autonomie | `codex/arctus350-flask-experimental` | Draft |
| [#175](https://github.com/Cola968/neral-helios/pull/175) | NERAL: optionales natives Arctus350 im Flask-Chat, experimentell und lokal | `codex/neral-chat-experience-stability` | Draft |
| [#174](https://github.com/Cola968/neral-helios/pull/174) | NERAL Tag 2: messbarer Produkt-/Agentenvertrag und Git-Ref-Sicherheitsprüfung | `plan/neral-autonomy-28d-20261027` | Draft |
| [#173](https://github.com/Cola968/neral-helios/pull/173) | NERAL: integrate truthful chat math and stabilize the real desktop/mobile workspace | `main` | Draft |
| [#172](https://github.com/Cola968/neral-helios/pull/172) | NERAL chat: recognize complete DE/EN arithmetic prompts without unsafe partial answers | `fix/arctus350-bare-arithmetic-native-chat-routing-20261028` | Draft |
| [#171](https://github.com/Cola968/neral-helios/pull/171) | NERAL: one-time read-only native Arctus350 evaluation on unseen carry v3 validation | `data/arctus350-carry-steps-curriculum-v3-20261028` | Draft |
| [#170](https://github.com/Cola968/neral-helios/pull/170) | docs: Tag 1 PR-Integrationsmatrix und Release-Gates (#133) | `plan/neral-autonomy-28d-20261027` | Draft |
| [#169](https://github.com/Cola968/neral-helios/pull/169) | Arctus350: bilingual verified carry-aware arithmetic v3, no further blind SFT | `docs/arctus350-20261028-evidence-and-next-steps` | Draft |
| [#168](https://github.com/Cola968/neral-helios/pull/168) | Arctus350: evidence-based full 28 October audit and safe post-trip handoff | `fix/arctus350-bare-arithmetic-native-chat-routing-20261028` | Draft |
| [#167](https://github.com/Cola968/neral-helios/pull/167) | NERAL local native chat: answer bare arithmetic via attributed safe calculator | `experiment/arctus350-separated-arithmetic-specialist-20261028` | Draft |
| [#166](https://github.com/Cola968/neral-helios/pull/166) | NERAL Arctus350: separate math-only experimental specialist, main chat weights protected | `experiment/arctus350-frozen-ffn-balanced-replay-20261028` | Draft |
| [#165](https://github.com/Cola968/neral-helios/pull/165) | Arctus350: conservative last-FFN-only replay with measured gradient balancing | `audit/arctus350-task-gradient-interference-20261028` | Draft |
| [#164](https://github.com/Cola968/neral-helios/pull/164) | Arctus350: read-only last-FFN math/format/chat gradient interference audit | `experiment/arctus350-balanced-math-v2-sft-20261028` | Draft |
| [#163](https://github.com/Cola968/neral-helios/pull/163) | Arctus350: safe FP32 arithmetic v2 rescue from locally pinned 256step candidate | `data/arctus350-balanced-arithmetic-v2-20261028` | Draft |
| [#162](https://github.com/Cola968/neral-helios/pull/162) | Arctus350: correct German/English arithmetic parity confound and protect prior heldouts | `experiment/arctus350-math-rescue-20261028` | Draft |
| [#161](https://github.com/Cola968/neral-helios/pull/161) | Arctus350: 256-step native math rescue, preserve 32/32 unseen exact words | `experiment/arctus350-unseen-exact-math-gpu-eval-20261028` | Draft |
| [#160](https://github.com/Cola968/neral-helios/pull/160) | Arctus350: rigorous 64-vs-96 unseen exact/math heldout GPU validation (today only) | `experiment/arctus350-unattended-guarded-curriculum-20261028` | Draft |
| [#159](https://github.com/Cola968/neral-helios/pull/159) | Arctus350: ONLY today 28 Oct safeguarded 3,630-example local FP32 training | `experiment/arctus350-targeted-exact-fp32-20261028` | Draft |
| [#158](https://github.com/Cola968/neral-helios/pull/158) | Arctus350: controlled native FP32 SFT from 64-step candidate with verified exact-output augmentation | `data/arctus350-exact-smallmath-curriculum-20261028` | Draft |
| [#157](https://github.com/Cola968/neral-helios/pull/157) | Arctus350: heldout-safe deterministic exact-output and small-math curriculum | `audit/arctus350-reference-answer-integrity-20261028` | Draft |
| [#156](https://github.com/Cola968/neral-helios/pull/156) | Arctus350: independent local math/code/format reference-answer audit before more training | `experiment/arctus350-training-readiness-20261028` | Draft |
| [#155](https://github.com/Cola968/neral-helios/pull/155) | Arctus350: genuine balanced-domain 240-step FP32 curriculum training on local RTX4070 | `experiment/arctus350-training-readiness-20261028` | Draft |
| [#154](https://github.com/Cola968/neral-helios/pull/154) | Arctus350: measure real clean SFT training readiness before next FP32 run | `fix/arctus350-no-zero-exact-review-20261028` | Draft |
| [#152](https://github.com/Cola968/neral-helios/pull/152) | Arctus350: reject manual-review success with 0% exact instruction compliance | `experiment/arctus350-clean64-dev-eval-20261028` | Draft |
| [#151](https://github.com/Cola968/neral-helios/pull/151) | Arctus350: actual 12-case read-only native evaluation of clean FP32 64-step candidate | `experiment/arctus350-2b-fp32-64step-clean-sft-20261028` | Draft |
| [#150](https://github.com/Cola968/neral-helios/pull/150) | Arctus350: real 64-step 2B FP32-master bilingual SFT on sanitized local data | `fix/arctus350-allturn-holdouts-20261028` | Draft |
| [#149](https://github.com/Cola968/neral-helios/pull/149) | Arctus350: close confirmed later-turn validation leakage with strict all-user-turn split gates | `fix/arctus350-release-integrity-20261028` | Draft |
| [#148](https://github.com/Cola968/neral-helios/pull/148) | Arctus350: real RTX4070 read-only greedy versus source-echo decoding A/B | `fix/arctus350-release-integrity-20261028` | Draft |
| [#147](https://github.com/Cola968/neral-helios/pull/147) | Arctus350: genuine read-only 1B/2B/SFT per-layer delta on local PC | `fix/arctus350-release-integrity-20261028` | Draft |
| [#146](https://github.com/Cola968/neral-helios/pull/146) | Arctus350: read-only cross-split numeric-template and multi-turn leakage audit | `fix/arctus350-release-integrity-20261028` | Draft |
| [#145](https://github.com/Cola968/neral-helios/pull/145) | feat: fail-closed CPU diagnostics for stored Arctus350 generation reports | `fix/arctus350-release-integrity-20261028` | Draft |
| [#143](https://github.com/Cola968/neral-helios/pull/143) | Arctus350: reject generated tokens after EOS (CPU-only integrity gate) | `fix/arctus350-release-integrity-20261028` | Draft |
| [#140](https://github.com/Cola968/neral-helios/pull/140) | Arctus 350M: block falsely promoted checkpoints via frozen-suite and generation integrity gates | `arctus350-chat-cloud-prep` | Draft |
| [#139](https://github.com/Cola968/neral-helios/pull/139) | NERAL P1: reproducible persona SFT drafts and honest RTX 4070 baseline evaluator | `main` | Draft |
| [#138](https://github.com/Cola968/neral-helios/pull/138) | NERAL: 28-day autonomous development roadmap + bounded coding-agent state machine | `main` | Draft |
| [#132](https://github.com/Cola968/neral-helios/pull/132) | NIS2: block fake scores, require frozen evidence and consistent checkpoint SHA | `main` | Draft |
| [#131](https://github.com/Cola968/neral-helios/pull/131) | NERAL Cloud Beta — independent 24/7-ready Qwen chat gateway and branded interface | `main` | Draft |
| [#130](https://github.com/Cola968/neral-helios/pull/130) | NERAL Assistant 4B: offline-reviewed SFT dataset pipeline + gated BF16 LoRA recipe | `main` | Draft |
| [#129](https://github.com/Cola968/neral-helios/pull/129) | NERAL Assistant v0: local Qwen3.5-9B reasoning + safe tools + UI integration | `main` | Draft |
| [#128](https://github.com/Cola968/neral-helios/pull/128) | Fix Arctus 350M evaluation tokenization to match SFT and chat | `arctus350-chat-cloud-prep` | Draft |
| [#127](https://github.com/Cola968/neral-helios/pull/127) | Arctus 350M: Grounded local documents in browser chat | `arctus350-chat-cloud-prep` | Draft |
| [#126](https://github.com/Cola968/neral-helios/pull/126) | Arctus 350M Chat v1: complete PC-off post-training preparation | `frontier/neral350-v1` | Draft |
| [#125](https://github.com/Cola968/neral-helios/pull/125) | Helios 180M FastChat v1: optimize for useful chatbot quality | `helios-1b-foundation` | Draft |
| [#124](https://github.com/Cola968/neral-helios/pull/124) | Helios 1B: new architecture foundation | `main` | Draft |
| [#123](https://github.com/Cola968/neral-helios/pull/123) | Phoenix Chatbot v1: project-owned behavior bootstrap pilot | `experiment/phoenix-blind-ab-audit-20260919` | Draft |
| [#122](https://github.com/Cola968/neral-helios/pull/122) | Helios Workspace 2.1: CUDA KV-cache Frontier 350 runtime | `main` | Draft |
| [#120](https://github.com/Cola968/neral-helios/pull/120) | Phoenix v0.4: mandatory native pilot guards, deduplicated launcher and offline CPU verification (draft) | `experiment/phoenix-sft-fp32-audit` | Draft |
| [#118](https://github.com/Cola968/neral-helios/pull/118) | Phoenix v0.3 DRAFT: original-data screens, local human review, guarded SFT and A/B contracts | `experiment/nextgen-v1-isolated` | Draft |
| [#117](https://github.com/Cola968/neral-helios/pull/117) | NextGen V1 DRAFT: real GPU resume proven; quality gates and ≤1M-token recovery in progress | `experiment/scale-v8-3b` | Draft |
| [#116](https://github.com/Cola968/neral-helios/pull/116) | V8.2: accepted-token 40k gate, truthful telemetry and single-load evaluator candidate | `experiment/scale-v8-3b` | Draft |
| [#113](https://github.com/Cola968/neral-helios/pull/113) | v7.1: exact Fast3 input path and lower host-sync overhead | `experiment/scale-v7` | Draft |
| [#112](https://github.com/Cola968/neral-helios/pull/112) | v7: fast guarded training with 1M exact recovery | `experiment/scale-v6` | Draft |
| [#105](https://github.com/Cola968/neral-helios/pull/105) | Integrate Claude Opus 5 through Experiential Labs | `main` | Draft |
| [#101](https://github.com/Cola968/neral-helios/pull/101) | Prep post-V6 confirmation and external bilingual audit | `frontier/neral350-v1` | Draft |
| [#100](https://github.com/Cola968/neral-helios/pull/100) | Helios app refresh v1 | `frontier/neral350-v1` | offen |
| [#98](https://github.com/Cola968/neral-helios/pull/98) | Research: Neral Nova 1B hybrid Mamba-3 architecture | `frontier/neral350-v1` | Draft |
| [#94](https://github.com/Cola968/neral-helios/pull/94) | Research: Neral 600M architecture search | `frontier/neral350-v1` | Draft |
| [#93](https://github.com/Cola968/neral-helios/pull/93) | Research: Arctus Prime 350M architecture bake-off | `frontier/neral350-v1` | Draft |
| [#91](https://github.com/Cola968/neral-helios/pull/91) | Neral 350M Frontier v1: 10B-token release architecture and training pipeline | `main` | Draft |
| [#89](https://github.com/Cola968/neral-helios/pull/89) | Repair: strengthen EOS termination and math transfer | `sprint/cpu-orion-data-hardening-20260909` | offen |
| [#88](https://github.com/Cola968/neral-helios/pull/88) | Data: CPU-only holdout, near-dedup and Math Pareto hardening | `data/mobile-feeder-v1` | offen |
| [#87](https://github.com/Cola968/neral-helios/pull/87) | ORION: CPU architecture accounting and crash-safe resume contracts | `perf/orion-neral-fastpath-v1` | offen |
| [#86](https://github.com/Cola968/neral-helios/pull/86) | Data: mobile-hosted 50M feeder + isolated training pilot | `train/neral50-chat-v2` | offen |
| [#85](https://github.com/Cola968/neral-helios/pull/85) | Perf: Neral 50M V3 ultra GPU fastpaths | `train/neral50-chat-v2` | offen |
| [#84](https://github.com/Cola968/neral-helios/pull/84) | Perf: Neral 50M Fastpath + Orion 100M fused architecture sprint | `main` | offen |
| [#83](https://github.com/Cola968/neral-helios/pull/83) | Docs: aktualisierte Neral-Roadmap bis 100M und Release | `main` | offen |
| [#67](https://github.com/Cola968/neral-helios/pull/67) | Compare hardware-balanced Helix-100M A2 | `main` | offen |
| [#44](https://github.com/Cola968/neral-helios/pull/44) | Start Helios app: responsive mobile + desktop shell | `main` | offen |
| [#43](https://github.com/Cola968/neral-helios/pull/43) | Start Helios v0.1 adaptive Android + Desktop app | `main` | offen |
| [#41](https://github.com/Cola968/neral-helios/pull/41) | Fix FineWeb2 revision blocking I1 training | `main` | offen |
| [#11](https://github.com/Cola968/neral-helios/pull/11) | Use KV-cache decoding in the local Helios app | `main` | offen |

## Konkrete Abarbeitungsreihenfolge

1. Den #215-Release-Kandidaten inklusive gemeinsam geprüftem Qwen-Hüllen-Fix und SOLVE-Bound-Fix **im Draft belassen**, weitere Fehler nur auf reproduzierter Grundlage beheben. `main` erst nach Nutzerreview/Freigabe.
2. Den aktiven, bereits getesteten Code genau einmal in einer frisch installierten bzw. freigegebenen Zielumgebung für Ollama, Deep/Tool-Protokolle, mobile App, PWA-Cache und Wiederherstellung abnehmen. Keine rückwirkende Behauptung von E2E-Testabdeckung.
3. Launch-Sicherheitsanforderungen von #206 prüfen, berechtigte Checklisten/Scanner konfliktfrei übernehmen und Datenschutz/Alters-/Rechtskontext für die geplante Distribution manuell klären.
4. PR-Graph und Relevanz/CI jeder alten Änderung einzeln bewerten; historische native Arctus-Experimente archivieren statt ungeprüft in das Qwen-Produkt zu mergen.
5. Arctus350-Weg getrennt mit echter Qualitätsevaluation, sauberen Daten-/Lizenzregeln und ausdrücklich freigegebenen GPU-Runs weiterführen.
6. Cloud-Auftragsqueue #214 und Cron-Worker #211 nur nach Prüfung ihrer Rechte/Kosten in eine reale, begrenzte Ausführungsarchitektur einbauen.

**Gegenwärtiger Status:** Ein nachgewiesener weiterer SOLVE-Bug ist behoben und integrierte CI ist grün. Der erfolgreiche dreiteilige echte Qwen-Test bestätigt nicht die übrigen offenen Produkt-/Modell-/Release-Gates. Ohne Freigabe keine automatischen Merges/Deployments oder Training.
