# NERAL: geprüfte Werkzeugwahl statt blindem Mehrfach-Reasoning

**Entscheidungsvorlage vom 08.11.2026** · Architektur- und Forschungsplan für NERAL 1.1/1.2. **Noch nicht implementiert, keine Qualitätsfreigabe.** Dieser Text verändert weder Modellgewichte noch die aktuelle NERAL-1.0-Releasekonfiguration.

## Entscheidung

**Go für einen kleinen, kontrollierten Router und deterministische Prüfwerkzeuge; No-Go für einen unbeschränkten ReAct-Agenten im Standardchat.**

Die Forschung und Praxistests sprechen für Tool-Use *bei Aufgaben, deren Ergebnis extern überprüfbar ist*. Sie begründen **nicht**, dass längeres Nachdenken, mehr Agentenschritte oder eine zweite LLM-Selbstkritik generell bessere Antworten liefern. Erst direkte, gepaarte Tests gegen NERAL ohne Router rechtfertigen eine Aktivierung.

Ziel: Das **gesamte NERAL-System** soll auf unbekannten Aufgaben korrekter werden. Dies ist ausdrücklich kein Beleg für verbesserte neuronale Gewichte oder ein eigenes, Qwen grundsätzlich überlegenes Sprachmodell.

## Vorhandene NERAL-Implementierung (Stand der Codeprüfung)

- `neral_local_chat_api.py`: LOW `/api/neral/chat/generate`, HIGH `/api/neral/chat/think`; gemeinsamer Chat-Lock, Modellherkunft und experimenteller Qwen3.5-HIGH-Pfad. Der HIGH-Pfad ist **nicht** gleichbedeutend mit einem universellen Agentenrouter.
- `neral_agent_autonomy.py`: separater Projektagent mit `AutonomousAgent`, erlaubten Aktionen (u. a. `solve`, `solve_math_symbolic`, `syntax_check`, `run_isolated_unit_test`, `web_search`), Stop-/Cancel-Regeln und Grenzen von bis zu 12 Schritten / 300 s per Default. Websuche nur nach expliziter Freigabe mit festem Suchbegriff; Änderungen benötigen Freigabe.
- `docs/NERAL_THINK.md`: exakte Prüfung **numerischer Endwerte**, höchstens eine Korrekturrunde, `verification_scope=numeric_final_only`. Keine verlässliche inhaltliche Verifikation beliebiger Texte.
- `neral_preview_server.py`: Preview zeigt `local_code_execution.available=false` und `project_files.available=false`. Syntax-Check prüft AST/Compiler, **führt keinen generierten Code aus**. Datei-/Bildanhänge sind nicht automatisch ausführbare Projektwerkzeuge.
- PR #461: experimenteller lokaler `HIGH=qwen3.5:9b` optional, `LOW=qwen3:8b`; PR #463 testet die reale GPU-Umschaltung separat. Kein globaler Modellwechsel ohne Release-Gate.
- Der 72er LOW/HIGH-Entwicklungslauf #42512932846 verwendet **beidseitig `qwen3:8b`** für `generate` vs `think`; er validiert *nicht* den Qwen3.5-Modellwechsel. Der 36er Cross-Model-Pilot aus Roadmap #454 ist kein unabhängiger Holdout.

**Konsequenz:** Vorhandene Werkzeuge **über eine schmale, neue Chat-Orchestrierung wiederverwenden**, keinen zweiten Agenten-Stack kopieren; Projektagent, Chatrouter und Live-GPU-Abnahme bleiben getrennt.

## Vorgeschlagene Architektur und Datenfluss

```text
Nutzerfrage + explizite Plugin-Freigaben
  -> Input/History/Origin/CSRF-Validierung (bestehende HTTP-Schicht)
  -> TaskRouter:
       a) deterministische Erkennung bei exakt unterstützter Arithmetik
       b) direkter Modellchat bei gewöhnlicher Konversation
       c) begrenzte strukturierte Modellentscheidung bei mehrdeutiger Aufgabe
  -> PolicyGate (Allowlist, Parameter-JSON-Schema, Nutzer-Consent, Datenabfluss)
  -> ToolExecutor (exakte Mathematik | Python-Syntax | später isolierte Tests | opt-in Web)
  -> Verifier mit Beweis/Ergebnisartefakt
       -> wenn externer Prüfer eindeutigen Fehler zeigt: höchstens 1 korrigierter Versuch
       -> sonst direkte Antwort / ehrlich nicht verifiziert
  -> Antwort mit Modell/Adapter/Tool/Scope/Status und gemessener Latenz
```

Die LLM-Entscheidung ist ein **Vorschlag**, niemals eine Autorisierung: Der Server entscheidet anhand aktivierter Fähigkeiten. Keine verdeckten Cloud-/Webaufrufe, keine automatischen Downloads, keine privilegierten Dateisystemoperationen.

### Router-Entscheidungen

1. **DIRECT:** Grüße, Erklärungen, kreative Texte und subjektive Fragen ohne überprüfbare externe Aktion: ein Modellaufruf, **kein Tool**. HIGH darf für schwierige Aufgaben verwendet werden, aber nicht jede Frage automatisch zweimal generieren.
2. **MATH_EXACT:** Nur bei sicher erkanntem, vom bisherigen AST-/Symbol-Solver unterstütztem Ausdruck. Tool berechnet den Endwert; Erklärung darf nicht fälschlich als vollständig geprüft gelten.
3. **PYTHON_STATIC:** Für ausdrücklich angeforderte Python-Syntaxprüfung; nur `ast.parse` und `compile` im bestehenden erlaubten Rahmen; keine Behauptung, dass Logik oder Laufzeitverhalten getestet wurden.
4. **PYTHON_TEST:** Erst nach eigener Sicherheits-/Sandbox-Abnahme. Tests in isolierter, ressourcenbegrenzter Umgebung ohne Netz, Home-Mount, Secrets oder Host-Schreibrechte. Bis dahin **nicht verfügbar** im normalen Preview-Chat.
5. **WEB_RESEARCH:** Nur nach ausdrücklicher Nutzerfreigabe und mit sichtbar ausgehendem Suchbegriff, Quellen und nicht vertrauenswürdigen Webinhalten als **Daten**, niemals Systemanweisung.
6. **PROJECT_AGENT:** Nur im bewusst gewählten Agentenmodus, mit Freigaben für Änderungen; kein impliziter Tool-/Dateizugriff durch bloße Chatfrage.
7. **UNKNOWN / INSUFFICIENT:** Keine erfundene Toolnutzung; offen kennzeichnen, wenn Prüfung nicht möglich ist.

### Router-Protokoll (Entwurf, kein implementierter API-Vertrag)

```json
{
  "decision": "DIRECT | MATH_EXACT | PYTHON_STATIC | PYTHON_TEST | WEB_RESEARCH | PROJECT_AGENT | INSUFFICIENT",
  "tool_arguments": {},
  "reason_code": "explicit_request | exact_match | needs_external_check | no_tool_needed",
  "requires_consent": false
}
```

Backend-seitige JSON-Schema-Prüfung: keine unbekannten Schlüssel, maximale Argumentlängen, sichere Pfade, bekannte Toolnamen und strikte Typen. Keine Reparatur durch unsichere JSON-/Code-Evaluierung. Pro Turn eine Entscheidung; erneutes Routen nur nach **nachweislichem** Werkzeugfehler, ansonsten begrenzter Fallback.

### Endzustände und echte Verifikation

`VERIFIED` (konkreter Test bestanden, mit Scope), `TOOL_FALLBACK` (deterministisch berechnet, **nicht** vom Modell), `UNVERIFIED`, `TOOL_FAILED`, `NOT_AUTHORIZED`, `TIMEOUT`, `CANCELLED`.

`verified=true` **nur** bei passendem unabhängigen Oracle. Numerische Gleichheit prüft keine Herleitung; Python-Syntax prüft keine Programmlogik; bestandener Unit-Test prüft nur seine Assertions; Webquellen allein verifizieren keine gesamte Antwort. Nie `verified=true` aus LLM-Selbsteinschätzung ableiten. Reale Artefakte statt bloßer „sieht gut aus“-Urteile.

## Ausführungsregeln für 12 GB RTX 4070 und lokale Daten

- **Ein Modell gleichzeitig im VRAM** und vorhandenen NERAL-Inferenz-Lock wiederverwenden. LOW/HIGH sequentiell laden; keine Parallel-Reflexion mehrerer 8–9B-Modelle.
- Zunächst maximal **3 Tool-Schritte** pro normaler HIGH-Anfrage und **1** gezielter Korrekturversuch. Konkrete Limits anhand real gemessener p95-Latenz abstimmen; weder minutenlange Loops noch unbegrenzte Re-tries.
- Kein Python `eval`/`exec`, keine Shell und keine vom Modell ausgelöste lokale Host-Ausführung. Für späteren Code-Runner separate, nachweislich isolierte Sandbox; standardmäßig aus.
- Echte Timeout-/Cancel-Zustände; Client-Abbruch zählt **nicht** als Nachweis, dass GPU-Inferenz bereits beendet wurde. Reale Windows-PC-Smoketests erforderlich.
- Web bleibt opt-in, Nutzer sieht den genauen Suchbegriff. Web-/Dateiinhalt kann Prompt-Injection enthalten; Tool-Policy wird nie durch den Inhalt verändert.
- Keine Rohprompts, Chats, Dateianhänge oder Antworten an GitHub; nur anonymisierte Aggregatmetriken. Modellname, Digest, Inferenzprofil, Adapterstatus und Tool-Herkunft in lokaler Antwort sichtbar.
- Experiment und Release getrennt: Feature-Flag **standardmäßig aus**; unveränderter LOW als Rollback. Kein Training, Gewichtswechsel, Cloud-API oder Auto-Merge.

## Messplan mit eingefrorenen, unabhängigen Aufgaben

**A/B-Vergleich** auf denselben bisher unbekannten Fällen, gleichen Modell-Digests, gleichem Profil und Tokenbudget:

- A: HIGH Qwen3.5 **ohne** neuen Router.
- B: HIGH Qwen3.5 **mit** Router/Tool-Policy.
- C: stabiler LOW Qwen3:8B als Produktbaseline. A vs B isoliert den Routereffekt, B vs C den Gesamtsystemeffekt.
- Mindestens 120 gesperrte Fälle gemäß #458: Mathematik 30, Logik/Reasoning 30, Python-Code und Debugging 40, reale lokale Tool-Aufgaben 20. Separates Entwicklungsset für Router-Verbesserungen. Gold-Oracle/Unit-Tests **vor** Ergebnissichtung fixieren.
- Messen: End-to-end Task-Erfolg, richtige/unnötige/verpasste Toolauswahl (Precision/Recall), gültige Parameter, richtige Nebenwirkungen, Sicherheitsverstöße, unberechtigte externe Aufrufe (muss 0 sein), Verifikationsfehler, HTTP-/OOM-/Timeout-Quote, p50/p95, GPU-VRAM und Erfolgsquote nach Korrektur.
- Für unstetiges Tool-Routing einzelne Fälle mehrfach ausführen (eigene Reliability-Metrik); Endzustände statt hübscher Antworttexte bewerten. Prompts oder Datensatzfamilien nicht vor Freigabe auf dieselben Holdoutfälle tunen.

**Go/No-Go:** Systemziel aus #454: B vs C **mindestens +10 Prozentpunkte**, exakter gepaarter McNemar p ≤ 0,05, keine kritischen Sicherheitsregressionen, keine Einzeldomäne >5 pp schlechter, reale Windows-/Paketabnahme grün. Zusätzlich muss **B vs A** einen positiven, reproduzierbaren Mehrwert zeigen; sonst Router **nicht** standardmäßig aktivieren. Wenn die Kriterien fehlen: Feature experimentell lassen und Resultate nicht als Intelligenzdurchbruch melden.

## Arbeitsfolge (Prognose, nicht bestätigter Releasezustand)

- **Bis 10.11. / NERAL 1.0:** laufende GPU-/Release-Gates #457/#458/#460, bekannte HIGH-HTTP-Fehler #459; **keinen neuen Router in den Release hineindrücken**.
- **11.–15.11. / NERAL 1.1 erster PR:** rein lokaler `TaskRouter` für DIRECT, MATH_EXACT, PYTHON_STATIC, mit JSON-Validator, PolicyGate und Fake-Backend-Tests. Kein Web-/Code-Execution-Automatismus.
- **16.–25.11.:** begrenzte HIGH-Korrekturschleife nur bei verifizierbarem Fehler, Opt-in-Web, unabhängige A/B-Messung; Aufteilung in kleine PRs.
- **Ende November–Dezember / 1.2:** isolierte Python-Tests, kontrollierte mehrstufige Aufgaben und Agent-UI **nur bei bestandenem Sicherheits- und Qualitätsgate**. Bei negativer Latenz-/Accuracy-Bilanz einfacher Plan-/Workflow statt ReAct-Schleife.

### Konkrete TDD-Reihenfolge

1. **Rote Tests**: normaler Smalltalk triggert null Tools; mehrdeutige Frage darf nicht web-suchen; Mathe-Exact liefert korrektes Provenienzlabel; ungültiges Tool-JSON wird geblockt; kein Tool ohne Consent.
2. Minimaler Router/PolicyGate als eigenständige Funktion, integriert in bestehenden `neral_local_chat_api.py`-Flow, dabei den vorhandenen gemeinsamen Lock beibehalten.
3. Fake-/Stub-Backend-Tests für Timeouts, fehlerhafte Toolresultate, Prompt-Injection in Web-/Dateitext, doppelte identische Aktionen, Cancel und fehlerhafte Session-/CSRF-Zustände.
4. Windows/Ubuntu CI + entpacktes ZIP/EXE + echte RTX-4070-Abnahme + A/B auf unabhängigen Fällen. Erst dann gezielte Feature-Freigabe.

## Bewertung externer Quellen

- **Anthropic Engineering: [Building effective agents](https://www.anthropic.com/research/building-effective-agents) (2024)** – empfiehlt möglichst einfache, komponierbare Workflows und weist auf Latenz-/Komplexitätskosten hin. **Unterstützt** schmalen Router statt freiem Multiagenten-System.
- **Anthropic: [Common workflow patterns](https://claude.com/resources/articles/common-workflow-patterns-for-ai-agents-and-when-to-use-them) (05.03.2026)** – evaluator/optimizer funktioniert bei **klaren Prüfkriterien**, nicht universell.
- **Google/Princeton: [ReAct](https://arxiv.org/abs/2210.03629)** – Beleg für die grundsätzliche Wirkung von Tool-Observation-Loops; anderes Modell/Benchmark, **kein NERAL-Wirksamkeitsnachweis**.
- **Docker: [Tool Calling with Local LLMs: A Practical Evaluation](https://www.docker.com/blog/local-llm-tool-calling-a-practical-evaluation/) (30.06.2025)** – beobachtete unnötige Calls, falsche Tools und Parameterfehler; misst Toolwahl und wiederholte Versuche. **Begründet** Router-/Argument- und False-Positive-Tests.
- **Microsoft et al.: [Thinkingbox](https://arxiv.org/html/2608.19741v1) (20.08.2026)** – bewertet tatsächlichen Zielzustand statt bloß plausible Toolketten; Warnung vor zufälligen Einzelerfolgen. Andere Domäne, **nicht auf NERAL übertragbare Erfolgszahlen**.
- **[OWASP LLM Top 10 2025](https://genai.owasp.org/llm-top-10/)** – Prompt Injection, Excessive Agency und unsichere Tool-Ausgaben; **begründet** Least-Privilege und explizite Freigaben.
- **Viet-Anh Nguyen: [Plan Once, Then Act](https://www.vietanh.dev/blog/2026-06-15-plan-once-then-act-small-model-agents) (15.06.2026)** – praxisnahe Erfahrung mit kleineren lokalen Modellen; Agentenloops können abbrechen oder unnötig langsam werden. **Anekdotischer Architekturhinweis**, nicht alleiniger Qualitätsbeleg.

## Bezug zum Projekt

Roadmap #454 · Sicherheit/Trainingsstopp #449 · HIGH-Integration #457 · Holdout #458 · konkrete Tool-Fixes #459 · Release #460 · Qwen3.5-Smoke PR #463.

**Kernaussage:** Erst messbar richtige Toolentscheidungen und eine wirklich externe Prüfung; dann – falls A/B einen Vorteil zeigt – mehr autonome Schritte. Keine Generalisierung aus bekannten 36-/72-Fällen und keine Gleichsetzung von Orchestrierung mit trainierten Modellgewichten.
