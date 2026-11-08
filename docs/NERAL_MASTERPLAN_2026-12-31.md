# NERAL MASTERPLAN: messbar besser als dieselbe Qwen-Basis bis 31.12.2026

**Stand:** 08.11.2026 · **Verantwortungsrahmen:** lokale RTX 4070 (12 GB), Ryzen 7 5700X, 32 GB RAM, keine externen LLM-APIs, keine bezahlten GPU-Stunden, keine heimliche Gewichts-Promotion. **Status: Entwicklungsziele, keine zugesagte Leistungsstufe.**

## Mission und Begriffsschärfe

**Nicht**: Qwen durch eine neue Oberfläche umbenennen oder Qwen3.5 als NERAL-eigenes Training ausgeben.

**Sondern**: Ein lokaler Assistent, der durch **eigene überprüfbare Software** (Werkzeugwahl, deterministisches Rechnen, isolierte Code-Prüfung, Planen, verifizierten Korrekturversuch) auf UNBEKANNTEN Aufgaben **mehr tatsächlich richtige Ergebnisse** erzeugt als dasselbe unveränderte Qwen3.5:9B-Modell ohne diese Zusatzarchitektur. Modell-Digest und Antwortprovenienz bleiben sichtbar.

Zwei Behauptungen, zwei getrennte Gates:
- **Systemintelligenz:** NERAL + Qwen3.5 löst mehr als direktes Qwen3.5 unter dokumentierten, fairen Rechen-/Tokenbedingungen. Tools und Code zählen als NERAL-Systemleistung.
- **Neuronale Modellintelligenz:** NERAL-LoRA auf exakt derselben Qwen-Basis übertrifft ungeänderte Gewichte bei deaktivierten Werkzeugen und identischen Inferenzbedingungen. Das ist das **spätere, wesentlich schwierigere** Ziel. Ohne positiven Blind-Holdout: keine Promotion.

## Verifizierter Ausgangspunkt

- 72 bekannte gepaarte Entwickleraufgaben auf lokaler RTX 4070, Lauf [#42512932846](https://github.com/Cola968/neral-helios/actions/runs/42512932846): LOW Qwen3:8B **32/72**, HIGH mit **derselben Qwen3:8B-Basis** **59/72**; HIGH durchschnittlich **26,74 s** vs LOW **6,42 s**, **0 als Tool-Antworten deklarierte HIGH-Resultate**. Das ist ein echter Entwicklungseffekt, **kein** unabhängiger Holdout und **kein** Beleg gegenüber Qwen3.5.
- Cross-Model-Entwicklungspilot laut [#454](https://github.com/Cola968/neral-helios/issues/454): Qwen3:8B 13/36, Qwen3.5:9B 31/36; vier HTTP-Fehler bei Qwen3.5. Nicht repräsentativ für allgemeine Qualität.
- Bereits vorhandene Verifikationsbasis: \`neral_high_verified_first.py\`, \`neral_think.py\`, \`neral_agent_autonomy.py\`, \`neral_agent_math.py\`. HIGH ist **kein** universal einsatzfähiger autonomer Agent.
- [PR #473](https://github.com/Cola968/neral-helios/pull/473): **experimenteller** Same-Base-Scorer und **manuell** ausführbarer 36er-Live-Sammler mit realem Qwen3.5, bestehendem HIGH und streng begrenztem zusätzlichem Arithmetic-Routing. Im Branch, nicht releasereif, kein fertiger GPU-Durchlauf.
- [PR #467](https://github.com/Cola968/neral-helios/pull/467): Forschungs-/Architekturdokumentation; [Issue #468](https://github.com/Cola968/neral-helios/issues/468) benennt Produktaufgaben und No-Go-Kriterien.

## Das NERAL-Prinzip

1. **Sicher klassifizieren, sonst direkt antworten.** Einfaches Gespräch = 1 Modellaufruf, keine erfundene Toolkette. Schwierige, überprüfbare Aufgabe = Werkzeug oder optional HIGH.
2. **Deterministische Werkzeuge statt LLM-Selbstlob.** Mathematik per erlaubtem AST/Solver, Programmierung zunächst Syntax und danach vollständig isolierte Tests. Externe Websuche nur explizit freigegeben.
3. **Ergebnis durch unabhängigen Beleg prüfen.** \`VERIFIED\` nur für reale, eng definierte Oracles; nicht für „das Modell hat zweimal zugestimmt“. Keine Verwechslung von Tool-Resultat mit Modellgewichten.
4. **Ein Korrekturversuch bei nachgewiesenem Fehler.** Hartes Schritt-, Zeit-, GPU-, Daten- und Berechtigungslimit. Kein unendlicher ReAct-Loop.
5. **Lernen nur aus nachweislichen Fehlerklassen.** Zuerst 2–3 konkrete systematische Schwächen beheben. Später LoRA auf verifizierten Daten, mit Retention-Gate und separater Adapter-Identität.

## Phasen mit harten Ergebnissen

| Termin | Lieferobjekt | Quantitatives Gate / Entscheidung |
|---|---|---|
| **08.–10.11.** | **NERAL 1.0 stabilisieren**: HIGH-Modellwechsel, Windows-Packaging, vier Qwen3.5-HTTP-Fehler, Cancel, Security, Provenienz | Main-/Windows-Release-Abnahme auf EXAKTEM SHA grün. Bei Fehlern 1.0 ohne unbewiesenen HIGH-Default ausliefern. Kein neues riskantes Training. |
| **11.–16.11.** | **Fairer 36er Dreiarm-Pilot**: A=Qwen3.5 direkt, B=aktuelles HIGH, C=HIGH plus extra geprüfter Tool-Router; private Daten nur auf PC | Exakte gleiche Qwen3.5-Modelldigests, User-Prompts, Profile; gepaarte richtige Antworten, Tool-Quote, HTTP/OOM, Median/p95, GPU, Fehlerfamilien. Pilot ist NUR Diagnose. Falls C=B oder schlechter: Router überarbeiten, nicht schönrechnen. |
| **17.–24.11.** | **NERAL 1.1** mit klein gehaltenem PolicyGate, Werkzeugrouter und realen Code-/Mathe-Oracles; 2–3 fehlgeschlagene Taskfamilien beheben | Test-first, keine ungefragten Web-/Datei-/Hostzugriffe. Unnötige Toolaufrufe auf Smalltalk und Prompt-Injection-Tests: 0. Windows/Ubuntu-Gates. Opt-in, Low-Rollback. |
| **25.11.–05.12.** | **Unabhängiger 120+-Holdout**, vorab gesperrt: Mathe30, Reasoning30, Python-Code40, lokale Tool-Fälle20 | **Ziel:** C vs A mindestens +10 Prozentpunkte, exakter gepaarter McNemar p≤0,05; zusätzlich **C vs B** positiver messbarer Nutzen. Kein Domänenrückgang >5 pp; null kritische Security-/OOM-Fälle; HTTP-Fehler explizit als falsch zählen. Falls FAIL: kein Überlegenheitslabel. |
| **06.–16.12.** | **NERAL 1.2:** sichere mehrstufige Code-/Projektaufgaben: Plan → lesen → vorschlagen → freigegebene Aktion → isolierter Test → Korrektur → Beleg | Endzustands-Tests statt Modellselbstbewertung. Max. drei normale Toolschritte / ein Korrekturversuch als Start, nur gemessen erweitern. Benutzerfreigabe für Projektänderungen. |
| **17.–25.12.** | **NERAL-Spezialisierung**: optionales LoRA-Experiment für Debugging/Tool-JSON/Projektverständnis, nur wenn die Basis stabil ist | Gleiches Basisgewicht/Prompt/Budget ohne Toolzugang gegen Adapter; unabhängige Retention, neue Beispiele, klarer Vorteil. Bei Fehlleistung Adapter deaktivieren, keine Überschreibung guter Gewichte. |
| **26.–31.12.** | **NERAL 1.3 Qualitätsbericht + ausgelieferte Version**: reale Produktbedienung, Regressionen, Offlinetauglichkeit, mehrstufige Fälle | 200+ gemischte Endzustandsaufgaben als ergänzende Validierung; getrennte NERAL-System-/LoRA-Scores; p50/p95, VRAM, API-Fehler, Sicherheitsfälle, bekannte Grenzen. Öffentlich nur belegte Angaben. |

### Leistungsziele bis 31.12. (nicht schon erreicht)

| Dimension | Zielwert |
|---|---|
| **Systemerfolg vs. Qwen3.5:9B direkt** | Mindestens **+10 pp** auf unbekanntem 120+-Set; ambitioniertes Stretch-Ziel **+15 pp** |
| **Tool-Router** | Mindestens **90 %** korrekte Wahl/Zurückhaltung auf unabhängigem Routing-Set; zusätzlich korrekte Argumente und Wirkung messen |
| **Antwortgeschwindigkeit** | Direkte Aufgaben möglichst nicht langsamer als Roh-Qwen; komplexe Aufgaben p50/p95 explizit ausweisen; keine künstliche Accuracy/s-Scheinmetrik |
| **Projektarbeit** | Reproduzierbare abgeschlossene, isoliert getestete Bugfixes statt plausiblem Code |
| **Zuverlässigkeit** | Keine unerlaubten Datenabflüsse, keine kritischen Tool-/Host-Sicherheitsverletzungen, keine OOMs im Freigabegate |
| **Eigene LoRA-Gewichte** | Wünschenswert, **keine 1.3-Abhängigkeit**. Ohne Blind-Holdout bessere Qualität niemals behaupten |

### Fairness des Benchmarks

- A **Raw Qwen3.5**, B **NERAL HIGH ohne neue Zusatzroute**, C **NERAL HIGH mit explizitem Tool-Preflight**; identische Gewichte/Profil/Fragen. Für gleiche Rechen- und Tool-Berechtigungen zusätzlich eine gleich ausgestattete **Qwen+Tools-Kontrollbaseline** aufnehmen, um den Mehrwert *unseres* Routers statt nur den Mehrwert eines Taschenrechners zu messen.
- Qwen3.5-9B wird standardmäßig NICHT neben Qwen3:8B gleichzeitig geladen. Exklusiver GPU-Slot und Freigabe nur nach vollständig grünem Release-SHA; kein automatischer Modell-Download.
- Keine Benchmark-Prompts oder per-case Modellantworten ins öffentliche GitHub. Nur signierte oder anderweitig intern nachvollziehbare Prüfsummen sowie Scalar-Aggregate. Publizierte Generatorfamilien sind **keine unbekannten Aufgabenstrukturen**.
- Codequalität per reproduzierbaren Unit-Tests in isoliertem, netzlosem Prozess; direkte Host-\`eval\`/\`exec\` ausgeschlossen. Webinhalte gelten als untrusted; Tool-Policy unüberschreibbar.
- Doppelte eigene Modellaufrufe erzeugen **Rechenmehrkosten**. Schnellere, aber falschere Ergebnisse sind keine Verbesserung. Primär tatsächlicher End-to-End-Erfolg, sekundär Latenz, Ressourcen, Zuverlässigkeit.
- Ein dominantes Rechen-Tool darf keine allgemeine Stärke suggerieren: domänenspezifische Scores und nicht-verifizierbare Konversationsaufgaben getrennt reporten.

## Welche „GPT-Leistung“ bis 31.12.2026?

**Zielkorridor, keine Messung und keine garantierte Äquivalenz:**

- Bei allgemeiner Konversation, alltäglicher Wissensarbeit und einfachen Coding-Fragen: ein lokal nutzbarer Assistent, dessen praktische Qualität **mindestens in die historische GPT-3.5-Klasse** aufsteigen soll. Der starke Qwen3.5-Unterbau macht dieses Ziel plausibel, aber noch nicht auf NERAL belegt.
- Bei **exakten Rechenaufgaben, abgesichertem Debugging und klaren Werkzeugaktionen**: Ziel sind auf ausgewählten Aufgaben **GPT-4-artige praktische Ergebnisse** durch externes Rechnen/Tests. Das bedeutet **nicht**, dass das 9B-Sprachmodell allgemein GPT-4-Niveau erreicht.
- **Kein seriöses Versprechen von genereller GPT-4-, GPT-4o-, GPT-5- oder GPT-6-Äquivalenz** bis Ende 2026 auf einer RTX 4070. Ein neuer GPT-Leistungsname ist keine Benchmark-Metrik. Für eine solche Aussage braucht NERAL dieselben unabhängigen, überprüften Benchmarks wie die jeweiligen Vergleichsmodelle.

Als externe Kontextquelle dient die **[offizielle Qwen3.5-9B-Modellkarte](https://huggingface.co/Qwen/Qwen3.5-9B)** mit vom Anbieter berichteten Ergebnissen (u. a. MMLU-Pro 82,5, LiveCodeBench v6 65,6). Diese Werte wurden **nicht** am lokalen NERAL-Setup reproduziert und sind keine direkten GPT-Äquivalenzwerte.

## Risikoregister und Konsequenzen

**R1 – Überanpassung an unsere 36/72 bekannten Familien.** Folge: nur Entwicklerdiagnostik; unabhängige neue Familien sperren, keine Tuning-Zyklen auf Holdout.

**R2 – Scheinverbesserung durch deutlich mehr GPU-Zeit.** Folge: Profil/Budget dokumentieren, separate Latenz- und Fairnessbaseline, einfache Anfragen ohne zweite Generation.

**R3 – CPU-Test grün, Windows-GPU-/Release-Job nicht.** Folge: kein Mergen/Defaultwechsel auf grünen Unit-Tests allein; echter Smokerun nachweisen.

**R4 – Tools sind falsch, zu breit oder unsicher.** Folge: striktes Input-Parsing, Sandbox, Freigaben, unabhängige Oracles, deterministische Fehler. Abbruch statt freie Ausführung.

**R5 – LoRA zerstört vorhandene Fähigkeiten.** Folge: Basismodell unverändert erhalten, Training optional, Retention/Holdout vor Promotion, reversible Switches.

**R6 – Öffentliche Erfolgsbehauptungen voreilig.** Folge: getrennte Labels \`system_gain_proven\`, \`weight_gain_proven\`, \`independent_holdout_verified\`. Alle zunächst \`false\`.

## Messbarer Abschluss am 31.12.

Die Veröffentlichung zählt nur dann als **„NERAL systemisch besser als Qwen3.5“**, wenn der unabhängige Vergleich bestanden ist. Ansonsten bleibt die Software nutzbar, aber die Entwicklungsbilanz nennt offen, welche Fehlergruppen und Tests noch nicht gewonnen wurden.

**Selbstbewusstes technisches Urteil:** Mit lokaler Orchestrierung und sicheren überprüfbaren Tools ist ein *domänenspezifischer* Vorsprung gegenüber dem selben Qwen3.5 realistisch. Ein allgemein stärkeres **neu trainiertes** 9B-Sprachmodell bis Jahresende ist keine vernünftige Release-Zusage. Wir investieren deshalb zuerst in reproduzierbar richtige Ergebnisse.
