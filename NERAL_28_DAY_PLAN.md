# NERAL – 28-Tage-Entwicklungsprogramm (28.10.–24.11.2026)

**Ziel:** Ein eigenständiger, verlässlicher NERAL-KI-Chatbot mit klarer Persönlichkeit und ein **beaufsichtigter autonomer Coding-Agent**, der nach einem Startauftrag weiterplant, Codeänderungen umsetzt, Tests ausführt, Fehler korrigiert und einen Draft-PR vorbereitet – ohne nach jedem einzelnen Schritt eine neue Chat-Nachricht zu verlangen.

**Verbindliche Identität:** NERAL soll langfristig **ein von Grund auf eigenständig trainiertes Arctus-Grundmodell** sein, nicht Qwen mit einer anderen Persönlichkeit oder einem LoRA-Adapter. Qwen3.5-9B ist ausschließlich ein offengelegtes, vorläufiges Produkt-Backend. Forschungs-/Trainingspriorität folgt [NERAL_NATIVE_MODEL_FIRST.md](NERAL_NATIVE_MODEL_FIRST.md); Arctus und NIS2 werden strikt getrennt evaluiert.

## Tagesfortschritt 28.10.2026 – Arctus 350M priorisiert

**Heute belegbar umgesetzt (nicht nur geplant):** [Draft PR #140](https://github.com/Cola968/neral-helios/pull/140)
härtet native 350M-Capability-, Generation- und Release-Checks.
[CI-Lauf 40672968178](https://github.com/Cola968/neral-helios/actions/runs/40672968178)
bestätigt **25 gezielte Tests plus 93 bestehende Chat-Tests grün (3 optional übersprungen)**.
Duplikate, NaN, leere Scoring-Spezifikationen, unterschiedliche eingefrorene
Cases, unbemerkte negative Kategorie-Deltas, fehlende Exact-Generation-
Werte und irreführende boolesche Release-Flags sind abgesichert.

**Nächster klarer Auftrag:** [Issue #141](https://github.com/Cola968/neral-helios/issues/141)
— tatsächlichen 2B-Checkpoint auf der RTX 4070 read-only inventarisieren,
separat vom 1B-SAFE-Stand unter denselben eingefrorenen Fällen bewerten,
EOS/Wiederholung/Instruktionen prüfen und erst danach kontrolliertes
Arctus-350M-Posttraining erwägen. Es wurde weder neu trainiert
noch der NIS2-Score durch diese Softwarearbeiten verbessert.
Geldkosten: **0 €** (Obergrenze 25 €, Strom ausgenommen).

## Wahrheit und Grenzen

- Ein GitHub-Actions-Tagescheck oder eine ChatGPT-Tagesmeldung kann den Zustand **erkennen**, schreibt dadurch aber nicht automatisch guten Code. Für tatsächliche Codearbeit ohne PC braucht es einen **dauerhaft/cloud-gehosteten Coding-Agenten** und passende Freigaben. GitHub Copilot Cloud Agent/Automations ist ein potenzieller Weg (kostenpflichtiger Copilot-Zugang erforderlich).
- Wir programmieren nicht direkt in `main`. Jede Änderung bekommt Tests und einen **Draft Pull Request**. Keine automatischen Merges, Deployments, Privilegienerweiterungen, Modellgewichte-Uploads oder Cloud-GPU-Trainings.
- ChatGPT-Sitzungen arbeiten nicht automatisch weiter, wenn sie beendet werden. Tägliche Aufgaben und Copilot-Automatisierungen sind getrennte, echte Scheduling-Systeme und müssen ausdrücklich eingerichtet sein.
- **Verbindliche Nutzergrenze vom 27.10.2026: höchstens 25 € zusätzliche direkte Projektausgaben insgesamt, ausdrücklich OHNE Haushaltsstrom; bevorzugt 0 €**. Nicht monatlich, kein automatisches Abo, keine Ausgabegenehmigung. Cloud-GPU, bezahlte Modell-APIs, Hosting und Abonnements zählen zum 25-€-Limit; lokale RTX-4070-Stromkosten zählen NICHT dazu. Cloud-/Trainings-/Inference-Kosten nur nach ausdrücklicher Zustimmung; Anbieter-Limits, Vorabkalkulation, Steuern, Gebühren und Restbudget vorher prüfen.
- Es gibt keinen garantierten täglichen Codegewinn: Ein Tag gilt als Fortschritt, wenn nachvollziehbare grüne Tests, ein geklärter Fehler, ein verifizierter Benchmark, ein Review oder ein nutzbares Teilfeature nachgewiesen wurde; bei Blockern wird der Blocker offen gemeldet.

## Priorität und Qualitäts-Gates

**P0:** Einfache, ehrliche, stabile Chatantworten samt Fehlerbehandlung; Cloud-Provider-Authentifizierung, Budgetlimits, Herkunftskennzeichnung, keine Fake-Demos.
**P1:** Coding-Agent mit `RUNNING/WAITING_APPROVAL/FAILED/COMPLETED`, fixem Scope, Plan, Sandbox, getesteten Patch-Retries und PR-Review-Gates.
**P2:** **Bestehenden Arctus 350M mit historisch über 2 Milliarden trainierten Tokens** priorisieren: 1B-SAFE und 2B-Fortsetzungscheckpoint getrennt inventarisieren, 350M-Branch `arctus350-chat-cloud-prep` als Quellbasis prüfen, echte RTX-4070-Rohmodell-Baseline vor kontrolliertem Posttraining. **Kein Neuanfang bei 50M/100M.**
**P3:** Produktpersönlichkeit und Agentenwerkzeuge weiterentwickeln, aber **Qwen-4B-LoRA-Fine-Tuning ist keine Priorität** für das native NERAL-Modell. SFT nur auf geeigneten **eigenen Arctus-Gewichten** nach Baseline, Daten-/Lizenz-Audit und Freigabe. **NIS2-Arctus bleibt ungemessen**, bis echte gefrorene Rohmodellberichte vorliegen.

### Was bedeutet 'arbeitet weiter'?

Eine **vom Nutzer klar zugewiesene Aufgabe** darf im Agenten maximal 60 Minuten beziehungsweise 3 automatische Test-/Reparaturdurchläufe verarbeiten, Dateien im freigegebenen Repository ändern, Tests in einer Sandbox starten und einen Draft-PR erzeugen. Er hält bei offenen Produktentscheidungen, unklaren Berechtigungen, fehlenden Tests, Budgetüberschreitung, Sicherheitsgefahr und Aufgabenänderung an. Folge-Issues werden nicht eigenmächtig gestartet; für wiederkehrende Arbeiten braucht es eine separate konfigurierte Automatisierung.

## Jede Woche ein messbarer Meilenstein

| Woche | Termine | Verbindliches Ziel |
|---|---|---|
| 1 | 28.10.–03.11. | Produkt-/Persona-Vertrag, stabiler Cloud-Chat, erste grüne Tests |
| 2 | 04.11.–10.11. | Coding-Agent MVP: Issue → Plan → Patch → Test → begrenzter Retry → Draft-PR |
| 3 | 11.11.–17.11. | Mehrstufiges Arbeiten + eigene Arctus-Modelllinie, Datenaudit und getrennte Rohmodell-Benchmarkgrundlagen |
| 4 | 18.11.–24.11. | Robustheit, Modellherkunft, Sicherheits-/Kosten-Gates, ehrlicher Releaseentscheid |

## Tägliche, überprüfbare Aufgaben

| Datum | Tagesfokus | Konkrete Arbeit | Nachweis / Done-Kriterium |
|---|---|---|---|
| 28.10.2026 · Tag 1 | **Bestandsaufnahme** | PR #129/#131/#132 prüfen; Abhängigkeiten und Merge-Reihenfolge dokumentieren. | Gereviewte PR-Matrix und gesperrte Release-Gates |
| 29.10.2026 · Tag 2 | **Produktvertrag** | Aus einer echten Nutzerfrage belastbare Anforderungen für NERAL Chat und Coding Agent formulieren. | Abnahmekriterien für NERAL Coding Agent v0 |
| 30.10.2026 · Tag 3 | **Persönlichkeitsprofil** | NERAL-Verhaltensprofil v1: natürliche deutsche Antworten, ehrlich, strukturiert, widerspruchsfähig. | Versioniertes Persona-Dokument und zehn Referenzdialoge |
| 31.10.2026 · Tag 4 | **Erste Persona-Tests** | Deterministische Tests für Herkunft, Sprache, Unsicherheitskommunikation und keine erfundenen Werkzeuge. | Automatisierte Persona-/Transparenz-Gates |
| 01.11.2026 · Tag 5 | **Cloud-E2E** | Hochwertige Cloud-UX: Chat starten, Fehler beheben, Verlaufsmodell testen. | Reproduzierbare Browser-Journey mit simuliertem Backend |
| 02.11.2026 · Tag 6 | **Reliability** | Fehlerfälle: Timeout, Provider-Ausfall, 401, 429, leere Antworten, erneuter Versuch. | Fehlerregressionen grün |
| 03.11.2026 · Tag 7 | **Wochenreview 1** | Diff, CI, Abdeckung und Risiken prüfen; nur belastbare Veränderungen zählen. | W1 Scorecard, technische Schulden priorisiert |
| 04.11.2026 · Tag 8 | **Coding-Agent-Vertrag** | Zustandsmaschine definieren: PLAN → PATCH → TEST → REVIEW → RETRY → PR → STOP. | Agenten-Protokoll, Sicherheitsmodell, Cancel-Verhalten |
| 05.11.2026 · Tag 9 | **Werkzeug-Sandbox** | Zugelassene Dateipfade, Git-Branch-Isolation, Prozess- und Netzfreigaben minimalisieren. | Fail-closed Sandbox-Tests |
| 06.11.2026 · Tag 10 | **Issue-Planer** | GitHub Issue in Teilaufgaben + Testdefinition zerlegen; ohne Codeänderungen. | Aufgabenplanung mit Akzeptanzkriterien |
| 07.11.2026 · Tag 11 | **Patch-Agent v0** | Ersten kleinen Patch in isoliertem Arbeitsbereich erzeugen (kein direktes main-Write). | Lokaler Testfall: Issue→Patch |
| 08.11.2026 · Tag 12 | **Autonomer Reparaturzyklus** | Nach fehlgeschlagenem Test selbst Ursache prüfen und begrenzt nachbessern. | Maximal drei Retry-Zyklen mit Erfolgs-/Abbruchnachweis |
| 09.11.2026 · Tag 13 | **PR-Erstellung** | Nach bestandenen Tests Draft-PR + nachvollziehbare Änderungszusammenfassung anbieten. | Issue→Draft-PR-Demonstration |
| 10.11.2026 · Tag 14 | **Wochenreview 2** | Coding-Agent-End-to-End und Sicherheits-Red-Team-Prüfung. | W2 Agent-Evaluationsbericht |
| 11.11.2026 · Tag 15 | **Beaufsichtigter Weiterlauf** | Option: gestartete Agentenaufgabe läuft in Cloud weiter, auch wenn Chat und PC aus sind. | Unterbrechungs-/Fortsetzungsprotokoll |
| 12.11.2026 · Tag 16 | **Repo-Knowledge** | Dateien, Architektur und Tests gezielt lesen; keine ungeprüften Internetanweisungen übernehmen. | Repo-Kontext und Zitierfähigkeit testen |
| 13.11.2026 · Tag 17 | **Mehrstufige Aufgaben** | Drei abhängige Teilaufgaben unter einem Arbeitsauftrag nacheinander ausführen. | Task-Graph mit überprüfbarem Abschluss |
| 14.11.2026 · Tag 18 | **Besseres Coding** | Einfache Python-/JavaScript-Aufgaben auf einem eingefrorenen internen Testsatz messen. | Vorher/Nachher Coding-Gates |
| 15.11.2026 · Tag 19 | **NERAL-Personality-Daten** | Eigene urheberrechtlich zulässige Gespräche kuratieren; Eval-Daten strikt aussparen. | Versionierter Datenmanifest-Entwurf |
| 16.11.2026 · Tag 20 | **Arctus 350M fortsetzen** | Historische ~2.073B–2.077B-Token-Linie und 1B-SAFE voneinander trennen; echten 350M-Code und Checkpoint-Lebenslauf dokumentieren; Qwen strikt getrennt halten. | 350M-Inventur, reproduzierbares RTX-4070-Baseline-Protokoll und geprüfte Posttraining-Gates – kein unautorisierter Jobstart |
| 17.11.2026 · Tag 21 | **Wochenreview 3** | Persona, Halluzinationsrisiko, Codingleistung, mehrstufiges Arbeiten messen. | W3 Benchmarkbericht mit tatsächlichen Testzahlen |
| 18.11.2026 · Tag 22 | **Agenten-Stresstest** | Rate-Limits, Kontextabbruch, Test-Zeitüberschreitung, bösartige Issues. | Robustheits-/Abuse-Testbericht |
| 19.11.2026 · Tag 23 | **Modellherkunft** | UX: Transparenz über Qwen, eigene NERAL-Software und Arctus; keine KI-Fake-Antworten. | Release-gerechte Provenance-Ansicht |
| 20.11.2026 · Tag 24 | **Performance** | Cold-start, Antwortlatenz, Kontextkosten und API-Fehler messen (nur freigegebene Anbieter). | Performanceprofil mit p50/p95 oder explizit 'nicht gemessen' |
| 21.11.2026 · Tag 25 | **Veröffentlichen vorbereiten** | Sicherheit, Datenschutz, Impressum/AGB-Anforderungen, Providerkosten, Modellkarten prüfen. | Go/No-Go Checkliste |
| 22.11.2026 · Tag 26 | **Private Nutzerprobe** | Fünf reale Dialoge: Coding, Wissenschaft, Alltag, Mehrsprachigkeit, Unsicherheit. | Blindvergleich und Testerfeedback (wenn reale Tester vorhanden) |
| 23.11.2026 · Tag 27 | **Release Candidate** | Nur bestandene PRs, Changelog, Demo-Video-Skript, Abweichungen. | NERAL RC Gate, ggf. ausdrücklich blockiert |
| 24.11.2026 · Tag 28 | **Entscheidung & X-Launch** | Funktionierende Live-Demo, Messwerte, Kostenlage und Sicherheits-Gate final gegenprüfen. | Ehrlicher Launch-Entscheid und ggf. veröffentlichungsfertiger X-Post |

## Budget und GPU-First-Strategie – absolute Obergrenze 25 €

**Höchstens 25 € an zusätzlichen direkten Projektausgaben (Cloud/Abos/Modelldienste), ohne Haushaltsstrom, nicht pro Monat.** Das ist keine Genehmigung, etwas zu kaufen. Standard bleibt **0 €**. Hardware: eigene NVIDIA GeForce RTX 4070 mit 12 GB VRAM, deren Laufzeit nicht immer verfügbar ist.

1. **Zuerst bestehenden Arctus 350M weiterentwickeln:** Das Modell hat bereits historisch über zwei Milliarden Tokens verarbeitet. 1B-SAFE-Checkpoint und 2B-Foundation-Fortsetzung separat auditen; native 350M-Codebasis vom Branch `arctus350-chat-cloud-prep` verwenden. **Kein Neustart bei 50M/100M**, keine ungeprüfte Checkpoint-Promotion. Qwen-4B-LoRA ist nicht der Weg zum gewünschten Originalmodell; 9B-Inferenz bleibt nur offengelegte Übergangs-Software.
2. **Bei knapper VRAM-Kapazität Speicher sparen:** kurze Sequenzen, kleine Batches, Gradientenakkumulation, BF16 nur wenn unterstützt, LoRA, Checkpointing, CPU-Offload nur wenn nötig. Quantisierungsverfahren erst nach konkretem Qualitätsvergleich einsetzen. Nicht unsichtbar eine kleinere/andere Modellbasis behaupten.
3. **Arctus 350M ist der primäre Forschungsweg:** eigene, historisch von Grund auf trainierte Gewichte und eingefrorenes NIS2-Eval unabhängig von Qwen. Der 350M-Code befindet sich in separaten GitHub-Branches; fehlender `main`-Ordner ist kein fehlendes Projekt. Erst aktuellen Checkpoint-/Tokenizer-Status verifizieren, dann kontrolliertes Posttraining auf lizenzgeprüften Daten ohne Benchmark-Kontamination.
4. **Erst nach belegtem lokalen Engpass Cloud nutzen:** Optional für einen *vorher freigegebenen* kurzen Lauf auf geeigneter GPU. Maximaler Vorschlag: bis 15 € GPU, bis 5 € echte Evaluation/Cloud-Inferenz, mindestens 5 € Reserve – **Gesamtlimit 25 €** inklusive eventueller Steuern/Gebühren, **ohne Stromkosten**. Kein automatischer Folgekauf, kein Abo ohne eigene Freigabe.
5. **Kein Erfolg ohne Test:** reales Vorher/Nachher-Eval inklusive deutscher Gesprächskonsistenz, Aufgaben-/Coding-Leistung, Halluzinationskontrolle, Latenz, Kosten.

### Hardware-Schutz für lange RTX-4070-Trainingsläufe

- **100 % GPU-Auslastung allein ist normal** und kein Schadensindikator. NVIDIA gibt für die RTX 4070 200 W Total Graphics Power und 90 °C maximale GPU-Temperatur an (Referenzdesign; Partnerkarten abweichend): https://www.nvidia.com/de-de/geforce/graphics-cards/40-series/rtx-4070-family/
- **Konservative interne Betriebsziele:** GPU-Core möglichst <80 °C, bei längeren 80–85 °C Kühlung prüfen, bei dauerhaft >85 °C oder auffälligen Lüfter-/Temperaturproblemen unterbrechen; zusätzlich VRAM/Hotspot und CPU-Temperatur beachten, falls Sensoren verfügbar. Diese Werte sind vorsichtige Projektziele, keine offiziellen NVIDIA-Abschaltgrenzen.
- GPU-Lüfter, Staub, Netzteil, Umgebungsluft, Stromstecker und Wartung sind Teil des Hardware-Risikos. Training kann Lüfterverschleiß und Stromverbrauch erhöhen; keine Garantie für Null-Risiko.
- Auf Windows kann `nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,power.draw,memory.used,memory.total --format=csv -l 5` Temperaturen, Last, Watt und VRAM während des Laufs anzeigen. Bei Sensorausfällen oder VRAM-OOM sicher abbrechen statt blind weitertrainieren.
- Keine GPU-Übertaktung, Mining-ähnliche Dauerlast oder automatische Nacht-Trainingsjobs ohne explizite Zustimmung. Training mit Checkpoints, damit kurze verfügbare PC-Zeit ausreicht.
- **Strom zählt ausdrücklich NICHT zum 25-€-Budget**; nur als optionale technische Information: Beispiel **300 W gesamte PC-Aufnahme × 10 Stunden = 3 kWh**; bei angenommenen 0,35 €/kWh wären das **1,05 €**, nicht nur die GPU-Leistung. Tatsächliche Werte messen, nicht raten.

## Kostenvergleich (nur mögliche Dienste, keine Kaufempfehlung)

| Option | Wert fürs Projekt | Bewertung |
|---|---|---|
| Kostenlose Entwicklung (GitHub CI + bestehende lokale Gewichte) | Planung, Tests, Pull Requests ohne zusätzliches Abonnement; kein autonomer bezahlter Agent | **Unbedingt zuerst** |
| GitHub Copilot Pro (US-Dollar-Listenpreis, Preis vor Kauf prüfen) | Kann Cloud-Coding-Aufgaben ermöglichen, verbraucht aber fast die Hälfte des **gesamten** 25-€-Rahmens schon für einen Monat | **Vorläufig nicht kaufen; GPU-First-Training priorisieren** |
| Hugging Face PRO ($9/Monat) | $2 monatliche inkludierte Inferenz-Credits, kein dauerhaft kostenloser 9B-GPU-Server | Nicht erste Wahl für Coding-Autonomie |
| Extra Inferenz-Guthaben | Reale Cloud-Chats/Baseline; keine automatische Modellverbesserung | Nur falls begrenzte Live-Tests blockiert sind |

**Quelle/Prüfpflicht:** https://github.com/features/copilot/plans ; https://docs.github.com/en/copilot/how-tos/use-copilot-agents/cloud-agent/create-automations ; https://huggingface.co/docs/inference-providers/pricing ; Tarife und Anspruch bei Kauf erneut überprüfen.

## Tagesbericht (einheitliches Format)

1. **Heute geändert:** SHA, PR, konkrete Dateien und Tests (oder 'keine Änderung').
2. **Funktioniert wirklich:** Beleglink zu CI/echten Modellantworten (nicht Mock).
3. **Größter Blocker:** Problem und nächster kleinster prüfbarer Schritt.
4. **Nächster Auftrag:** genau ein Issue mit Scope, Akzeptanzkriterien und Stoppbedingungen.
5. **Kosten:** Bezahlte Nutzung: 0 €, falls keine Freigabe; unbekannte tatsächliche Anbieter-Kosten als 'nicht verifiziert' kennzeichnen.

## GitHub-Umsetzung

- [NERAL Assistant lokal PR #129](https://github.com/Cola968/neral-helios/pull/129)
- [NERAL Cloud Beta PR #131](https://github.com/Cola968/neral-helios/pull/131)
- [NIS2 Integritäts-Gates PR #132](https://github.com/Cola968/neral-helios/pull/132)

**Release-Blocker:** echter End-to-End-Chat mit Modell, reproduzierbare Coding-/Safety-Benchmarks, Budget-/Missbrauchsschutz, DSGVO/Datenschutz, Betriebsmonitoring und öffentliche rechtliche Pflichtangaben. **NERAL-native** darf nur beworben werden, wenn tatsächlich eigene Arctus-Gewichte aktiv sind und unabhängig geprüfte Leistung nachgewiesen wurde; ein Qwen-Backend ist klar als Übergang zu kennzeichnen. Kein erfundener NIS2-Wert, keine versteckte Qwen-Herkunft.
