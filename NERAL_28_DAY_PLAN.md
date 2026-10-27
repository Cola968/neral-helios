# NERAL – 28-Tage-Entwicklungsprogramm (28.10.–24.11.2026)

**Ziel:** Ein eigenständiger, verlässlicher NERAL-KI-Chatbot mit klarer Persönlichkeit und ein **beaufsichtigter autonomer Coding-Agent**, der nach einem Startauftrag weiterplant, Codeänderungen umsetzt, Tests ausführt, Fehler korrigiert und einen Draft-PR vorbereitet – ohne nach jedem einzelnen Schritt eine neue Chat-Nachricht zu verlangen.

**Gegenstand:** Produktentwicklung, nicht automatisch bessere Modellgewichte. Qwen3.5-9B bleibt vorläufig die deklarierte Modellbasis. Das eigene Arctus-Modell und dessen NIS2 werden getrennt evaluiert.

## Wahrheit und Grenzen

- Ein GitHub-Actions-Tagescheck oder eine ChatGPT-Tagesmeldung kann den Zustand **erkennen**, schreibt dadurch aber nicht automatisch guten Code. Für tatsächliche Codearbeit ohne PC braucht es einen **dauerhaft/cloud-gehosteten Coding-Agenten** und passende Freigaben. GitHub Copilot Cloud Agent/Automations ist ein potenzieller Weg (kostenpflichtiger Copilot-Zugang erforderlich).
- Wir programmieren nicht direkt in `main`. Jede Änderung bekommt Tests und einen **Draft Pull Request**. Keine automatischen Merges, Deployments, Privilegienerweiterungen, Modellgewichte-Uploads oder Cloud-GPU-Trainings.
- ChatGPT-Sitzungen arbeiten nicht automatisch weiter, wenn sie beendet werden. Tägliche Aufgaben und Copilot-Automatisierungen sind getrennte, echte Scheduling-Systeme und müssen ausdrücklich eingerichtet sein.
- Alle externen Inferenz-/Cloud-Kosten standardmäßig **0 €**. Das vorgeschlagene maximale Experiment-Budget ist **10 € gesamt**, keine Ausgabegenehmigung. Nur nach ausdrücklicher Freigabe und Provider-Billing-Caps aktivieren.
- Es gibt keinen garantierten täglichen Codegewinn: Ein Tag gilt als Fortschritt, wenn nachvollziehbare grüne Tests, ein geklärter Fehler, ein verifizierter Benchmark, ein Review oder ein nutzbares Teilfeature nachgewiesen wurde; bei Blockern wird der Blocker offen gemeldet.

## Priorität und Qualitäts-Gates

**P0:** Einfache, ehrliche, stabile Chatantworten samt Fehlerbehandlung; Cloud-Provider-Authentifizierung, Budgetlimits, Herkunftskennzeichnung, keine Fake-Demos.
**P1:** Coding-Agent mit `RUNNING/WAITING_APPROVAL/FAILED/COMPLETED`, fixem Scope, Plan, Sandbox, getesteten Patch-Retries und PR-Review-Gates.
**P2:** Eigenständige NERAL-Persönlichkeit mit eigenen und rechtlich sauberen SFT-Beispielen, empirischer A/B-Evaluierung statt Marketing.
**P3:** Training von Arctus und 4B-Anpassungen erst nach Daten-/Lizenz-Audit, Budgetfreigabe und einer vollständigen Baseline; **NIS2-Arctus bleibt ungemessen**, bis echte gefrorene Rohmodellberichte vorliegen.

### Was bedeutet 'arbeitet weiter'?

Eine **vom Nutzer klar zugewiesene Aufgabe** darf im Agenten maximal 60 Minuten beziehungsweise 3 automatische Test-/Reparaturdurchläufe verarbeiten, Dateien im freigegebenen Repository ändern, Tests in einer Sandbox starten und einen Draft-PR erzeugen. Er hält bei offenen Produktentscheidungen, unklaren Berechtigungen, fehlenden Tests, Budgetüberschreitung, Sicherheitsgefahr und Aufgabenänderung an. Folge-Issues werden nicht eigenmächtig gestartet; für wiederkehrende Arbeiten braucht es eine separate konfigurierte Automatisierung.

## Jede Woche ein messbarer Meilenstein

| Woche | Termine | Verbindliches Ziel |
|---|---|---|
| 1 | 28.10.–03.11. | Produkt-/Persona-Vertrag, stabiler Cloud-Chat, erste grüne Tests |
| 2 | 04.11.–10.11. | Coding-Agent MVP: Issue → Plan → Patch → Test → begrenzter Retry → Draft-PR |
| 3 | 11.11.–17.11. | Mehrstufiges Arbeiten + Repo-Kontext + Persona und Benchmarkgrundlagen |
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
| 16.11.2026 · Tag 20 | **Lehrer/Schüler-Pfad** | 4B-QLoRA vs. 9B-Inferenz vs. Arctus-Rohmodell sauber trennen. | Reproduzierbarer Fine-Tuning-Plan ohne ungefragten Jobstart |
| 17.11.2026 · Tag 21 | **Wochenreview 3** | Persona, Halluzinationsrisiko, Codingleistung, mehrstufiges Arbeiten messen. | W3 Benchmarkbericht mit tatsächlichen Testzahlen |
| 18.11.2026 · Tag 22 | **Agenten-Stresstest** | Rate-Limits, Kontextabbruch, Test-Zeitüberschreitung, bösartige Issues. | Robustheits-/Abuse-Testbericht |
| 19.11.2026 · Tag 23 | **Modellherkunft** | UX: Transparenz über Qwen, eigene NERAL-Software und Arctus; keine KI-Fake-Antworten. | Release-gerechte Provenance-Ansicht |
| 20.11.2026 · Tag 24 | **Performance** | Cold-start, Antwortlatenz, Kontextkosten und API-Fehler messen (nur freigegebene Anbieter). | Performanceprofil mit p50/p95 oder explizit 'nicht gemessen' |
| 21.11.2026 · Tag 25 | **Veröffentlichen vorbereiten** | Sicherheit, Datenschutz, Impressum/AGB-Anforderungen, Providerkosten, Modellkarten prüfen. | Go/No-Go Checkliste |
| 22.11.2026 · Tag 26 | **Private Nutzerprobe** | Fünf reale Dialoge: Coding, Wissenschaft, Alltag, Mehrsprachigkeit, Unsicherheit. | Blindvergleich und Testerfeedback (wenn reale Tester vorhanden) |
| 23.11.2026 · Tag 27 | **Release Candidate** | Nur bestandene PRs, Changelog, Demo-Video-Skript, Abweichungen. | NERAL RC Gate, ggf. ausdrücklich blockiert |
| 24.11.2026 · Tag 28 | **Entscheidung & X-Launch** | Funktionierende Live-Demo, Messwerte, Kostenlage und Sicherheits-Gate final gegenprüfen. | Ehrlicher Launch-Entscheid und ggf. veröffentlichungsfertiger X-Post |

## Kostenentscheidung bei 10 €

| Option | Wert fürs Projekt | Bewertung |
|---|---|---|
| Kostenlose Entwicklung (GitHub CI + bestehende lokale Gewichte) | Planung, Tests, Pull Requests ohne zusätzliches Abonnement; kein autonomer bezahlter Agent | **Unbedingt zuerst** |
| GitHub Copilot Pro ($10/Monat, EUR-Endpreis je nach Steuern/Wechselkurs) | GitHub Cloud Coding Agent, Issue-Zuweisung und geplante Automatisierungen bei aktivierter Funktion; Credits begrenzt | **Beste erste kleine Investition für autonome Entwicklung**, aber 10 € können überschritten werden |
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

**Release-Blocker:** echter End-to-End-Chat mit Modell, reproduzierbare Coding-/Safety-Benchmarks, Budget-/Missbrauchsschutz, DSGVO/Datenschutz, Betriebsmonitoring und öffentliche rechtliche Pflichtangaben. Kein erfundener NIS2-Wert, keine versteckte Qwen-Herkunft.
