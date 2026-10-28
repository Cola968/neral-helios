# NERAL – Produktvertrag v0 und Coding-Agent-Abnahme (Tag 2)

**Vorbereitet am 28.10.2026 für Tag 2 (29.10.2026)** des [28-Tage-Plans](NERAL_28_DAY_PLAN.md). Dieses Dokument ist eine Spezifikation, **keine Behauptung, dass ein autonomer Agent bereits einsatzfähig wäre.** Basierend auf [Issue #137](https://github.com/Cola968/neral-helios/issues/137) und dem Nutzerwunsch, dass nach einem einmaligen Programmierauftrag tatsächlich weitergearbeitet wird.

## Eine echte Nutzerfrage – klare Bedeutung

> „Kannst du auch weiter am Projekt arbeiten und sind wir im Zeitplan?“
>
> „Ohne jede weitere Nachricht soll NERAL nach meinem freigegebenen Auftrag testen, reparieren und einen Draft-PR vorbereiten.“

**Produktziel:** NERAL ist ein verlässlicher Chat und ein **beaufsichtigter** Coding-Agent: *ein* bewilligter Auftrag soll von der Planung bis zum überprüfbaren Draft-PR durchlaufen, ohne dass für jede Teiloperation eine neue Chat-Nachricht nötig ist. Eine Folgeaufgabe, ein Merge, ein Deployment oder bezahlter Modellbetrieb ist davon **nicht** automatisch genehmigt.

**Verbindliche Modellidentität:** Das langfristig eigenständig trainierte **NERAL-native**-Modell ist **Arctus350** und nicht Qwen. **Qwen** darf nur als klar ausgewiesenes vorläufiges externes Backend dienen. Die gegenwärtige Flask-Byte-App (20M/40M/50M), die separate Arctus350-Linie und Qwen-Inferenz sind unterschiedliche Systeme. Werkzeugrechnungen dürfen niemals als Fähigkeiten nativer Gewichte ausgegeben werden.

## Zehn messbare Abnahmekriterien

| ID | Forderung | messbarer Abnahmetest | Ist-Stand / Gate |
| --- | --- | --- | --- |
| AC-01 | Expliziter Startauftrag mit GitHub-Issue, Repository und erlaubtem Branch. | Start ohne gültiges Issue oder mit Branch `main` wird **vor jedem Adapteraufruf** abgelehnt. | Zustandsmaschine mit CPU-Tests; **kein** echter Cloud-Ausführer. |
| AC-02 | Die KI arbeitet nach *einer* Freigabe mindestens drei abhängige Schritte weiter. | In isoliertem Beispiel: Issue → Plan → echter fehlgeschlagener Test → Patch → echter grüner Test; kein weiterer Nutzerprompt innerhalb des Scopes. | Simulierter Ablauf vorhanden, reale Coding-Adaption **NICHT implementiert**. |
| AC-03 | Grenzen und Abbruch. | Maximal 3 gescheiterte Reparaturzyklen, maximal **60 Minuten** pro Auftrag; Deadline, Abbruch und Adaptorausfall führen ohne Folgeaktion zum dokumentierten Stop. | Kontrolllogik mit Unit-Tests; blockierende Adapter brauchen eigene OS-Time-outs. |
| AC-04 | Isolation und Änderungsschutz. | Kein Commit auf `main`, `release` oder ungültige `agent/*`-Git-Refs; nur freigegebene Pfade, keine Secrets im Sandbox-Arbeitsbereich. | Ref-Validierung im Kern; **echte** Worktree-/Sandbox-Isolation fehlt. |
| AC-05 | Echtes GitHub-Artefakt statt Erfolgsbehauptung. | Test zeigt existierenden Issue, echte Tests, Commit-SHA und **Draft-PR** im zugewiesenen Repo; Agent stoppt bei `WAITING_FOR_REVIEW`. | Bisher nur Test-Doubles; Ende-zu-Ende **NICHT implementiert**. |
| AC-06 | Modell- und Antwort-Herkunft ist sichtbar. | Jede Antwort im Chat ist als natives NERAL-Byte-Modell, **Arctus350**, externes **Qwen**, Werkzeug oder lokale Regel erkennbar; fehlende Modelle erzeugen keine Fake-Antwort. | Flask-Verbesserungen in Draft-PR #173; echte Arctus350-Integration ausstehend. |
| AC-07 | Produktqualität vor Freigabe. | Fünf repräsentative DE/EN-Dialogfälle: Antwort, Unsicherheit, Tool-Quelle und Fehlermeldung ohne Irreführung; reale Modellqualität getrennt von Test-Fixtures und Rechner prüfen. | UI-/Tool-Tests in Draft-PR #173; native Dialogqualität **nicht bestanden oder nicht unabhängig belegt**. |
| AC-08 | Robustheit bei Fehlern. | Timeout, 401, 429, Netzwerkabbruch, ungültige Stream-Enden und abgebrochene Aufgabe führen zu verständlichem Zustand, kein stiller Erfolg oder unsichere Wiederholung. | Teilweise Flask-/Browser-Tests; Provider-E2E steht aus. |
| AC-09 | Budget und Privatsphäre. | Standard **0 €**, höchstens 25 € direkte Gesamtausgaben (ohne Strom) nur nach Einzelzustimmung; keine Modellgewichte/privaten Chats in Cloud, keine Secrets im Audit. | Policy formuliert; durch echten Kosten-/Berechtigungs-Adapter noch nicht erzwungen. |
| AC-10 | PC-aus-Betrieb nur bei vorhandener Cloud-Laufzeit. | Auftrag läuft auf **GitHub-hosted** oder vergleichbar isoliertem, freigegebenem Worker bei ausgeschaltetem Privat-PC; prüfbare Statuswechsel und bewusstes Stoppen. | Kein realer kontinuierlicher Worker, also **NICHT implementiert**. |

## Verbindlicher Interaktionsvertrag

1. **Vorabauftrag:** Der Benutzer definiert ein Issue, Repo, zulässige Pfade, Erfolgskriterium und Kostenrahmen; Start verlangt explizite Freigabe. Standardmäßig keine kostenpflichtigen Aufrufe.
2. **Bearbeiten:** Nur ein freigegebener Task; protokollierte Phasen `PLANNING → RED_TEST → IMPLEMENTING → VERIFYING → READY_FOR_DRAFT_PR → WAITING_FOR_REVIEW`. Bei Testfehlern höchstens maximal 3 Reparaturversuche. Keine ungeprüfte Ausgabe von Modelltext als Systembefehl.
3. **Stopp und Freigabe:** Bei Änderung des Auftrags, fehlender Autorisierung, fehlenden echten Tests, Kostenrisiko, beschädigter Sandbox, zeitlicher Grenze oder unklarer Entscheidung sicher abbrechen; **kein automatisch gestarteter Folgeauftrag**, niemals unaufgefordert mergen/deployen.
4. **Ehrlicher Fortschritt:** `RUNNING`, `WAITING_APPROVAL`, `FAILED`, `COMPLETED` sind Benutzerstatus; eine mit Mock-Tools grüne Zustandsmaschine darf nicht `COMPLETED` als echter GitHub-Patch melden.
5. **Arbeitsplatz/Cloud:** PC aus bedeutet keine lokale GPU; ein gestarteter Cloud-Worker benötigt eigene Ausführungserlaubnis und Sicherheitsgrenzen. Die tägliche ChatGPT-Automatisierung und GitHub Actions CI sind kein selbstständig code-schreibender Agent.

## Belegführung, Grenzen und Reihenfolge

- **Tag 1:** PR [#170](https://github.com/Cola968/neral-helios/pull/170) prüft PR-Abhängigkeiten, #173 enthält echte Flask-App-Verbesserungen. Beide bleiben Draft.
- **Tag 2:** Dieses Dokument mit zehn messbaren Kriterien plus deterministische CPU-Schutztests in eigenem Draft-PR. Die Kriterien sind vor der späteren Implementierung festgelegt.
- **Tag 3:** Persona-Dokument und zehn unabhängige Dialoge erst dann als Qualitätsevidenz werten, wenn Auswertung und Modellherkunft sichtbar sind.
- **Woche 2:** Adapter-Schicht, reale Sandbox und realer Issue→Test→Patch→CI→Draft-PR-Ablauf. Ohne diese Komponenten bleibt AC-02/05/10 ausdrücklich offen.
- **Woche 3/4:** Echte Arctus350-Evals, Sicherheit, Kosten und ehrlicher Go/No-Go-Release. **Keine** Hochrechnung aus den heutigen kleinen synthetischen Matheaufgaben auf breite KI-Intelligenz.

**Validierung:** `python -m unittest tests.test_neral_day2_contract tests.test_neral_dev_agent -v` auf GitHub-hosted Ubuntu; dies ist ein **messbarer Abnahmetest** für die Vertragsvollständigkeit und Git-Ref-Validierung, **nicht** für einen autonomen Coding-Agenten. Keine weiteren GPU-/PC-Jobs Donnerstag–Sonntag, keine kostenpflichtige Cloud-Inferenz, kein automatischer Merge, keine Model-Promotion.
