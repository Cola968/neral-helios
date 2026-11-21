# NERAL V4.1 — 200 Aufgaben, unabhängiger Blindtest

**Status: vorbereitet, nicht durchgeführt.** Nach erfolgreicher V4.1-Live-Abnahme
werden 200 neue, noch nicht im Trainings-, Regressionstest- oder Erfahrungsdatensatz
verwendete Aufgaben erstellt. Das private JSONL und seine Lösungen werden nicht in
GitHub veröffentlicht, nicht zum Prompting hinzugefügt und nicht in AutoLab
importiert. Kein Score wird aus dem 20/20-Web-V4-Test extrapoliert.

## Distribution
- \`everyday\`: 35 natürliche Erklärungen, Wissensfragen und Dialoge
- \`multiturn\`: 40 Folgen mit referenziellen Anschlussfragen und Themenwechsel
- \`live_web\`: 35 aktuelle Nachrichten/Zahlen mit zum Testzeitpunkt nachprüfbaren Quellen
- \`math_logic\`: 30 geprüfte Rechen-, Constraint- und Z3-Aufgaben
- \`programming\`: 25 neue Programme/Fehlerkorrekturen, ausführbare Tests
- \`writing\`: 20 Textaufgaben, verblindete unabhängige Bewertung
- \`uncertainty\`: 15 Unsicherheits-, Faktenkorrektur- und Begrenzungsfälle

**Jede Aufgabe läuft mit Amber, Quartz und Obsidian** (600 Antworten).
Entscheidend sind pro Kategorie Antwortkorrektheit, Belegqualität,
Verhalten bei Kontextwechseln, Wiederherstellbarkeit, Latenz je Modus und
GPU-Speicher. Als „verifiziert“ zählen nur unabhängige Prüfungen. Web-Quellen
müssen neu gelesen sein; gespeicherte Ergebnisse reichen nicht.

Die Datei liegt lokal außerhalb des Repos. Vor dem Ausführen:
\`python -m scripts.neral_v41_blind200 --sealed D:/neral-private/v41/sealed200.jsonl --manifest-out D:/neral-private/v41/manifest.json\`

Anschließend wird ein separater Live-Runner mit Freigabe und Limits gestartet;
die Validierung allein führt keine Web-, GPU- oder HTTP-Aufrufe aus. Eine
statistische Nähe zur Spitze lässt sich aus 200 Aufgaben mit je kleinen
Kategorien nicht zuverlässig für alle Domains behaupten.

## Live-Abnahme

Owner-Staging ist nur über den ausdrücklich freigegebenen, branchgebundenen
Workflow `neral-v41-owner.yml` zulässig. Vor einem Merge müssen
die echte 7-Kategorien-/3-Alias-HTTP-Abnahme sowie der HTTP-200-Nachtest
auf Port 9690 bestätigt sein. Bei Fehlern gilt der vollständige Rollback.

## Implementierter manueller Blindtest (21.11.2026)

Pipeline: `.github/workflows/neral-v41-blind200-manual.yml`. Ausschließlich
`workflow_dispatch` auf `main` mit
`approval=NERAL-V41-BLIND200-APPROVED` und `mode=full` führt die echte
200×3-Abnahme auf `W39-GAMINGPC-Neral` aus. Ein separater CPU-Preflight
muss bestehen. Anschließend wird die gepinnte App mit Backup/Rollback auf
den isolierten Benchmark-API-Pfad gebracht; der Owner-Smoke prüft sieben
Chatkategorien und die drei Legacy-Aliase.

Der Generator `scripts/neral_v41_blind200_build.py` erzeugt den versiegelten,
mit SHA-256 attestierten privaten JSONL auf dem **lokalen Windows-Rechner**.
Das Archiv wird mit historischen v39_benchmark/live_verified-Fragen verglichen;
fehlen überprüfbare Archiv-Provenienzen, scheitert die Versiegelung. Die
20 aktuellen Meldungs- und 15 Repo-Zahlenreferenzen stammen vor dem ersten
Modellaufruf aus eigenständig gelesenen RSS/API-Antworten. Fehlende
aktuelle Quellendaten führen zum Abbruch, nicht zu ausgedachten Goldwerten.
Historische Dubletten werden normalisiert ausgeschlossen; eine vollständige
semantische Garantie für „noch nie thematisch gesehen“ kann diese Prüfung
nicht leisten.

Der Runner `scripts/neral_v41_blind200_live.py` verwendet für jeden Fall eine
frische HTTP-/CSRF-Sitzung. Multi-Turn-Vorläufe spielen tatsächlich generierte
Assistenznachrichten in die Folgefrage zurück. `benchmark_isolated=true`
muss vom Server bestätigt sein; damit entfallen historische Antwortwiederholung,
Speicher-Hints und die automatische Sammlung verifizierter Trainingsbeispiele.
Bei fehlender Bestätigung wird **vor** dem Lesen der privaten Prompts abgebrochen.

Zahlen und Logik werden exakt mit unabhängigen Referenzen bewertet, Code in
dem bestehenden, netzlosen und schreibgeschützten Docker-Image
`python:3.11-alpine` ausgeführt, Webtitel/Zahlen gegen vorab unabhängige
Live-Snapshots **und** tatsächlich gelesene passende Seiten geprüft.
Die Auswertung enthält Treffer, Fehler, unbewertete Fälle, mittlere sowie
maximale Latenz je Stufe/Kategorie. Für freie Erklärung und Schreiben wird
bewusst keine vorgetäuschte automatische Qualitätsnote vergeben; diese
Antworten sind für eine externe verblindete Prüfung zu übernehmen.

Der komplette Datensatz, Referenzwerte und Modellantworten bleiben unter
`%LOCALAPPDATA%/NERAL-private/v41-blind200/<run_id>`; GitHub-Artefakte
enthalten ausschließlich Manifest, SHA-256 und aggregierte Kennzahlen.
Ein regulärer Deployment- oder Merge-Push startet **keinen** privaten
600-Antworten-Benchmark automatisch.

**Status:** Pipeline implementiert. Ein erfolgreiches Ende-zu-Ende-Ergebnis
wird erst durch einen abgeschlossenen manuellen Workflow mit
`NERAL_BLIND200_COMPLETE` und erneutem HTTP 200 auf Port 9690 belegt.
