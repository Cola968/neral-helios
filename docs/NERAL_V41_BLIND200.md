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
