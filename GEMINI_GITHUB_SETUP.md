# Gemini: sichere Entwicklung für NERAL/HELIOS über GitHub

Diese Integration funktioniert **ohne lokale Installation**. Sie benutzt Googles
`google-github-actions/run-gemini-cli` und kann ausschließlich manuell über den
GitHub-Actions-Tab gestartet werden.

## Einrichtung

1. Einen Google-AI-Studio-API-Schlüssel erstellen und **nur** in GitHub unter
   `Settings → Secrets and variables → Actions → New repository secret` mit
   Namen `GEMINI_API_KEY` hinterlegen. Niemals den Schlüssel in Issues,
   Chat-Nachrichten oder Dateien speichern.
2. Die Workflow-Datei `.github/workflows/gemini-neral-manual.yml` nach
   Prüfung in den Default-Branch übernehmen.
3. Im Repository `Actions → Gemini NERAL development (manual) → Run workflow`
   auswählen. Das Feld `task` beschreibt die gewünschte begrenzte Aufgabe.
4. Nach dem Lauf unter `Actions → Run → Artifacts` die Datei
   `gemini-neral-proposed-changes` ansehen. Prüfe `candidate.patch` und
   die Testergebnisse in der Job-Zusammenfassung.

## Sicherheitsgrenzen

- Die Action hat **nur lesenden** Repository-Zugriff (`contents: read`).
- Der Checkout besitzt keine dauerhaft gespeicherten GitHub-Push-Credentials.
- Kein Scheduler, keine automatischen Merges, keine Trainingsläufe.
- Gemini kann einen lokalen Patch vorbereiten, aber nicht direkt pushen.
- Der GitHub-Runner ist vom privaten Windows-PC getrennt. Es gibt keine
  lokalen Arctus-Modellgewichte, und es wird keine GPU bereitgestellt.
- Die Action führt die Gemini CLI im automatischen Modus aus. Sie ist darum
  trotz der Prompt-Vorgaben **nicht** als Sandbox gegen bösartige Befehle
  oder kompromittierte Abhängigkeiten zu betrachten. Vor Benutzung GitHub-
  und Google-Nutzungslimits sowie API-Kosten prüfen.
- Eine erfolgreiche Ausführung oder Kompatibilität mit dem gewählten
  Google-API-Konto ist noch nicht bestätigt.

## Später mögliche Erweiterungen

Erst nach erfolgreichem Test und gesonderter Freigabe: automatisierte
Pull-Request-Vorschläge, fest definierte regelmäßige Prüfungen sowie ein
Budget-/Quota-Limit. Diese Erweiterungen gehören nicht zur Erstinstallation.
