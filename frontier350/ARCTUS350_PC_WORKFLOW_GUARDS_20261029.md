# Arctus350: PC-Jobs nur nach ausdrücklichem manuellem Start

Sechs ältere Workflows auf dem Forschungszweig konnten bei passenden
Pull-Request-Änderungen einen selbstgehosteten Windows-Runner anfordern. Einer
davon startet ein begrenztes FP32-Training; andere führen Inferenz aus oder
lesen beziehungsweise schreiben lokale Forschungsdaten.

Dieser Draft begrenzt diese sechs Fälle: Ihr selbstgehosteter Job läuft nur bei
`workflow_dispatch` mit dem explizit auf `true` gesetzten Boolean
`approve_pc`. Die Voreinstellung ist `false`. Die vier bereits vorhandenen
GitHub-gehosteten CPU-PR-Prüfungen bleiben erhalten. Für manuelle Starts
werden deren Vorbedingungen weiterhin ausgeführt. Die Arbeitsschritte,
Checkpoints und vorhandenen Ressourcenprüfungen bleiben unverändert.

Eine spätere manuelle Ausführung ist nur für den dann konkret vom Nutzer
genehmigten Job und Zeitraum vorgesehen. Das Boolean-Feld ist eine
Bedienungs- und Konfigurationshürde, kein kryptografischer Nachweis der
Einwilligung: Personen mit ausreichenden GitHub-Rechten könnten es setzen.
Auch andere, separat auslösbare oder branchgebundene Workflows existieren
weiter und sind durch diesen Draft nicht pauschal abgeschaltet.

Solange diese Änderung nur ein Draft ist, bleiben die vorhandenen Workflows
auf ihren ursprünglichen Branches unverändert. GitHub zeigt einen manuell
auslösbaren Workflow nur an, wenn dessen Definition auf dem Default-Branch
vorliegt; eine spätere Integration muss deshalb bewusst geplant werden.
Dieser Draft startet und genehmigt keinen PC-Job.

Lokale statische Regressionstests, ohne Modell- oder PC-Runner:

```bash
python -m unittest tests.test_arctus350_pc_dispatch_guards -v
```
