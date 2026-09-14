# Claude Opus 5 in Helios

Die Integration ist vorbereitet, ohne Neral zu ersetzen. Neral bleibt lokal; Claude Opus 5 ist eine zusätzliche, bewusst auswählbare Cloud-Option im Modellmenü.

## Einmal einrichten

1. In Experiential Labs unter **API Keys** einen Key erstellen. Der Key beginnt mit `xpl_`.
2. Im Neral-Ordner **Configure Opus.cmd** doppelklicken und den Key eingeben. Der Key wird mit Windows DPAPI für das aktuelle Windows-Benutzerkonto verschlüsselt unter `local_state/explabs-key.dpapi` gespeichert. `local_state/` ist bereits von Git ausgeschlossen.
3. Neral vollständig neu starten und anschließend **Start Neral + Opus.cmd** verwenden.
4. Unter dem Nachrichtenfeld die Modellauswahl öffnen und **Claude Opus 5 · Experiential Labs** auswählen.

Alternativ kann `EXPLABS_API_KEY` vor dem Start als Umgebungsvariable gesetzt werden. Dann ist `Configure Opus.cmd` nicht nötig.

## Architektur

`Browser -> lokales Helios-Backend -> Experiential Labs -> claude-opus-5`

Der Browser erhält den Experiential-Key nie. Die vorhandenen Localhost-, Session- und Rate-Limit-Guards bleiben aktiv. SOLVE und Codeprüfung bleiben lokale Werkzeuge. Der bestehende Neral-Checkpoint bleibt unverändert und weiterhin auswählbar.

Die Claude-5-Routen können Samplingparameter fest vorgeben. Die Integration sendet deshalb bewusst kein `temperature` oder `top_p`, sondern nur die Felder, die benötigt werden (`model`, `messages`, `max_tokens`).

## Was diese Stufe noch nicht macht

Claude Opus 5 kann in Helios chatten, UI-/Codevorschläge erzeugen und lange Implementierungen schreiben. Es erhält in dieser Stufe **keinen direkten Schreibzugriff auf Repository-Dateien**. Ein autonomer UI-Umbau aus dem Helios-Chat heraus braucht zusätzlich explizite, bestätigungspflichtige Datei-Tools; das sollte getrennt gebaut werden, statt dem Cloud-Modell pauschalen Dateizugriff zu geben.
