# NERAL 1.0 Local Preview – Release-Kandidat am 10.11.2026

**Produktziel:** Ein tatsächlich benutzbarer lokaler NERAL-Assistent mit
Chat, optionalem THINK, verifizierten Werkzeugen und expliziter
Modellherkunft. Der nächste öffentliche Termin ist **10. November 2026**.
Dies ist ein Plan und ein Startprofil, **kein bestätigter Release**.

## Was sofort startbar sein soll

1. Auf dem Windows-PC **Ollama starten**. `ollama list` muss
   `qwen3:8b` zeigen (alternativ das ausdrücklich gewählte
   `qwen3.5:9b`). Das Startprofil lädt *kein Modell* nach.
2. Vollständige Helios-Dateien inklusive `deployment/` und des
   bestehenden CPU-App-Starts bereitstellen. `Start NERAL 1.0 Local.cmd`
   doppelklicken. Bei bereits belegtem Port 9686 die alte App vorher
   schließen, damit der Launcher nicht still ein anderes Backend verwendet.
3. In Helios oben **NERAL AI** auswählen. Der Direkteinstieg öffnet den
   vorhandenen, session- und origin-geschützten lokalen KI-Chat und setzt
   den Fokus in das Nachrichtenfeld. **Der alte Hauptchat mit 50M bleibt
   bis zur eigenständigen UI-Migration separat erhalten.**
4. **FAST** arbeitet über das bestehende lokale Ollama-Backend mit
   explizitem `think=false` und 1.024 Tokens. Optionales **THINK**
   nutzt `think=true` und maximal 2.048 Tokens; die bestehende
   separate Werkzeugverifikation bleibt bestehen. Tool-JSON-Aufrufe
   bleiben auf 512 Tokens und `think=false`.
5. Modellherkunft bleibt sichtbar: Das lokale Qwen3-8B ist eine
   **Qwen-Foundation**, keine durch LoRA trainierte und geprüfte
   NERAL-Eigengewichtsversion. Es darf nicht als unabhängiges Modell
   oder höheres GPT-/Claude-Level ausgegeben werden.

Ein manuelles Startprofil kann mit
`powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\Start-Neral-1.0-Local.ps1 -CheckOnly`
*ohne App-Start* prüfen, ob die lokalen Modelldateien via `ollama list`
sichtbar sind. Für Qwen3.5-9B die Option `-Model qwen3.5:9b` angeben.
Vorhandene Legacy-App-Abhängigkeiten können durch den **bestehenden**
`Start-Neral.ps1` eingerichtet werden, wenn sie fehlen. Der neue
Startweg führt selbst keine Modell-Downloads aus.

## Aktuell gemessene Grenzen

- [Qwen3-8B gegen Qwen3.5-9B](https://github.com/Cola968/neral-helios/actions/runs/42482092860):
  Qwen3-8B erreichte 29/72, Qwen3.5-9B 8/72 bei 512 Tokens;
  31 bzw. 62 Antworten waren unvollständig. Das ist ein
  **synthetischer Entwicklungsvergleich** und stark budgetabhängig.
  Es rechtfertigt Qwen3-8B als ersten Offline-Startkandidaten,
  nicht als allgemein überlegenes Modell.
- [Qwen/v5/experimenteller Kandidat](https://github.com/Cola968/neral-helios/actions/runs/42481342450):
  4/24 gegenüber 5/24 und 8/24 in einem 24er-Pilot mit 512 Tokens.
  Transferteil nur 1/12 beim Kandidaten. Die Gewichte sind
  **nicht zur Promotion freigegeben**.
- Die Korrektur des lokalen Ollama-Backendlimits ist durch Tests
  verifiziert; die **tatsächliche Qualitätsverbesserung auf einer neuen
  Testmenge** ist weiterhin offen.

## Verbindliche Release-Entscheidung

**1.0 als lokales System** kann nur verantwortbar freigegeben werden,
wenn alle folgenden Funktionen durch echte End-to-End-Abnahmen
belegt sind:

- Windows-Start `Start NERAL 1.0 Local.cmd` auf echtem Nutzer-PC,
  OLLAMA-Runtime und geladenes Modelldigest; keine falsche
  LoRA-Anzeige oder stiller Backendwechsel.
- Mindestens einfache und mehrschrittige Nutzerfragen über **FAST**
  und **THINK** mit sinnvollen Antworten und klaren Fehlerhinweisen,
  einschließlich Tokenlimit- und Zeitüberschreitungsfällen.
- Gegenprüfung von bereits implementierten exakten Rechen-/Logikwerkzeugen
  mit nachvollziehbarem Herkunftshinweis, ohne dass ungeprüfte
  Ergebnisse als verifiziert erscheinen.
- Sicherheitsprüfung: Session/Host/Origin, kein externer Modellprovider,
  Agenten-Schreibfreigabe und isolierte Testausführung weiterhin intakt.
- Browser-Test auf Windows Desktop und schmalem Mobilformat,
  besonders die neue **NERAL AI**-Schaltfläche und die
  Modellherkunftsanzeige.
- Releaseartefakt/Installationspaket mit reproduzierbarem Stand,
  lokaler Datensicherung und geprüften Lizenzhinweisen.

**NICHT als 1.0-Gewichtsrelease behaupten:** Ein eigenständiger
Intelligenzvorsprung des v7/v8-LoRA. Der statistische Releaseprozess
in `GATE.md` bleibt dafür unverändert verbindlich; kein Adapter
wird durch das 1.0-Startprofil automatisch ausgewählt, freigegeben,
trainiert oder in einen Modellvergleich geschmuggelt.

Falls die End-to-End-Abnahmen bis zum 10. November nicht bestehen,
ist die korrekte Kennzeichnung **NERAL 1.0 Local Preview**
anstatt einer als stabil angepriesenen 1.0. Fehlgeschlagene Tests
werden nicht durch geänderte Versionslabels behoben.
