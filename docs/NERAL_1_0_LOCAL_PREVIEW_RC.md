# NERAL 1.0 Local Preview – Release Candidate / Windows

Stand: 8. November 2026. **Es wurde noch keine stabile 1.0
freigegeben**, und kein experimenteller NERAL-v7/v8-LoRA-Adapter wurde
für den Release freigeschaltet.

## Was dieses Paket ist

Ein eigenständiger, lokaler NERAL-KI-Chat, der das bereits installierte
**Qwen3-8B** über die loopback-gebundene Ollama-Schnittstelle nutzt.
Die alte 50M-Helios-App, deren `deployment/`-Ordner und CPU-Checkpoint
werden **nicht mehr benötigt**. Installierte Qwen-Gewichte sind
**nicht im ZIP enthalten** und werden nicht automatisch geladen.

NERAL 1.0 Local Preview stellt bereit: FAST-Chat, optionalen THINK-
Modus, eingeschränkte lokal verifizierte Mathematik-/Logik-/Python-
Ergebnisse, Herkunftskennzeichnung, CSRF-geschützten localhost-Chat
und einen leichten Windows-Starter.

Die **NERAL**-Oberfläche nutzt ein fremdes Qwen-Grundmodell. Das ist
**kein** Nachweis eines eigenständigen neuronalen Intelligenzgewinns.

## Installation auf Windows

Voraussetzungen: Windows, Python 3.11+ mit `flask` und `waitress`,
Ollama lokal gestartet und **bereits installiertes** `qwen3:8b`.
Das Modell ist auf dem bisherigen NERAL-Windows-PC bereits in
`ollama list` nachgewiesen; künftige Installationen müssen es
erneut prüfen.

1. Das geprüfte Release-Candidate-Artefakt aus GitHub Actions herunterladen.
   Die enthaltene ZIP und `SHA256SUMS.txt` gehören zusammen. SHA
   in PowerShell gegen `Get-FileHash <ZIP> -Algorithm SHA256` prüfen.
2. ZIP in einen eigenen Ordner entpacken. `local_state/` und
   persönliche Chats werden von diesem Paket nicht überschrieben
   oder mitgeliefert.
3. In PowerShell `ollama list` aufrufen. Qwen3-8B muss dort stehen.
4. `Start NERAL 1.0 Local.cmd` doppelklicken. Beim ersten
   Start fragt NERAL, ob eine isolierte Python-Umgebung unter
   `local_state/preview-venv` mit Flask und Waitress eingerichtet
   werden darf. **Nur nach Zustimmung** werden diese zwei
   Webserver-Abhängigkeiten über pip installiert. Keine Modellgewichte
   werden heruntergeladen; die QLoRA-/PyTorch-Trainingsumgebung
   und das System-Python bleiben unverändert.
5. `Start NERAL 1.0 Local.cmd` doppelklicken.
   Der Starter prüft den Ollama-Modellnamen und lokale Python-Laufzeit,
   verweigert eine zweite Instanz auf Port **9690** und startet
   den lokalen Waitress-Server. Browser-Adresse:
   **http://127.0.0.1:9690/**.
6. Zuerst eine kurze Frage im Modus FAST testen, dann optional THINK.
   Für neue Nutzer steht `-CheckOnly` im PowerShell-Starter bereit.
   Das führt keine Modellinferenz aus.

**Wichtig:** Der bestehende lokale Python-Trainings-Interpreter dient nur als Python-Ausgangsbasis. Die Webserver-Pakete werden in einer eigenen Vorschau-Umgebung installiert. Der Interpreter in
`NERAL-QLoRA-Research\\training-env-py312` wird bevorzugt, sonst
ein vorhandenes System-Python. Beide sind als Ausgangsbasis zulässig, wenn Python 3.11+ vorhanden ist.
Die Webserver-Pakete werden nur nach Einwilligung installiert; kein
Modell wird automatisch geladen oder heruntergeladen.

Alternativ vor dem Start:
`powershell -NoProfile -ExecutionPolicy Bypass -File .\\Start-Neral-1.0-Local.ps1 -CheckOnly`

Mit `-PrepareOnly` kann die isolierte Python-Umgebung nach Zustimmung
vorbereitet werden, ohne den Server oder Browser zu starten. Bei
freigegebenen CI-Abnahmen kann `NERAL_PREVIEW_INSTALL_CONSENT=1`
den interaktiven Dialog ersetzen.

## Sicherheit und Daten

Der Server ist ausschließlich an `127.0.0.1:9690` gebunden,
nicht öffentlich oder im WLAN erreichbar. HTTP ist nur auf Loopback
vorgesehen. Der Browser erhält ein zufälliges Sitzungs-Cookie und
ein per separatem Endpoint ausgegebenes CSRF-Token. Anfragen an
unerlaubte Hosts oder aus anderen Web-Ursprüngen werden blockiert.
Strikte SameSite/HttpOnly-Cookies, CSP und eine begrenzte Eingabegröße
sollen Cross-Site- und Skript-Injektionen erschweren.

Gesprächsverlauf liegt nur im Arbeitsspeicher des Browser-Tabs;
keine Cloud-Chatdatenbank. Lokale Start- und Fehlerlogs werden in
`local_state/` geschrieben, aber die Web-App speichert standardmäßig
keine Chat-Nachrichten und exponiert keine Tracebacks in der API.

Für andere Geräte, Remotezugriff, öffentlichen Internetbetrieb und
Mehrbenutzer-Sitzungen ist dieser Preview-Server **nicht freigegeben**.
Die bisherige separat abgesicherte Android-Gateway-Lösung ist davon
nicht automatisch betroffen.

## Harte Release-Kriterien

Die GitHub-Workflowdatei
`.github/workflows/neral-1-0-release-rc.yml` überprüft:

- echte Windows- und Linux-Tests der eigenständigen App;
- CSRF-/Origin-/Host-Ablehnung, Modellherkunft, FAST- und THINK-
  Routen ohne falsch freigegebene NERAL-LoRA;
- reproduzierbaren, inhaltsbegrenzten ZIP-Build inklusive Datei-SHA;
- die tatsächliche lokale Qwen3-8B-Inferenz **auf dem privaten
  Windows-Runner**: FAST-Beispielfrage, THINK mit unabhängig geprüfter
  Antwort und HTTP-Sicherheit;
- Ausgabe eines ZIP-Artefakts **nur**, wenn sowohl CPU-Tests als auch
  der echte Windows-PC-Smoke grün sind.

Der Live-PC-Test greift nicht auf private Chats oder Projektdateien zu.
GitHub erhält ausschließlich den JSON-Skalarbericht, keine Prompts
und keine generierten Texte.

## Keine voreilige Leistungsbehauptung

Der alte 72-Fälle-Vergleich mit 512 Tokens hatte für Qwen3-8B 29/72
korrekte Antworten und 31 abgeschnittene Ausgaben; für Qwen3.5-9B
8/72 korrekte und 62 abgeschnittene Ausgaben.
Diese Zahl ist stark abhängig vom Tokenbudget und darf **nicht** als
allgemeiner Modellvergleich ausgegeben werden.

Auch der experimentelle v7-Kandidat mit 8/24 in einem kleinen Pilot
hat das unabhängige 600-Fälle-Gate aus `GATE.md` nicht bestanden.
Der neue v8-Forschungsworkflow enthält keine automatisch freigegebenen
Gewichte. Daher ist es **NERAL 1.0 Local Preview**, bis die
weiteren Originalholdout-, Browser-, Lizenz- und Nutzerabnahmen
separat bestanden sind.

Eine getestete lokale Anwendung ist ein Software-Release. Ein
nachgewiesener neuronaler Intelligenzsprung ist ein anderer Maßstab.
Beide Ergebnisse werden getrennt veröffentlicht.
