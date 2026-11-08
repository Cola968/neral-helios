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
4. Im Paketverzeichnis `python -m pip install -r requirements-neral-preview.txt`
   **nur bei fehlenden Python-Abhängigkeiten selbst** ausführen.
   Dieser Schritt benötigt Internet zum Installieren von Flask/Waitress;
   die Modellinferenz selbst arbeitet ohne Cloud-KI.
5. `Start NERAL 1.0 Local.cmd` doppelklicken.
   Der Starter prüft den Ollama-Modellnamen und lokale Python-Laufzeit,
   verweigert eine zweite Instanz auf Port **9690** und startet
   den lokalen Waitress-Server. Browser-Adresse:
   **http://127.0.0.1:9690/**.
6. Zuerst eine kurze Frage im Modus FAST testen, dann optional THINK.
   Für neue Nutzer steht `-CheckOnly` im PowerShell-Starter bereit.
   Das führt keine Modellinferenz aus.

**Wichtig:** Der bestehende lokale Python-Trainings-Interpreter in
`NERAL-QLoRA-Research\\training-env-py312` wird bevorzugt, sonst
ein vorhandenes System-Python. Beide sind zulässig, wenn sie Flask und
Waitress enthalten. Es werden keine Python- oder Modellinstallationen
ungefragt durchgeführt.

Alternativ vor dem Start:
`powershell -NoProfile -ExecutionPolicy Bypass -File .\\Start-Neral-1.0-Local.ps1 -CheckOnly`

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


## Erweiterung 1.0: Sprachchat und Plugins (Feature-PR #393)

**Sprachchat:** In der lokalen Browseroberfläche gibt es die Schaltfläche
`Mikrofon`. Die Aufnahme startet nur nach einem Klick und Browser-Freigabe.
Die Browser-Spracherkennung überträgt je nach Browser und Systemeinstellung
Ton an einen **externen Browseranbieter**. Das ist **nicht garantiert
offline** und kein rein lokales STT. Die erkannte Nachricht kann nach Wahl
direkt an den lokal laufenden NERAL-Chat gesendet oder vorher geprüft werden.
Die Option `Vorlesen` gibt Antworten über die Sprachsynthese des Browsers
wieder. `Stopp Audio` beendet diese Wiedergabe. Nicht unterstützte
Browserfunktionen werden deaktiviert, nicht als verfügbar vorgetäuscht.

**Websuche:** Standardmäßig aus. Erst nach expliziter Aktivierung pro
Nachricht wird **nur der Suchbegriff** an DuckDuckGo HTML gesendet,
gegebenenfalls an Bing RSS. Kein API-Schlüssel, keine Abo-Beschränkung.
Die maximal drei Texttreffer samt HTTPS-Quellen werden als ungeprüft
gekennzeichnet und dem lokalen Modell als Kontext gegeben; die einzelnen
Webseiten werden nicht geladen. Das ist keine garantierte Live-Suche und
keine Verifikation von Fakten; Suchanbieter können blockieren.

**Plugin-Status:** `/api/neral/preview/plugins` unterscheidet
verfügbare lokale Mathe-/THINK-Funktionen, konfigurierte aber nicht live
verifizierte Websuche und browserabhängige Sprachfunktionen von **nicht
implementierten** Projektdatei- und Codeausführungs-Plugins. Zusätzliche
API-Route `/api/web/search/status` mit ehrlichem Anbieterstatus.
CSRF, Host- und Origin-Prüfungen gelten auch für die Suche.

**Keine Modellgewichts-Promotion:** Die zusätzlichen Fähigkeiten können
dem Gesamtsystem bei konkreten Aufgaben helfen, bedeuten aber **nicht**,
dass die Qwen-Gewichte verbessert wurden. Eine allgemeine Überlegenheit
über Qwen erfordert einen neuen, unbekannten, reproduzierbaren A/B-Test.
