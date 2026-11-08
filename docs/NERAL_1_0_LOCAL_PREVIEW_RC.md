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

1. Unter GitHub Actions das Artefakt des **aktuellen** Main-Commits herunterladen.
   `NERAL-1.0-Local-Preview-CPU-TESTED-NOT-PC-VERIFIED` ist nur für
   **manuelle Tests**, auch wenn der GPU-Runner noch beschäftigt ist.
   Erst `NERAL-1.0-Local-Preview-RELEASE-CANDIDATE` setzt zusätzlich
   einen bestandenen realen Windows-Ollama-Test voraus. Keines davon
   bedeutet, dass NERAL-Gewichte überlegen sind. Im CPU-Paket gibt
   `SOURCE-COMMIT.txt` den exakten Git-Commit an. Die enthaltene ZIP
   und `SHA256SUMS.txt` gehören zusammen. SHA in PowerShell gegen
   `Get-FileHash <ZIP> -Algorithm SHA256` prüfen.
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
5. Nach dem Start öffnet sich NERAL. Der Starter prüft den Ollama-Modellnamen und lokale Python-Laufzeit,
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

## Erweiterung 1.0: Sprachchat und Plugins
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


## Code Check und ausdruecklich gewaehlte Dateien

- Python Code Check analysiert den gerade eingegebenen Python-Quelltext durch `ast.parse` und eine nicht ausgefuehrte Python-Kompilierung. Kein Benutzer-Code wird ausgefuehrt, kein Shellprozess
  gestartet, keine Laufzeit-, Semantik- oder Sicherheitstests behauptet.
- Datei-Kontext liest im Browser nach ausdruecklicher Dateiauswahl bis zu zwei
  Textdateien (je maximal 1200 Byte, zugelassene Quelltext-Endungen). Nur beim
  naechsten aktiven Senden wird ihr Inhalt mit in den lokalen Modellprompt
  aufgenommen. Dateiinhalte werden danach aus dem Browser-Zwischenspeicher
  geloescht. Kein genereller Verzeichniszugriff oder Zugriff auf vertrauliche
  Projektdateien ohne Nutzerauswahl.
- `project_files` (automatischer Dateisystem-Agent) und
  `local_code_execution` (Codeausfuehrung) bleiben deaktiviert.
  Der begrenzte Datei-Kontext und der Python-Syntaxcheck sind getrennte Tools.


## Dateianhänge, Bildanalyse und Antwort-Feedback (Feature-PR)

Die Browseroberfläche erlaubt nach ausdrücklicher Dateiauswahl höchstens
**drei Dateien à 4 MiB** pro Anfrage. Sie werden ausschließlich an
127.0.0.1:9690 übertragen, dort im Arbeitsspeicher verarbeitet und
nicht dauerhaft gespeichert.

- **PNG, JPEG, WebP:** Bildbeschreibung mit bereits installiertem
  lokalem Ollama-Vision-Modell: bevorzugt qwen3-vl:2b (ca. 1,9 GB
  quantisierte Gewichte), alternativ qwen2.5vl:3b.
- **PDF:** Text von bis zu 15 Seiten mittels pypdf; Bildscans ohne
  Textschicht werden abgelehnt (noch kein OCR).
- **DOCX:** Text aus der begrenzten Office-XML-Datei.
- **TXT, MD, CSV, JSON und ausgewählte Code-Dateien:** UTF-8-Text als
  gekürzter, schreibgeschützter Chatkontext. Code wird nicht ausgeführt.

Qwen3:8B ist das **Textmodell**. Für Bilder erstellt das Vision-Modell
zuerst eine Beschreibung; das Textmodell erhält sie als **ungeprüfte
Bildinformation**. Eine fehlerfreie Erkennung ist nicht garantiert.
Kürzungen von Anhängen werden im Ergebnis gekennzeichnet.

Es gibt **keine automatischen Modell-Downloads**. Nach eigener
Entscheidung kann der Nutzer ein passendes lokales Modell mit dem
Kommando ollama pull qwen3-vl:2b installieren. Ohne bereits installiertes
Vision-Modell meldet der Dienst einen Fehler, statt ein Bild zu erfinden.
Der echte Windows-GPU-Live-Test für Bildanalyse steht noch aus.

PDF-Auswertung setzt pypdf aus requirements-neral-preview.txt voraus.
Bei einer bereits vorhandenen isolierten local_state/preview-venv kann
es fehlen. NERAL meldet das ausdrücklich. Nachinstallation nur mit
Nutzerentscheidung in dieser isolierten Umgebung, niemals automatisch
in der QLoRA-Trainingsumgebung.

Unter jeder Assistentenantwort stehen **Gefällt mir** und **Kopieren**.
Likes sind tablokal und werden nicht auf dem Server gespeichert;
sie verändern die Modellgewichte nicht. Kopieren verwendet die lokale
Browser-Zwischenablage.

Multipart-Anfragen durchlaufen dieselben Host-, Origin-, CSRF- und
Sitzungsprüfungen wie der bisherige Chat. Dateien werden nicht auf
dem Server gespeichert. Die neue Funktion ist bis zur CI- und
Nutzerabnahme **noch kein stabiler NERAL-1.0-Release**.
