# NERAL 1.0 Local Preview – Release Candidate / Windows

Stand: 8. November 2026. **Es wurde noch keine stabile 1.0
freigegeben**, und kein experimenteller NERAL-v7/v8-LoRA-Adapter wurde
für den Release freigeschaltet.

## Schnellstart ohne GitHub Actions

1. [NERAL-Downloads auf GitHub Releases](https://github.com/Cola968/neral-helios/releases) öffnen und beim neuesten **NERAL 1.0 Local Preview** die Datei `NERAL-1.0-Local-Preview.zip` herunterladen. **Nicht** `Source code.zip` wählen. Nur durch die echten Windows-PC- und CPU-Tests geprüfte Pakete werden hier als Vorschau veröffentlicht.
2. ZIP-Datei vollständig entpacken, im Ordner `NERAL-1.0-Local` auf **Start NERAL 1.0 Local.cmd** doppelklicken. Beim ersten Start die Einrichtung der eigenen Python-Webserverumgebung mit `J` erlauben.
3. NERAL öffnet sich automatisch im Browser. Bei Bedarf ein lokales Konto erstellen; später reicht derselbe Doppelklick.

Voraussetzung: **Ollama läuft bereits, `qwen3:8b` ist installiert und Python 3.11+ ist auf dem PC vorhanden.** Der Starter lädt keine Modellgewichte automatisch und verändert weder vorhandene LoRA-Gewichte noch die Trainingsumgebung. Eine kurze Anleitung liegt auch als `START HIER.txt` direkt im ZIP.

Die Vorschau ist lokal, kostenlos und noch keine stabile öffentliche 1.0.

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

1. Bevorzugt den oben beschriebenen direkten Download unter **GitHub Releases** verwenden. Für Entwickler gibt es unter GitHub Actions weiterhin das Artefakt des aktuellen Main-Commits.
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
   `local_state/preview-venv` mit Flask, Waitress und pypdf eingerichtet
   werden darf. **Nur nach Zustimmung** werden diese Python-Abhängigkeiten
   über pip installiert. Keine Modellgewichte
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
offline** und kein rein lokales STT. Beim Sprechen werden vorläufige
Wörter angezeigt, aber niemals automatisch gesendet. Nur das endgültige
Erkennungsergebnis kann bei aktivierter Option `Sprache direkt senden` an
den lokal laufenden NERAL-Chat übergeben werden. Wenn im Eingabefeld bereits
eigener Text steht, wird das Diktat angehängt und **nicht** ungefragt
abgeschickt. `Mikrofon` erneut anklicken oder `Verlauf löschen` bricht die
Erkennung ab und verwirft auch verspätet eintreffende Erkennungsereignisse.
Die Option `Vorlesen` gibt Antworten über die Browser-Sprachsynthese in
Abschnitten wieder (maximal die ersten 8000 Zeichen). `Tempo` bietet
0,85-fach, normal oder 1,15-fach. Eine bevorzugte deutschsprachige
Systemstimme wird verwendet, falls verfügbar; Stimme und Qualität bleiben
vom Browser abhängig. `Stopp Audio` beendet auch die weitere Segmentfolge.
Ein automatisches, dauerhaft eingeschaltetes Mikrofon gibt es absichtlich
nicht. Nicht unterstützte Browserfunktionen werden deaktiviert.

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
Die Windows-PC-Abnahme prüft nun zusätzlich, ob das lokale Textmodell
wirklich ein nur in einer angehängten Textdatei enthaltenes Prüfwort
wiedergibt. Ist bereits ein unterstütztes Vision-Modell installiert,
prüft sie auch einen im Arbeitsspeicher erzeugten gültigen PNG-Anhang
mit der realen lokalen Bild- und Textpipeline. Ohne Vision-Modell wird
**kein Modell heruntergeladen**; der skalare Prüfbericht markiert die
Bildanalyse ausdrücklich als nicht live verifiziert. Ein bestandenes
CPU-Gate allein beweist keine Bildanalyse auf dem Nutzer-PC.

PDF-Auswertung setzt pypdf aus requirements-neral-preview.txt voraus.
Bei einer bereits vorhandenen isolierten local_state/preview-venv
prüft der Windows-Starter die Abhängigkeiten erneut. Fehlt pypdf,
fragt er vor einer Installation in **dieser** Umgebung ausdrücklich
nach Zustimmung. Die Trainingsumgebung und System-Python bleiben
unverändert; es gibt keinen automatischen Modell-Download.

Unter jeder Assistentenantwort stehen **Gefällt mir** und **Kopieren**.
Likes sind tablokal und werden nicht auf dem Server gespeichert;
sie verändern die Modellgewichte nicht. Kopieren verwendet die lokale
Browser-Zwischenablage.

Multipart-Anfragen durchlaufen dieselben Host-, Origin-, CSRF- und
Sitzungsprüfungen wie der bisherige Chat. Dateien werden nicht auf
dem Server gespeichert. Die neue Funktion ist bis zur CI- und
Nutzerabnahme **noch kein stabiler NERAL-1.0-Release**.

## Lokale Benutzerkonten und Chat-Speicherung

Die Anmeldung ist **lokal auf diesem PC**, nicht bei einem Internetdienst. Ohne Anmeldung ist der Gastchat weiterhin verfügbar; dessen Verlauf wird nicht dauerhaft gespeichert. Mit einem lokalen Konto bleiben Chats nach dem Schließen des Browsers erhalten. Gespeicherte Gespräche lassen sich in der Sidebar öffnen, suchen und löschen.

Der Benutzername umfasst 3–32 erlaubte Zeichen, das Passwort 12–128 Zeichen. Passwörter werden ausschließlich als scrypt-Hashes gespeichert. Die SQLite-Datenbank liegt außerhalb des austauschbaren Installationsordners unter Windows in `%LOCALAPPDATA%\NERAL\accounts.sqlite3`, unter Linux im XDG-Datenordner beziehungsweise `~/.local/share/NERAL/accounts.sqlite3`. Beim Installieren einer neuen ZIP-Version bleiben die Konten deshalb erhalten.

**Wichtige Einschränkung:** Die SQLite-Datei ist nicht verschlüsselt. Ein lokales NERAL-Passwort schützt die Anmeldung in der App, nicht vor jemandem mit Dateisystemzugriff auf das Betriebssystemkonto. Die Anwendung bleibt ausschließlich an `127.0.0.1:9690` gebunden; diese Anmeldung ist nicht für einen öffentlich erreichbaren Mehrbenutzer-Server ausgelegt.

Im Menü **Einstellungen** lässt sich ein JSON-Export der eigenen Gespräche herunterladen, das Passwort ändern oder das lokale Konto mitsamt Chats endgültig löschen. Passwortänderungen verlangen das alte Passwort und beenden auch andere Sitzungen. Kontolöschung erfordert das Passwort und löscht alle zugehörigen gespeicherten Chats. Es gibt keine E-Mail-Wiederherstellung, keinen Cloud-Abgleich und keinen automatischen Datenimport. Für einen eigenen Sicherungsstand den Export verwenden.

Beim Konto- oder Chatwechsel entfernt die Oberfläche nicht gesendete Texte, ausgewählte Dateianhänge und alte Nachrichten aus der Ansicht. Verzögerte Ergebnisse einer vorherigen Anmeldung werden verworfen. Schnell aufeinanderfolgende Antworten werden beim lokalen Speichern nacheinander verarbeitet.

Stripe ist für diese lokale Preview **nicht aktiviert**: Es gibt keine Abos, keine Bezahlpflicht und keine echten Zahlungen. Die optionalen Websuch- und Browser-Sprachdienste behalten ihre gesonderten Datenschutzhinweise. Die lokale Anmeldung behauptet keine Freigabe von NERAL-LoRA-Modellgewichten.

Zur Prüfung der neuen Konten dienen `python -m unittest discover -s tests -p test_neral_1_0_accounts.py -v`, `node --test tests/test_neral_1_0_account_client.cjs` und die unveränderten vollständigen Release-Tests.
