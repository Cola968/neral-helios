# Helios Workspace 2

Diese Version erweitert die bestehende Flask-/JavaScript-App. Sie verändert keine Trainingsskripte, Trainingsdaten oder Originalgewichte. Die Runtime erzwingt CPU. Es gibt keine Fremdmodelle und keinen Modell-Fallback.

## Funktionen

- Desktop und Mobile behalten Sidebar, aufgeräumten Composer und Modellwahl. Einstellungen ergänzen Dark/Light, Schriftgröße, Streaming und Synchronisierung.
- Modellantworten verwenden NDJSON-Streaming mit Abbruch, begrenzter Warteschlange, 120-Sekunden-Rechenlimit und Filterpuffer. Kleine Antworten können wegen des Puffers erst am Ende erscheinen. SOLVE und Codeprüfung bleiben ausdrücklich separate Werkzeuge.
- Chats und Entwürfe werden mit dem PC synchronisiert. Gleichzeitige Bearbeitung erzeugt eine Konfliktkopie. Löschungen werden bei aktivierter Synchronisierung übertragen. Temporäre Chats werden nicht gespeichert oder synchronisiert.
- Projekte liegen zentral auf dem PC. Geöffnete Projekte prüfen alle 30 Sekunden auf neue Revisionen, solange kein Projektfeld fokussiert ist. CASCADE bleibt bestätigungspflichtig; RIPPLE und LIVING arbeiten mit den hinterlegten lokalen Abhängigkeiten.
- Der Service Worker hält die Oberfläche offline verfügbar. Gespeicherte Chats und neue Entwürfe bleiben lokal nutzbar. Antworten und Projektzugriffe benötigen den PC; es gibt keine Offline-Inferenz auf Android und noch keine Offline-Warteschlange für Projektänderungen.
- Schnellzugriff: Strg+Umschalt+Leertaste innerhalb der App sowie PWA-Verknüpfungen `/?quick=chat` und `/?quick=solve`. Dies bereitet den Einstieg vor; globales Hotword, Hintergrundmikrofon und ein Android-Systemassistent sind nicht implementiert.
- Der Modellkatalog unterstützt ausdrücklich importierte eigene Neral-Versionen innerhalb der vorhandenen Architekturverträge. Unvollständige Exporte erscheinen nicht in der Auswahl.
- Sitzungsprüfung für alle API-Routen, separate Limits für Generierung und Steuerung, Host-/Origin-Prüfung, begrenzte Eingaben und generische Fehler mit Diagnose-ID. Das TLS-Gateway reicht Streams weiter und hält seinen Backend-Schlüssel vom Browser fern.
- Unter Windows schützt DPAPI neu geschriebene Chat- und Projektinhalte auf dem Server. Bestehende JSON-Datensätze bleiben lesbar und werden bei erneutem Schreiben geschützt. DPAPI-Daten sind an den Windows-Benutzer gebunden. Auf anderen Betriebssystemen bleibt der Schutz Aufgabe der Dateiberechtigungen. Browserkopien bleiben unverschlüsselt; Exportdateien ebenfalls.
- Abgebrochene Teilantworten bleiben mit Kennzeichnung erhalten. Einstellungen bieten Neuverbindung und Diagnoseexport ohne Chattexte, Dateipfade oder Zugangscodes.

## Start und Paket

ZIP vollständig entpacken, dann `Start Neral.cmd` öffnen. Workspace 2 nutzt `http://127.0.0.1:9686/`, damit eine ältere App auf 9684 weiterlaufen kann. Der Starter übernimmt die inzwischen auf GitHub ergänzte Python-Einrichtung, prüft die Installersignatur und löscht keine defekte bestehende Python-Umgebung. Eine frische Einrichtung benötigt Internet; danach erfolgt die Inferenz lokal.

Anschließend `Start Android.cmd`: gleicher WLAN-Zugang, HTTPS und Kopplung wie in `HELIOS_INSTALL.md`. Android erhält eine installierbare PWA, keine APK. Ein bereits laufendes altes Gateway auf 9443 muss vor Start des neuen Gateways beendet werden. Keine Firewall- oder Routerregeln werden automatisch geändert.

Alte Browseradressen haben einen eigenen Speicherbereich: vorhandene Chats dort exportieren und in Workspace 2 importieren. Für eine bestehende Installation die Dateien aktualisieren, den Ordner `local_state` behalten und die App neu starten. DPAPI-Datenbanken nicht auf einen anderen Windows-Benutzer übertragen; dafür den App-Export verwenden. Alle gekoppelten Geräte gehören demselben Besitzer; keine Mehrbenutzerverwaltung.

## Nach dem Training: ausdrücklicher Import

Der Import beobachtet keine Trainingsverzeichnisse und startet keinen Job. Erst nach abgeschlossenem Training den freigegebenen eigenen Checkpoint und seine unabhängig geprüfte SHA-256 verwenden:

```powershell
python import_neral.py --source "C:\Pfad\zum\freigegebenen.pt" --sha256 "<64-stellige-SHA256>" --id neral-i1-final --label "Neral 50M · I1 Final" --assets deployment
```

Der Import prüft Quelle und Prüfsumme, lädt nur auf CPU mit `weights_only=True`, prüft Architektur, exakte Parameterzahl, endliche und gebundene Gewichte sowie striktes State-Dict. Er schreibt ausschließlich eine neue unveränderliche App-Kopie unter `deployment/installed_models/` und ein Importprotokoll. Ein Lock verhindert parallele Katalogänderungen; nach einem Crash muss ein verbliebenes `import.lock` geprüft werden. Ein unvollständiger Import wird nicht aktiviert. Die Quelle bleibt unverändert.

Danach App neu verbinden und die neue Version ausdrücklich auswählen. Erst nach Trainingsfreigabe einen kurzen CPU-Chat-/Streaming-Test durchführen und den separaten Frozen-Benchmark auswerten. Der Import ist **keine Qualitäts-Promotion**; SOLVE und Helios-Ergebnisse zählen nicht als Modellleistung. Der bisherige Checkpoint bleibt verfügbar. Diese App-Arbeit enthält keine neue Trainings- oder Benchmarkmessung.

## Reproduzierbare App-Prüfungen

```text
python -m pytest -q tests/test_helios_runtime.py tests/test_projects.py tests/test_helios_i1_ready.py tests/test_helios_install.py tests/test_helios_workspace.py
node --test tests/test_helios_history.js
node tests/test_helios_browser.cjs
node tests/test_helios_workspace.cjs
node tests/test_helios_pwa.cjs
```

Die Browser-Fixtures erzeugen keine Modellinferenz. Der PWA-Test benötigt eine laufende App; `HELIOS_TEST_URL` setzt ihre Adresse. `HELIOS_PREPARATION_MODE=1` sperrt Laden und Inferenz auch beim Live-Test. Die CPU-CI umfasst API/Security, History sowie beide Fixture-Browsersuiten. Eine Installation auf einem physischen Android-Gerät bleibt separat zu prüfen.

Lokal gemessen am 10.09.2026: 82 Python-Tests bestanden, 8 Node-History-Tests bestanden, Desktop/Mobile-Fixtures und Zwei-Geräte-/Streaming-Fixtures bestanden. Live-PWA auf Port 9688 im Vorbereitungsmodus: Manifest, Service Worker und Offline-Entwurf bestanden, null Modellanfragen. PowerShell-Parser und JavaScript-Syntaxprüfungen bestanden. Keine neue Modellinferenz, kein Benchmark und kein GPU-Aufruf in diesen Prüfungen.

Windows-Starter-Korrektur: registrierte Python-Installationen und frühere Neral-Ordner werden erkannt. Ein erfolgreicher Installer im Wartungsmodus löst eine erneute Suche aus; Zielpfade mit Leerzeichen werden korrekt übergeben. Offline-Regression mit tatsächlichem Python-Versionsaufruf und simuliertem Installer-Code 0 bestanden. Der entpackte Starter wurde mit vorhandener CPU-App-Umgebung gestartet: API meldet eigenen I1-Checkpoint geladen, CPU, Vorbereitungsmodus aus; keine Testgenerierung ausgeführt.
