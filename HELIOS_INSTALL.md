# Neral CPU-App: Windows und Android

## Jetzt schreiben

Am PC: http://127.0.0.1:9684/ öffnen. Neral 50M D3/I1 ist als eigener Checkpoint auf der CPU aktiv. Die GPU wird nicht verwendet. Im Menü oben rechts „App installieren“ wählen. Alternativ das ZIP entpacken und `Start Neral.cmd` ausführen: öffnet ein eigenes App-Fenster. Python 3.11+ ist erforderlich; bei einer frischen Installation installiert der Starter ausschließlich CPU-PyTorch und die App-Abhängigkeiten. Der lokale Modell-Export ist im ZIP enthalten, keine fremden Gewichte.

## Android

Die installierbare Web-App erhält ein eigenes Symbol/Fenster. Es ist keine APK und keine Offline-KI auf dem Handy: Die Berechnung läuft auf deinem eingeschalteten PC im gleichen WLAN.

1. Zuerst `Start Neral.cmd`, dann `Start Android.cmd` auf dem PC ausführen.
2. Anleitung und einmaligen Kopplungscode in `local_state/phone/HANDY-ZUGANG.txt` lesen. Für diese vorbereitete Installation liegt die Datei im Arbeitsordner der App.
3. Nur das dort erzeugte öffentliche `Neral-Android-CA.crt` auf das eigene Handy übertragen. In Android unter Sicherheit → weitere Sicherheitseinstellungen → Verschlüsselung/Anmeldedaten → Zertifikat installieren → CA-Zertifikat installieren. Bezeichnungen variieren nach Hersteller. Die private Datei `server.key` niemals übertragen.
4. Die HTTPS-Adresse aus der Anleitung in Chrome öffnen und koppeln. Danach Chrome-Menü → „App installieren“. Ein Code gilt für dieses lokale Gateway; Sitzungen laufen nach 8 Stunden ab und enden bei einem Gateway-Neustart.
5. Falls Windows die Verbindung blockiert, die konkrete Python-App/Port 9443 nur im privaten Heimnetz zulassen. Keine Router-Portfreigabe. Firewallregeln wurden von uns nicht verändert. Bei einer Änderung der PC-IP muss der lokale Handy-Zugang neu erzeugt und das neue Zertifikat eingerichtet werden.

Warum HTTPS: installierbare PWAs benötigen auf entfernten Geräten eine sichere Verbindung. Siehe [Chromium-Installationskriterien](https://web.dev/articles/install-criteria) und [Android-Zertifikate](https://support.google.com/pixelphone/answer/2844832?hl=de). Zertifikat später in Android entfernen, wenn der Zugang nicht mehr verwendet wird. Die CA-Signaturschlüssel werden bei Einrichtung nicht gespeichert.

## Auswahl des eigenen Modells

D3/I1 hat im vorhandenen vollständigen Versionsvergleich NIS 65,2458 gegenüber 44,0625 für 20M Preview02. H1 erreichte 65,1433 und löste das Regressions-Gate aus; M2.1 blieb gleich und erfüllte das Verbesserungs-Gate nicht. Neuere Mobile-Kandidaten verlangen laut ihren Berichten noch Frozen-Evaluation. Deshalb wird für diese App D3/I1 gewählt, ohne neue Promotion oder neue Benchmarkbehauptung.

Dies ist eine experimentelle Vorschau, kein allgemein verlässlicher Chatbot. Die vorhandenen D3/I1-Werte für freie Ausgabe waren EOS 11 %, Wiederholungen 7,41 % und RawMath 8 %. Die Testanfrage „Hallo, wer bist du?“ lieferte auf CPU 48 Byte-Tokens in 3,0956 s (15,51 Tokens/s), aber eine inhaltlich unpassende Antwort. Das sind echte technische Messwerte, kein Nachweis guter Chat-Qualität. SOLVE bleibt ein getrenntes Werkzeug.

Quellcheckpoint SHA-256: `ab48ea9266444c19c67325a9f344813f5b69692cf65749d6abf5b5a15ff222da`. App-Export enthält nur Gewichte, Konfiguration und Herkunftsmetadaten. Originalcheckpoint, Trainingsdateien und Daten wurden nicht verändert. Aktuelle Export-Prüfsumme steht in `deployment/model-selection.json`.

## Datenschutz und Betrieb

Chats bleiben im Browserspeicher des jeweiligen Geräts. PC- und Handy-Chats werden noch nicht automatisch synchronisiert; Import/Export ist vorhanden. Die neue Adresse hat einen eigenen Speicherbereich; alte Chats bei Bedarf im alten Fenster exportieren und hier importieren. Projekte liegen auf dem PC. Offline können gespeicherte Chats gelesen werden; Antworten und Projekte benötigen den laufenden PC. Der Service Worker speichert ausschließlich die App-Oberfläche, keine API-Antworten oder Kopplungscodes.

Der normale Server bindet nur 127.0.0.1. Android nutzt einen separaten TLS-Gateway mit Host/Origin-Prüfung, begrenzten Kopplungsversuchen und Secure/HttpOnly/SameSite-Cookie. CPU-App-Port ist 9684, TLS-Port 9443. `HELIOS_PREPARATION_MODE=1` sperrt Modellladen und Inferenz weiterhin optional.

## Ausgeführte Prüfungen

63 Python-Tests (CPU, Runtime, Projekte, Security, PWA-Routen und Gateway); 8 Node-Tests; Browser-Regression für Desktop/Handybreiten und Light/Dark; JavaScript-Syntax. TLS-Zertifikat auf dem PC mit eigener CA validiert, ungepaarte API mit HTTP 401 abgewiesen und authentifizierter HTTPS-Zugriff auf den eigenen 50M-CPU-Server erfolgreich geprüft. Windows-Starter hat die paketierte App erfolgreich geöffnet. Android-Installation auf einem physischen Handy und dessen Firewall/WLAN-Erreichbarkeit können erst auf deinem Gerät bestätigt werden.
