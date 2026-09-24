# Neral CPU-App: Windows und Android

## Jetzt schreiben

Am PC: http://127.0.0.1:9686/ öffnen. Neral 50M D3/I1 bleibt als eigener Legacy-Checkpoint auf der CPU aktiv. Ein ausdrücklich importiertes Arctus/Phoenix-350M-Bundle läuft dagegen in einem getrennten lokalen Frontier-Prozess und kann eine vorhandene CUDA/BF16-PyTorch-Umgebung verwenden. Im Menü oben rechts „App installieren“ wählen. Alternativ das ZIP entpacken und `Start Neral.cmd` ausführen: öffnet ein eigenes App-Fenster. Python 3.11+ ist erforderlich; bei einer frischen Installation installiert der Starter CPU-PyTorch und die Kern-App-Abhängigkeiten und versucht zusätzlich die optionale `tokenizers`-Runtime zu installieren. Scheitert nur diese optionale Installation, bleibt Legacy-Neral nutzbar; Frontier 350 benötigt `tokenizers`. Der Starter installiert nicht automatisch ein großes CUDA-PyTorch-Paket. Der vorhandene Legacy-Modell-Export ist im ZIP enthalten, keine fremden Gewichte.


## Frontier 350 auf der RTX-GPU

Ein Frontier-Modell wird erst nach einem ausdrücklichen, SHA-geprüften Import in der Modellwahl angezeigt. Beim ersten Auswählen startet Helios den zugehörigen lokalen Sidecar automatisch. Der Sidecar bindet nur an `127.0.0.1`; Checkpoint-SHA und Tokenizer-SHA müssen vor der Aktivierung exakt zum registrierten Bundle passen.

Der Starter sucht zuerst nach einer bereits vorhandenen Python-Umgebung, in der `torch.cuda.is_available()`, BF16 und `tokenizers` funktionieren. Wird eine gefunden, setzt er sie als `HELIOS_FRONTIER_PYTHON`. Für eine andere vorhandene CUDA-Umgebung kann die Variable vor dem Start ausdrücklich gesetzt werden. `HELIOS_FRONTIER_DEVICE=cpu|cuda|auto` steuert die Gerätewahl; Standard ist `auto`. Die 20M/50M-Haupt-App bleibt davon unabhängig CPU-only.

Der aktuelle Frontier-Decoder verwendet einen vorallokierten KV-Cache, streamt Antworten, unterstützt Stop/Cancel, stoppt erkennbare Wiederholungsschleifen und meldet Tokens/s sowie VRAM. Der Standard-Import begrenzt den nutzbaren Kontext weiterhin konservativ auf 512 BPE-Tokens, obwohl die Architektur einen größeren Maximalwert besitzt; ein höherer Deployment-Kontext muss ausdrücklich beim Import gesetzt und separat qualifiziert werden.

Für den bekannten historischen 2.073B-Checkpoint bleibt der Quellcheckpoint unverändert. Der Importer akzeptiert einen Legacy-Pickle-Fallback nur beim fest eingebauten Continuity-SHA und schreibt anschließend einen neuen bereinigten `frontier350-inference-v1`-Checkpoint ohne Optimizer/RNG/Trainingsobjekte. Beispiel:

```powershell
python import_frontier350.py --source "D:\Pfad\zum\2.073B\resume.pt" --sha256 "f7a73e283421975acc34e13afc38378be4eab11da34e8e6b46fc7cb0cda883e7" --tokenizer "D:\Pfad\zu\frontier350_assets\tokenizer.json" --tokenizer-sha256 "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd" --id neral-arctus-350 --label "Arctus 350M · 2.073B" --assets deployment
```

Nach erfolgreicher Auswahl merkt Helios nur die Modell-ID und stellt dieselbe explizite Auswahl beim nächsten Start wieder her. Es findet dabei keine automatische Qualitäts-Promotion statt.

## Arctus technisch qualifizieren

Der einfachste Weg ist jetzt ein Doppelklick auf:

`Setup Arctus 350.cmd`

Falls das bereinigte Bundle noch fehlt, öffnet der Launcher zwei Dateidialoge: zuerst für den bekannten 2.073B-Arctus-Checkpoint, danach für die zugehörige `tokenizer.json`. Die fest eingebauten SHA-256-Werte, Architektur und Gewichte werden geprüft; anschließend wird der minimale `frontier350-inference-v1`-Export erzeugt und direkt die RTX-Qualifikation gestartet. Existiert `deployment/installed_models/neral-arctus-350` bereits, wird der Import übersprungen und direkt qualifiziert.

Für einen bereits eingerichteten Bundle kann alternativ nur `Qualify Arctus.cmd` gestartet werden.

Der Launcher sucht automatisch nach `HELIOS_FRONTIER_PYTHON`, `NERAL_FRONTIER_PYTHON`, einer Projekt-`.venv`, der lokalen Helios-Python-Umgebung und danach nach `python.exe`. Verwendet wird nur eine Umgebung, in der CUDA, BF16 und `tokenizers` tatsächlich funktionieren.

Der Qualifikationslauf startet einen isolierten Frontier-Sidecar ausdrücklich mit `--device cuda`, führt einen Warm-up, drei Streaming-Smokes und einen echten Cancel-Test aus und beendet den Sidecar danach wieder. Gemessen werden First-Token-Latenz, Decode-Tokens/s, Peak-/Allocated-VRAM, CUDA/GPU-Informationen, Streaming, KV-Cache und Cancel. Antworten werden im JSON-Bericht nicht im Klartext gespeichert, sondern nur über Länge und SHA-256 referenziert.

Der Bericht liegt unter `local_state/qualifications/frontier350-qualification-*.json`; Runtime-Logs liegen daneben. Ein erfolgreicher Lauf ist **nur eine technische Runtime-Qualifikation**, keine Qualitäts-Promotion und kein Trainings-Gate. Der Quellcheckpoint und das importierte Bundle werden nicht verändert.

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

Chats und Entwürfe synchronisieren sich mit dem PC; gleichzeitige Änderungen bleiben als Konfliktkopien erhalten. Temporäre Chats sind ausgenommen. Einstellungen erlauben das Abschalten der Synchronisierung. Projekte liegen zentral auf dem PC. Offline bleiben Oberfläche, lokale Chats und Entwürfe verfügbar; Antworten und Projekte benötigen den PC. Der Service Worker speichert keine API-Antworten oder Kopplungscodes. Neue Windows-Datenbankinhalte sind mit DPAPI geschützt, Browserkopien bleiben unverschlüsselt. Details und Post-Training-Import: HELIOS_V2.md.

Der normale Server bindet nur 127.0.0.1. Android nutzt einen separaten TLS-Gateway mit Host/Origin-Prüfung, begrenzten Kopplungsversuchen und Secure/HttpOnly/SameSite-Cookie. CPU-App-Port ist 9686, TLS-Port 9443. `HELIOS_PREPARATION_MODE=1` sperrt Modellladen und Inferenz weiterhin optional.

## Ausgeführte Prüfungen

63 Python-Tests (CPU, Runtime, Projekte, Security, PWA-Routen und Gateway); 8 Node-Tests; Browser-Regression für Desktop/Handybreiten und Light/Dark; JavaScript-Syntax. TLS-Zertifikat auf dem PC mit eigener CA validiert, ungepaarte API mit HTTP 401 abgewiesen und authentifizierter HTTPS-Zugriff auf den eigenen 50M-CPU-Server erfolgreich geprüft. Windows-Starter hat die paketierte App erfolgreich geöffnet. Android-Installation auf einem physischen Handy und dessen Firewall/WLAN-Erreichbarkeit können erst auf deinem Gerät bestätigt werden.
