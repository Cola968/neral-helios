# NERAL – lokaler Sprachmodus (CPU, Deutschland)

Der neue Button **Lokaler Sprachmodus** ist ausdrücklich getrennt vom
vorhandenen Browser-Mikrofon. Beim lokalen Modus werden Audiodaten ausschließlich
an `127.0.0.1:9690` übertragen. Der Server benutzt lokal gespeicherte
**Faster-Whisper Base**-Gewichte zur deutschen Transkription und
**Piper de_DE-thorsten-medium** zur deutschen WAV-Sprachausgabe. Er lädt im
laufenden Chat **keine Modelle automatisch** und sendet keine Audioaufnahme an
einen externen KI- oder Sprachanbieter.

## Einfacher Start auf Windows

1. Starte NERAL einmal über **Start NERAL 1.0 Local.cmd**. Dadurch wird die
   isolierte Umgebung `local_state/preview-venv` angelegt.
2. Doppelklicke **Install NERAL Voice.cmd**. Bestätige die einmalige
   Installation von Python-Paketen und den Download der zwei offenen Modelle.
   Es entstehen Datenübertragungen an **PyPI** und **Hugging Face**, ausschließlich
   während der einmaligen Installation nach deiner Zustimmung.
3. Starte NERAL neu. Der Button **Lokaler Sprachmodus** prüft die lokale
   Verfügbarkeit und öffnet das Mikrofon erst nach dem Klick.

Die Modelle liegen ausschließlich in `local_state/voice/`, außerhalb der
Release-ZIP-Datei und der Git-Historie. Die normalen Qwen-/LoRA-Gewichte und
die Trainingsumgebung werden nicht verändert. Die Whisper-STT- und Piper-TTS-
Inferenz läuft jeweils auf der CPU, um die RTX 4070 für NERAL zu reservieren.

## Ablauf und Grenzen

Nach Klick auf den lokalen Sprachmodus hört der Browser lokal auf Sprache
und erkennt nach etwa **0,8 Sekunden Stille** das Ende der Äußerung. Die
Aufnahme wird als mono 16-kHz PCM-WAV in einem **CSRF-geschützten** Aufruf an
den Loopback-Server übergeben (0,25–12 s). Der lokale STT-Text gelangt
über die vorhandene NERAL-Chatoberfläche mit dem ausgewählten LOW/HIGH-Modus
zum Ollama-Modell. Dessen fertige Antwort wird anschließend von Piper in
kurzen Blöcken in WAV gesprochen. Während des Vorlesens ist die Aufnahme
stummgeschaltet; danach kannst du weiterreden, bis du den lokalen Modus
explizit ausschaltest. Der Knopf **Stopp Audio** und **Verlauf löschen**
beenden die aktuelle Sprachsitzung und sperren verspätete Antworten.

Der Modus ist **halbduplex und sprechpausenbasiert**, nicht das vollständig
simultane, latenzoptimierte Streaming-System von ChatGPT Voice. Er hat
zurzeit keine gesprochenen Zwischenantworten während Qwen rechnet und
keine zuverlässige Unterbrechung laufender CPU- oder GPU-Inferenz.
Die Antwortlatenz hängt von Rechner, Whisper, Piper und Qwen ab und ist
**nicht** auf dem persönlichen Windows-PC gemessen. Das Browser-Mikrofon
mit externer Browser-Spracherkennung ist weiterhin getrennt erhältlich;
es wird beim lokalen Modus deaktiviert, nicht als Fallback eingeschaltet.

## Paket und Privatsphäre

Das ZIP enthält nur lokale Quellprogramme und den Ein-Klick-Installer,
**keine** automatisch heruntergeladenen Modelle, keine Aufnahmen, keine
Chatlogs und keine Trainingsgewichte. Das lokale Backend verarbeitet PCM
direkt im RAM. Die Erkennung wird fail-closed abgelehnt, wenn Pakete oder
Modelldateien fehlen. Keine Remote-Spracherkennung, keine Cloud-TTS, keine
automatische Freigabe von Mikrofondaten.

Quellen für die separat installierbaren Sprachmodelle:
- https://huggingface.co/Systran/faster-whisper-base
- https://huggingface.co/rhasspy/piper-voices/tree/main/de/de_DE/thorsten/medium
- https://github.com/SYSTRAN/faster-whisper
- https://github.com/rhasspy/piper

Die Software-Regressionstests können die API, Audioformate, Abbrüche und
Browser-Zustände nachstellen, ersetzen aber keinen realen
Mikrofon-, TTS- und GPU-Leistungstest am Nutzer-PC.
