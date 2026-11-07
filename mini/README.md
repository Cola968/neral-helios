# NERAL Mini – lokale Windows-Vorschau

**Ein Doppelklick, eine eigene Chatoberfläche, lokales LoRA.**
NERAL Mini ist bewusst getrennt vom älteren Helios-50M-Chat.
Es verwendet die lokal gespeicherten Qwen2.5-Coder-7B-Gewichte als
Foundation und lädt einen **eigenen experimentellen NERAL-LoRA-Adapter**.

## Start auf Windows

1. Auf GitHub **Actions → NERAL Mini Windows preview** öffnen und
   im erfolgreichen Lauf das Artefakt **NERAL-Mini-Windows** herunterladen.
   Alternativ die Dateien im Repository herunterladen.
2. ZIP vollständig in einen Ordner entpacken.
3. **Start NERAL Mini.cmd** doppelt anklicken.
4. NERAL öffnet **http://127.0.0.1:9691/**. Nachricht eingeben.
   Beim ersten Antworten lädt der lokale Adapter.
5. Zum Beenden Strg+C im geöffneten Terminal drücken.

**Voraussetzung:** Die vorhandene NERAL-QLoRA-GPU-Python-Umgebung
mit CUDA-PyTorch, transformers, peft, accelerate und bitsandbytes.
Die RTX-GPU benötigt mindestens 10 GiB VRAM. Der Starter verwendet
standardmäßig folgende bereits bekannte lokale Umgebung:

    %USERPROFILE%\NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe

Standardpfade für die vorhandenen NERAL-Gewichte:

    %USERPROFILE%\.cache\neral\qwen25-smoke\c03e6d358207e414f1eca0bb1891e29f1db0e242\installed_models\hf-qwen25-coder-7b
    %USERPROFILE%\NERAL-QLoRA-Research\actions-r32-42078722575-1\adapter

Der Standardadapter ist der **experimentelle Rank-32-Pilot**,
kein unabhängig validiertes Produktmodell.

## Andere lokale NERAL-Adapter verwenden

Wenn ein neuer trainierter und versiegelter Adapter vorliegt, kann
der tatsächliche lokale Ordner über eine Umgebungsvariable gesetzt
werden. Beispiel in PowerShell:

    $env:NERAL_MINI_ADAPTER_PATH = 'D:\NERAL\mein-neuer-adapter'
    .\Start-NERAL-Mini.ps1

Andere Python-Installation:

    $env:NERAL_MINI_PYTHON = 'D:\mein-env\Scripts\python.exe'

NERAL prüft die lokalen Qwen- und NERAL-Prüfsummen, bevor Gewichte
in den GPU-Speicher geladen werden. Der Adapter muss die Dateien
adapter_config.json, adapter_model.safetensors und neral-adapter.json
enthalten. Unpassende oder beschädigte Modelle werden nicht geladen.

## Datenschutz und Grenzen

- Nur auf 127.0.0.1:9691 erreichbar, nicht aus dem WLAN.
- Keine externe KI-API, keine Downloads von Gewichten oder Trainingsdaten.
- Kein Websearch, Computer Use, Shellzugriff oder Codeausführung.
- Verlauf nur im Speicher dieses Browserfensters.
- Maximal zwölf Verlaufseinträge, 224 neue Token pro Antwort.
- Nur eine laufende GPU-Berechnung, noch kein Token-Streaming
  oder verlässlicher serverseitiger Abbruch.
- Qwen bleibt die Foundation. Der eigene NERAL-LoRA-Adapter ist ein
  zusätzlich trainierter Gewichtsanteil, kein eigenständig vortrainiertes 7B-Modell.

Diese Mini-Version behauptet kein GPT-Niveau oder einen
nachgewiesenen allgemeinen Qualitätsgewinn. Das braucht Benchmarks.

Bei zu wenig GPU-Speicher laufende Trainingsjobs und andere Modelle
zuerst beenden. NERAL Mini installiert keine Pakete heimlich.

## Entwicklungstests

    python -m pytest -q tests/test_neral_mini.py

Die CPU-Tests benutzen einen simulierten Adapter. Die tatsächliche
GPU-Inferenz muss auf dem Windows-PC mit vorhandenen Modellgewichten
zusätzlich bestätigt werden.
