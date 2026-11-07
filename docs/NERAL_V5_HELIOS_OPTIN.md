# NERAL v5 explizit als lokalen Helios-Adapter auswählen

Der v5-Rang-16-Adapter aus Trainingslauf 42378021782 wurde auf der Windows-RTX-4070 am 07.11.2026 über denselben lokalen HF-/LoRA-Inferenzcode erfolgreich geladen und hat eine kurze Antwort erzeugt. Beweis: https://github.com/Cola968/neral-helios/actions/runs/42391428644

**Status:** Forschungsadapter, keine veröffentlichte oder allgemein überlegene Modellversion. Der neuartige 120-Aufgaben-Mathe-Test lag mit 13/120 nur gleichauf mit Qwen.

## Experimentelles v5 auswählen

In der PowerShell, **vor dem Start des Helios-Python-Prozesses**:

    $env:NERAL_AGENT_BACKEND = "hf_adapter"
    $env:NERAL_AGENT_ADAPTER_CHANNEL = "v5"

Die gespeicherten Dateien werden dann automatisch an diesen lokalen Pfaden gesucht:

- Qwen-Foundation: $HOME/.cache/neral/qwen25-smoke/c03e6d358207e414f1eca0bb1891e29f1db0e242/installed_models/hf-qwen25-coder-7b
- Gesiegelter LoRA-Adapter: $HOME/NERAL-QLoRA-Research/v5-r16-480-42378021782-1/adapter

Es gibt keinen Download, keine Cloud-Inferenz und keine stillschweigende Rückkehr zu Ollama, wenn die v5-Dateien fehlen. Der Programmstart meldet den Fehler dann ausdrücklich.

**Wichtig zum Python-Start:** `Start-Neral.ps1` ist vorwiegend für die ältere CPU-App eingerichtet. Seine eigene Python-Umgebung muss nicht dieselben CUDA-/PEFT-Pakete wie das QLoRA-Training enthalten. Starte Helios deshalb für einen echten v5-Versuch mit einer Python-Umgebung, in der Torch/CUDA, Transformers, PEFT, BitsAndBytes und Flask nachweislich installiert sind. Der erfolgreiche Windows-GPU-Test verwendete $HOME/NERAL-QLoRA-Research/training-env-py312/Scripts/python.exe. Vor dem Start über diesen Python-Interpreter prüfen, ob auch Flask verfügbar ist.

Vom vollständigen Helios-Projektverzeichnis aus und nach Prüfung der Abhängigkeiten:

    $py = Join-Path $HOME 'NERAL-QLoRA-Research\training-env-py312\Scripts\python.exe'
    & $py -c 'import flask, torch, transformers, peft, bitsandbytes; assert torch.cuda.is_available()'
    & $py app.py

Falls die Abhängigkeitsprüfung fehlschlägt, nicht behaupten, die App laufe. Bestehende Helios-Instanzen auf Port 9686 vor dem Start kontrolliert beenden. Der v5-Adapter ist im separaten lokalen Agenten-/NERAL-7B-Chat nutzbar, **nicht** automatisch als Ersatz für die ältere Standard-Chat-Engine.

## Kompatibilität und Rückweg

Ohne `NERAL_AGENT_ADAPTER_CHANNEL` bleibt das bisherige Legacy-Verhalten erhalten. Mit `legacy` wird der ältere Rang-32-Adapter gewählt. Bei `auto` + explizit `v5` wird bei fehlenden Dateien **kein** Ollama-Fallback vorgetäuscht. Manuelle Pfade über `NERAL_AGENT_BASE_PATH` und `NERAL_AGENT_ADAPTER_PATH` haben immer Vorrang. Unbekannte Kanäle werden abgewiesen.

Nach der v5-Erprobung in derselben Shell:

    Remove-Item Env:NERAL_AGENT_ADAPTER_CHANNEL -ErrorAction SilentlyContinue
    Remove-Item Env:NERAL_AGENT_BACKEND -ErrorAction SilentlyContinue

Damit verwendet ein **neu gestarteter** Helios-Prozess wieder die bisherigen Standardwerte. Die Anpassung speichert keine persönlichen Eingaben, exportiert keine Modellgewichte und führt kein Training aus.
