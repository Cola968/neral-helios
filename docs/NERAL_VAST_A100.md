# NERAL v5 – Vast.ai A100 40 GB (experimentell)

**Wichtig:** Dies ist eine neue Linux-QloRA-Pipeline. Sie wurde noch nicht auf der gemieteten A100 ausgeführt. Die vorhandene Windows-v5-Version bleibt unverändert, der neue Rang-32-Adapter wird nicht automatisch in Helios aktiviert.

## Vor dem Mieten

- Angebot aus dem Screenshot: Vast.ai m:77967, Japan, eine A100 PCIe mit **40 GB VRAM**, angezeigte GPU-Gebühr **$0.341/Stunde**; Verfügbarkeit und Preis vor Kauf prüfen.
- Template: **PyTorch (Vast)**, SSH/Jupyter, **Linux x86_64**. Die sichtbare CUDA-12.4-Vorlage ist nur die Containerbasis. Das Setup installiert in einer eigenen Python-Umgebung **torch==2.8.0+cu128**, prüft den NVIDIA-Treiber, BF16 sowie BitsAndBytes NF4. Es scheitert kontrolliert, falls das Image nicht kompatibel ist.
- Datenträger: **mindestens 100 GB Container Size**, nicht den Standard von 16 GB. Qwen-Basis und HF-Cache benötigen beide Speicher.
- Nur 1 GPU, möglichst verifizierter Host, On-Demand und kein unterbrechbarer Spot-Host. Keine privaten Projektdateien auf der Instanz ablegen.
- GPU-Preis allein: 2 h = $0.682, 4 h = $1.364. Hinzu kommen disk/storage, Netzwerk und eventuell Steuern. Gesamtkosten vor der Anmietung im Vast-Dialog prüfen; das ist kein Preisversprechen.
- **destroy/löschen** nach dem Herunterladen der Adapter, nicht nur stop: angehaltene Instanzen können weiter Speichergebühren erzeugen.

## Nach dem Mieten – alle Befehle per SSH ausführen

Der erste Befehl klont nur das GitHub-Projekt:

    cd /workspace
    git clone --depth=1 https://github.com/Cola968/neral-helios.git
    cd /workspace/neral-helios

Das Setup klont zusätzlich den exakten alten Trainer-Commit für die Modell- und Versionsprüfung. Installation und GPU-Prüfung, **ohne Modellgewichte herunterzuladen und ohne Training**:

    bash scripts/neral_vast_setup.sh

Nur bei Erfolg fortfahren. Das Linux-Setup ist durch CPU-Tests abgesichert, noch kein GPU-Benchmark.

Das öffentliche Qwen2.5-Coder-7B-Instruct zum festen Revisionshash herunterladen, Dateien SHA-256-prüfen und die unveränderten NERAL-v5-Train/Val/Test-Splits reproduzieren:

    source /workspace/neral-cloud-env/bin/activate
    cd /workspace/neral-helios
    python -B -m scripts.neral_vast_cloud prepare

**SMOKE-Probelauf:** Nur 3 Optimizerschritte, Rang 16; zuerst NF4-Backward und Modell-Export auf der GPU verifizieren:

    python -B -m scripts.neral_vast_cloud smoke --rank 16 --steps 3 --max-length 512 --approve RUN_NERAL_RESEARCH_ONLY

Erst wenn der SMOKE-Lauf erfolgreich ist: Experimenteller QLoRA-Rang-32-Lauf mit maximal 480 Schritten und demselben Generator (1.920 Train-Records, Test-Split nie trainieren):

    python -B -m scripts.neral_vast_cloud train --rank 32 --steps 480 --max-length 512 --approve RUN_NERAL_RESEARCH_ONLY

Erwartete lokale Outputverzeichnisse:

    /workspace/neral-cloud/output/neral-v5-cloud-smoke-r16-s3/
    /workspace/neral-cloud/output/neral-v5-cloud-train-r32-s480/

**Beide Läufe sind Forschungsadapter, keine freigegebenen NERAL-Versionen.** Vor Zerstörung der Instanz die Dateien adapter_model.safetensors, adapter_config.json und neral-cloud-research.json über die private SSH-/Jupyter-Verbindung sichern. Nicht auf GitHub committen, keine Tokens oder SSH-Private-Keys im Chat veröffentlichen.

Die gewählte Rang-32-Methode ist eine **neue Implementierung**, nicht ein identischer Neustart des Windows-Trainers. Sie nutzt BF16/NF4, 5e-5 Lernrate, Microbatch 1, Gradient Accumulation 4, completion-only loss, alle sieben LoRA-Zielmodule und denselben 512-Token-Eingang wie v5. Für echten Fortschritt ist anschließend ein separater, fairer A/B-Benchmark erforderlich.

**Zeit grob einplanen:** Erstinstallation, Download und SMOKE ungefähr 30–90 Minuten; das nachfolgende Training möglicherweise 20–60 Minuten. Beides sind unsichere Planungswerte, keine gemessene A100-Leistung. Nach Trainingsende läuft die Vast-Rechnung weiter, bis die Instanz gelöscht wird.

**Nicht zusätzlich mieten**, wenn bereits eine Instanz läuft. Cloud-Instanzen und Datenübertragungen verwaltet Vast.ai, nicht GitHub Actions. Das NERAL-Skript selbst mietet keine Hardware und führt keine Zahlungen aus.
