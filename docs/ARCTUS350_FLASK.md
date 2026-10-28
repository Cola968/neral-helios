# Optionales experimentelles Arctus350 in Flask

Dieser Draft baut auf PR #173 (`11a7f9cfb98aa342d3b80921fa23afe44ff06228`) auf. Der Standard bleibt der bestehende Byte-Modellpfad. `START_NERAL.bat` → `launcher.py` → `app.py` erhält eine optionale lokale Arctus350-Inferenz. Keine Gewichte, Tokenizerdateien, Downloads, Trainingsläufe oder Veröffentlichungen werden hinzugefügt.

## Lokal aktivieren

Python 3.12 und CPU-PyTorch wie für die bestehende App installieren. Die optionale Abhängigkeit steht in `requirements-arctus350.txt`. Die normale Byte-App benötigt sie nicht.

PowerShell, in einer bereits lokal eingerichteten Umgebung:

```powershell
python -m pip install -r requirements-arctus350.txt
$env:HELIOS_DEVICE = 'cpu'
$env:HELIOS_PREPARATION_MODE = '0'
$env:HELIOS_ENABLE_ARCTUS350 = '1'
$env:HELIOS_ARCTUS350_CHECKPOINT = 'D:\lokaler-pfad\checkpoint.pt'
$env:HELIOS_ARCTUS350_TOKENIZER = 'D:\lokaler-pfad\tokenizer.json'
# Aus dem vertrauenswürdigen lokalen Laufbericht übernehmen und lokal vergleichen.
$env:HELIOS_ARCTUS350_SHA256 = '<verifizierte-64-stellige-checkpoint-pruefsumme>'
python app.py
```

Anschließend im Modellmenü **Arctus350 · Experimentell** wählen. Die Pfade sind ausschließlich Serverkonfiguration; API-Aufrufer können keine Dateipfade übergeben. `HELIOS_CHECKPOINT=arctus350` ist eine zusätzliche ausdrückliche Startauswahl, keine Voreinstellung. `HELIOS_PREPARATION_MODE=1` sperrt jeden Modellladevorgang. Die Menühilfe und kontrollierte Fehlermeldungen erklären fehlende Assets und Abhängigkeiten. `Modell entladen` entfernt die Runtime-Referenzen; Betriebssystem und PyTorch entscheiden über die tatsächliche Rückgabe reservierten Speichers.

Die Checkpoint-SHA wird vor der eingeschränkten Deserialisierung über denselben offenen Dateihandle geprüft. Nur eigene, lokal vertrauenswürdige Checkpoints verwenden: Eine passende Prüfsumme allein belegt keine Herkunft oder Modellqualität. Architektur und Tokenizerbindung müssen exakt passen; fehlende, zusätzliche oder abweichend geformte Tensoren werden abgelehnt. Keine automatische Suche in Trainingsverzeichnissen. Ein fehlgeschlagener Wechsel erhält das vorherige Modell. Für einen erfolgreichen Wechsel muss während der Prüfung Platz für bisherigen und neuen Zustand verfügbar sein.

## Native Quellen und Abgrenzung

Vier bestehende Inferenzbausteine werden gezielt aus Experiment-Commit `b1e3c6bf3b5ebe5c4c7ba0db401e8046fbffa51b` übernommen:

| Datei | Verwendung |
| --- | --- |
| `frontier350/model350.py` | unveränderte native 349.825.152-Parameter-Architektur |
| `frontier350/chat350_contract.py` | unveränderte Architektur-/Tokenizerkonstanten |
| `frontier350/chat_prompt350.py` | unveränderte segmentweise BPE-Chatkodierung |
| `frontier350/secure_checkpoint350.py` | vorhandenes `weights_only=True` mit begrenzter NumPy-Allowlist; akzeptiert zusätzlich einen offenen Dateihandle |

Der erforderliche eigene 16K-BPE-Tokenizer hat SHA-256 `4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd` und BOS/EOS/SEP/PAD 0/1/2/3. Der alte `frontier350/tokenizer350.py` ist ein anderer Byte-Tokenizer und wird ausdrücklich nicht verwendet oder übernommen. Kein Gradio-Server, optimierter Fremdpfad, Trainingscode oder vollständiger Experiment-Stack wird eingebunden.

Der Adapter verwendet den bestehenden nativen Forward-Pfad auf **CPU/FP32**, ein konservatives App-Limit von 1024 BPE-Token und höchstens 512 neue Token. Die native Architektur unterstützt 8192; die App wirbt nicht mit diesem größeren Budget. BPE-Kontextlänge wird auf dem Server tatsächlich tokenisiert, nicht aus Bytezahlen geschätzt. Die vorhandenen HTTP-Grenzen von 4096 UTF-8-Bytes bleiben zusätzlich bestehen. Kein KV-Cache; CPU-Antworten können langsam sein. Abbruch und die 120-Sekunden-Grenze werden zwischen Forward-Schritten geprüft, nicht innerhalb eines laufenden Torch-Operators. GPU-Ausführung wird in dieser Integration noch nicht angeboten.

Alle Modellladevorgänge und die gesamte Tokenisierung/Generierung/Dekodierung verwenden dieselbe Sperre. Mehrere Tabs können somit den Tokenizer nicht während einer Anfrage austauschen. Die UI wartet für Modellladen bis zu drei Minuten. Modellantworten tragen `source=native-model`, `model_family=neral-arctus350`, Checkpoint-ID und `runtime_id=neral-arctus350:<sha256>`. Werkzeugantworten behalten ihre eigenen Quellen und werden nicht als Arctus-Leistung bezeichnet.

Arctus-Antworten ohne EOS werden bei erreichtem Antwortbudget als unvollständig markiert. Nur vollständige, nicht importierte Antwortpaare mit passender Modellfamilie und exakter Checkpoint-Identität werden wiederverwendet. Die Identität überlebt Speichern, Synchronisieren und Neuladen. Ein HTTP-Client mit veralteter Runtime-ID erhält 409. Arctus-Anfragen mit fremdem oder nicht zugeordnetem Verlauf werden abgelehnt. Das sind Kontextzuordnungsregeln, keine kryptografische Beglaubigung beliebiger vom Benutzer editierter Gesprächsdaten.

## Prüfungen und Grenzen

```sh
python -m pytest -q tests/test_arctus_flask.py tests/test_helios_runtime.py tests/test_helios_chat_experience.py tests/test_helios_workspace.py tests/test_helios_i1_ready.py tests/test_helios_install.py
node --test tests/test_helios_history.js tests/test_arctus_history.js
node tests/test_helios_chat_experience.cjs
node tests/test_arctus_browser.cjs
```

Die Workflow-Datei `arctus350-flask-cpu.yml` nutzt ausschließlich einen GitHub-gehosteten Ubuntu-CPU-Runner, leere Modellverzeichnisse und isolierten Zustand. Die Python-Tests prüfen Handler, transaktionalen Wechsel, Entladen, Sperren, Streaming, Abbruch, Herkunft, Verlauf, Tokenizer-Ablehnung und den eingeschränkten Datei-Loader. Ein winziges synthetisches Modell prüft die echten nativen Layer einschließlich Meta-Laden, gebundener Gewichte und Forward auf CPU; es ersetzt keinen Test mit dem 350M-Checkpoint. Weitere Controller-/Generierungstests verwenden ausdrücklich synthetische Testmodelle. Die Browserprüfungen starten echtes `app.py`; Rechenwerkzeuge werden nicht gemockt.

Kein lokaler 350M-Checkpoint liegt in der Arbeitsumgebung vor. Ein vollständiger Ladevorgang des bewahrten Checkpoints, dessen tatsächlicher RAM-Bedarf und echte Sprachqualität sind daher unbestätigt. Kein PC-Runner wird gestartet und kein zeitgesteuerter Job eingerichtet. Die Verfügbarkeit von Hugging-Face-Jobs konnte wegen eines Connectorfehlers nicht festgestellt werden. Dieser Draft ist keine Qualitätsfreigabe für Arctus350.
