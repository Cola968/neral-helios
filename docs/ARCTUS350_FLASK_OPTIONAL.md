# Optionale Arctus350-Anbindung: vorbereitet, reale Abnahme offen

Stand: 28.10.2026. Basis ist PR #173, Head
`11a7f9cfb98aa342d3b80921fa23afe44ff06228`. Kein Merge, kein Modelltraining,
keine Änderungen an vorhandenen Gewichten, keine Modell-/Tokenizerdownloads.

**Die echte Checkpoint-Kompatibilität ist noch nicht belegt.** Im bereitgestellten
Workspace fehlen Checkpoint und Tokenizer-Datei. Der erhaltene Wert
`4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd`
ist der bereits im Quellvertrag festgelegte **Tokenizer-SHA-256**, nicht ein
Checkpoint-Pfad oder Checkpoint-Hash. Es wurde deshalb kein 350M-Checkpoint geladen.

## Vertrag und Implementierung

Der optionale Adapter `arctus350_runtime.py` verwendet den unveränderten
Referenz-Modellcode aus PR #140, Commit `105502d9be58517f87d8cfff5d9a5b0e286c22b3`.
Die kopierten Inferenzquellen und ihre Herkunft stehen in
[`arctus350_core/ORIGIN.md`](../arctus350_core/ORIGIN.md).
Kein Experiment- oder Trainingszweig wurde zusammengeführt.

- Modell: `frontier350-v1`, 349.825.152 Parameter, 24 Layer, Breite 1152,
  18 Q-/2 KV-Heads, FFN 3136, gebundene Embedding-/Ausgabematrix.
- Tokenizer: eigener 16.384er BPE mit dem oben genannten SHA; BOS=0, EOS=1,
  SEP=2, PAD=3. Der ByteTokenizer der bisherigen App ist **inkompatibel**.
- Checkpoint: ausdrücklich konfigurierter SHA-256; eingeschränktes
  `torch.load(weights_only=True)` mit enger NumPy-RNG-Allowlist. Keine unsichere
  Pickle-Ausweichroute. Konfiguration, Tokenizer-Metadatum, Schlüssel, Shapes,
  endliche Fließkommawerte und gebundene Gewichte werden geprüft.
- Nur einheitliche Tensor-Dtypes werden akzeptiert. Fused/umbenannte Schlüssel,
  Byte-Checkpoints, Quantisierungen und gemischte Gewichtstypen fallen durch.
  Der Adapter ändert weder Gewichtsdateien noch gespeicherte Tensorwerte und
  erzeugt keinen Optimizer. Ein Meta-Modell vermeidet die zufällige Vollallokation
  eines zweiten 350M-Modells vor der Shape-Prüfung.
- Prompts werden pro Rollen-Segment wie im Arctus-SFT-Vertrag tokenisiert.
  BOS/PAD/SEP sind bei der Ausgabe maskiert; EOS beendet die Generierung.
- Aktuell ausschließlich CPU; Temperatur/Top-k entsprechen der Flask-API
  (Top-k weiter maximal 260). Keine Behauptung numerischer oder qualitativer
  Parität mit der separaten GPU-/Top-p-/Repetition-Penalty-CLI.

## Ausschließlich lokal konfigurieren und ausdrücklich laden

Abhängigkeiten sind eine bewusste, optionale Installation, keine Startautomatik:

```powershell
python -m pip install -r requirements-arctus350.txt
# Zusätzlich das für die App vorgesehene CPU-PyTorch 2.8 installieren, falls nötig.
$env:HELIOS_ARCTUS350_CHECKPOINT = 'C:\lokal\vorhandener-checkpoint.pt'
$env:HELIOS_ARCTUS350_SHA256 = '<64-stelliger SHA-256 DIESES Checkpoints>'
$env:HELIOS_ARCTUS350_TOKENIZER = 'C:\lokal\vorhandener-tokenizer.json'
$env:HELIOS_PREPARATION_MODE = '0'
# Optional: vermeidet auch den bisherigen automatischen Byte-Modellstart.
$env:HELIOS_CHECKPOINT = 'arctus350'
python app.py
```

Die drei Asset-Einstellungen werden nur vom lokalen Prozess gelesen. HTTP-Clients
können keine Dateipfade wählen. Es gibt keine Suche in Trainingsverzeichnissen,
keine Hub-Aufrufe und keine Download-Funktion. Ohne vollständige Konfiguration
wird Arctus350 nicht angeboten. Konfiguration allein lädt Arctus350 **nie**.
Auch `HELIOS_CHECKPOINT=arctus350` löst kein Laden beim Start aus.

In der vorhandenen Modellauswahl **Arctus350 → ausdrücklich laden** wählen.
Dies sendet den authentifizierten `POST /api/model` mit
`{"checkpoint":"arctus350"}`. Der Vorbereitungsmodus sperrt weiterhin jeden
Modellwechsel. Fehler lassen die vorherige Runtime erhalten. Nach erfolgreichem
Laden melden `/api/info` und Antworten `model_family: arctus350`, den tatsächlichen
Checkpoint-SHA und den BPE-Tokenizer. `native_arctus350_integrated: true` benennt
die vorhandene Adapter-Unterstützung, **nicht** einen geladenen Checkpoint;
maßgeblich dafür ist `checkpoint_loaded`.

Die Oberfläche kennzeichnet Arctus350 als eigenes experimentelles Modell.
Rechenwerkzeuge und Sicherheitsregeln bleiben separat. Herkunft und
Checkpoint-SHA werden gespeichert; Modellkontext verwendet nur vollständige,
nicht importierte eigene Nachrichten mit passender Modell-ID **und** passendem
Checkpoint-Hash. Ältere Nachrichten ohne Hash bleiben sichtbar, gehen bei
bekanntem aktuellem Hash konservativ nicht in den Modellkontext ein.

## Echte Abnahme mit bereitgestellten Dateien

Der vorbereitete Prüf-Befehl nutzt den echten Flask-Testclient und lädt erst über
`POST /api/model`. Er schreibt nur temporären App-Zustand; Gewichte und Tokenizer
werden ausschließlich gelesen. JSON-Ergebnis auf stdout, Exit 0 nur bei
vollständigem Erfolg, Exit 1 bei Fehler, Exit 2 bei unvollständiger Abbruchmessung:

```powershell
python tools/probe_arctus350_app.py `
  --checkpoint 'C:\lokal\vorhandener-checkpoint.pt' `
  --sha256 '<CHECKPOINT-SHA256>' `
  --tokenizer 'C:\lokal\vorhandener-tokenizer.json' `
  --max-new 8
```

Der Probe prüft explizites Laden, Stream-Start/-Abschluss und Herkunft, einen
Stopp nach Beginn eines echten Forward-Passes und die Freigabe der Sperren.
Ein bewusst injizierter Forward-Fehler prüft die bereinigte Stream-Fehlermeldung,
ohne Gewichte zu beschädigen. Anschließend werden beide Datei-Hashes erneut
geprüft. Die Modelltexte werden im Ergebnis nicht ausgegeben. Falls die Antwort
bereits vor dem Stopp fertig war, wird der Abbruch ausdrücklich als **nicht
gemessen** ausgewiesen. Das ist keine positive Abnahme.

## Durchgeführte Prüfungen und verbleibende Grenzen

Die lokalen Tests verwenden einen winzigen synthetischen Neral350M und einen
synthetischen Tokenizer. Sie belegen Rollen-Segmentierung, Spezialtoken-Maske,
unveränderte synthetische Gewichte, Abbruch zwischen Forward-Pässen, Fehlerrouten,
Provenienz und Sperrfreigabe im echten Flask-Handler. Erfolgreiches Installieren
in `/api/model` wird mit einer ersetzten Datei-Ladegrenze geprüft. Das sind
**keine** erfolgreichen Lade- oder Inferenztests des tatsächlichen 350M-Assets.

Offen bis zur Bereitstellung: Dateien samt Checkpoint-SHA, reale sichere
Deserialisierung/Shapes/Dtype, vollständige Modell-/Tokenizer-Kompatibilität,
CPU-RAM und Laufzeit, echtes Streaming und Abbruch mit diesem Asset sowie dessen
Modellqualität. CPU lädt derzeit den gesamten Checkpoint einschließlich etwaiger
Trainings-Metadaten in den RAM. Ladezeit und RAM sind hier nicht gemessen.

Abbruch und das 120-Sekunden-Generierungsbudget werden zwischen Forward-Pässen
geprüft; ein laufender PyTorch-Aufruf wird nicht hart unterbrochen. Das Budget ist
keine harte Deadline und gilt nicht für das Laden. Es gibt keinen KV-Cache.
Die Oberfläche schätzt das Tokenbudget mit UTF-8-Bytes konservativ; der Server
prüft die tatsächlichen BPE-IDs. Das 4096-Byte-Eingabelimit bleibt aktiv.
Keine GPU-/Windows-/physische Mobilgeräte-Abnahme und keine Qualitätsfreigabe.

Reviewbefunde, Vergleich zum unveränderten Head und Testprotokolle:
[`evidence/pr173_independent_review/README.md`](../evidence/pr173_independent_review/README.md).
