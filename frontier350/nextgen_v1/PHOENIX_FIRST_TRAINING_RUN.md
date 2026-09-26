# Phoenix: erster echter 350M-Trainingslauf auf deinem Windows-PC

**Status: Anleitung, Startcheck und Live-Fortschrittsleser implementiert; kein tatsächlicher Lauf gestartet.** GitHub Actions, dieser Chat und synthetische Tests haben keinen Zugriff auf deine privaten Gewichte und Trainingsdaten oder deine RTX 4070. Es sind weiterhin **0 neue 350M-Trainingstokens** nachgewiesen. Dem historischen Checkpoint fehlen AdamW-Momente: `phoenix_weight_only_pilot.py` beginnt daher mit einem **neuen** Optimizer und neuen Zählern, nicht mit einer exakten historischen Fortsetzung.

## Voraussetzungen, die nicht automatisch als erfüllt gelten

Lies `WEIGHT_ONLY_PILOT_OPERATOR_CONTRACT.md`. Vor dem Lauf müssen Trainings- und Validierungsdaten auf Herkunft, Lizenz, Datenschutz, Qualität und semantische Überschneidungen geprüft sein. Ein unabhängiger, eingefrorener deutsch/englischer Holdout muss tatsächlich vorliegen und überprüft werden. Die Freigabedatei darf erst danach die realen positiven menschlichen Entscheidungen enthalten und muss das geplante Setup exakt binden. Keine Freigabe vortäuschen und keine privaten Dateien auf GitHub übertragen. Hash-Prüfungen ersetzen keine menschliche Begutachtung.

Die folgenden Pfade sind **Platzhalter**, keine bekannten Pfade deines Rechners. Starte PowerShell im geprüften Checkout des Entwicklungszweigs `experiment/phoenix-blind-ab-audit-20260919`, ohne ungesicherte Änderungen zu verwerfen. Nutze nur vorhandene lokale Dateien. Der Ausgabeordner muss neu sein; keine Originaldateien überschreiben. Das Pilotprogramm verlangt über 200 GiB freien Speicher plus 12 GiB Reserve sowie eine freie geeignete RTX 4070. Die Konfiguration (128 × 1 × 16 = 2048 Tokens je Optimizer-Schritt, Ziel 262.144 Tokens) muss **genau so vorab freigegeben** sein und ist keine belegte ideale Einstellung.

```powershell
$assets = 'D:/DEIN_ECHTER_ASSETS_ROOT'
$source = 'D:/DEIN_ORIGINAL_CHECKPOINT/resume.pt'
$train = 'D:/DEINE_GEPRUEFTEN_TRAIN_SHARDS'
$valid = 'D:/DEINE_GETRENNTEN_VALID_SHARDS'
$out = 'D:/DEIN_NEUER_NICHT_EXISTIERENDER_PILOT_ORDNER'
$approval = 'D:/DEINE_ECHTE_FREIGABE.json'
$review = 'D:/DEIN_TATSAECHLICHER_QUELLENREVIEW.json'
$holdout = 'D:/DEIN_ECHTER_FROZEN_HOLDOUT.json'
$approvalSha = (Get-FileHash -Algorithm SHA256 $approval).Hash.ToLowerInvariant()
$pilotArgs = @('--execute','--trusted-local-checkpoint',
  '--source-checkpoint',$source,'--assets-root',$assets,
  '--train-data',$train,'--valid-data',$valid,'--out',$out,
  '--approval',$approval,'--approval-sha256',$approvalSha,
  '--source-review',$review,'--frozen-holdout',$holdout,
  '--target-new-tokens','262144','--seq','128','--microbatch','1',
  '--grad-accum','16','--lr','0.00002','--compile-mode','none',
  '--save-every','10','--eval-every','10')
python -B frontier350/nextgen_v1/phoenix_preflight_only.py @pilotArgs
```

`CPU_PREFLIGHT_FAIL` bedeutet, dass zunächst der gemeldete Blocker zu klären ist; fehlende oder ungeprüfte Unterlagen sind **kein** Grund für erfundene positive Review-Flags. `CPU_PREFLIGHT_PASS_GPU_NOT_CHECKED` bedeutet nur: Die nativen CPU-, Datensatz- und Freigabe-Konsistenzprüfungen sowie die exakte Tokenüberschneidungsprüfung waren erfolgreich. Menschliche Freigabe ist dadurch nicht unabhängig bewiesen und GPU-Speicher/Temperatur sind nicht geprüft. Der reale Trainer wiederholt die Gates.

**Nur nach wahrheitsgemäßer gültiger Freigabe und bestandenem Preflight, explizit auf deinem eigenen Rechner:**

```powershell
python -B frontier350/nextgen_v1/phoenix_weight_only_pilot.py @pilotArgs
```

Dies ist der echte Trainingsaufruf, **keine** bereits gestartete Session. Ein Fehler, fehlender Speicher oder zu hohe Temperatur beweisen keinen Erfolg. Der neue Checkpoint wird nur in `$out/resume.pt` gespeichert. Historische und neue Tokens niemals addieren. Die derzeit nicht implementierte, separat zu genehmigende Stufenerweiterung nicht durch Manipulation der Freigabe umgehen.

## Echte Fortschrittswerte beobachten

Nach tatsächlichem Trainingsstart in einem **zweiten** PowerShell-Fenster, im selben Checkout mit demselben `$out`-Wert:

```powershell
python -B frontier350/nextgen_v1/phoenix_training_progress.py --run-dir "$out" --follow
```

Der read-only Monitor liest nur lokale `metrics.jsonl` und optional `summary.json`. Er zeigt bestätigbare **Log-Angaben** zu `new_tokens`, Ziel, Fortschrittsprozent, Optimizer-Schritten, letztem Trainings-/Validierungs-Loss und aus neuen Tokens geteilt durch verstrichene Laufzeit errechnetem **Wall-Clock-Durchsatz einschließlich Validierung/Speicher-Overhead**. Er meldet fehlerhafte oder widersprüchliche Logs, hält private Pfade aus der Ausgabe heraus und behauptet weder GPU-Kernel-Durchsatz noch Antwortqualität oder verifizierte Checkpoint-Bytes. Ohne Log meldet er keinen Trainingserfolg. `Ctrl+C` beendet die Anzeige.

Einmalige Momentaufnahme statt fortlaufender Anzeige:

```powershell
python -B frontier350/nextgen_v1/phoenix_training_progress.py --run-dir "$out"
```

Alternativ bleiben die unveränderten Rohlogs auf dem PC lesbar: `Get-Content "$out/metrics.jsonl" -Tail 10 -Wait`. **Ein erzeugter Log-Eintrag allein beweist keinen tatsächlich intakten Checkpoint**; dafür sind der Original-Hash, die vollständige Resume-Datei und die unabhängige A/B-Auswertung gesondert zu prüfen. Keine 60–70.000 Tokens/s ohne reale Messung behaupten. Erst unabhängig bewertete Antworten auf demselben eingefrorenen Holdout können Chatqualität belegen.
