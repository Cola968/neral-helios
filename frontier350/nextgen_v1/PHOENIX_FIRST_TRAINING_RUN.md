# Phoenix: erster echter 350M-Trainingslauf auf deinem Windows-PC

**Status: Anleitung und Startcheck implementiert, kein tatsächlicher Lauf gestartet.** GitHub Actions, dieser Chat und die synthetischen Tests haben keinen Zugriff auf deine privaten Gewichte und Trainingsdaten oder deine RTX 4070. Es sind weiterhin **0 neue 350M-Trainingstokens** nachgewiesen. Der historische Checkpoint hat fehlende AdamW-Momente: `phoenix_weight_only_pilot.py` beginnt deshalb einen **neuen** Optimizer und neue Zähler, keine exakte historische Fortsetzung.

## Voraussetzungen, die nicht automatisch als erfüllt gelten

Lies `WEIGHT_ONLY_PILOT_OPERATOR_CONTRACT.md`. Vor einem realen Lauf müssen die Trainings- und Validierungsdaten hinsichtlich Herkunft, Lizenz, Datenschutz, Qualität und semantischer Überschneidungen geprüft sein. Ein unabhängiger, eingefrorener deutscher/englischer Holdout muss tatsächlich existieren und überprüft werden. Die Freigabedatei darf erst danach die tatsächlichen positiven menschlichen Entscheidungen enthalten und muss **exakt** das vollständige geplante Setup binden. Keine Freigabe vortäuschen, keine privaten Dateien auf GitHub übertragen. Der native Trainer kontrolliert Hashes und exakte Tokenüberschneidungen, kann aber Lizenzrecht, Datenschutz oder semantische Unabhängigkeit nicht selbst nachweisen.

Die folgenden Pfade sind **Platzhalter**, nicht die behaupteten tatsächlichen Pfade deines PCs. Starte PowerShell im überprüften Checkout des Entwicklungszweigs `experiment/phoenix-blind-ab-audit-20260919`, ohne ungesicherte Änderungen zu verwerfen. Trage nur echte lokale Pfade ein. Der Ausgabeordner muss neu sein, die vorhandenen Dateien dürfen nicht überschrieben werden. Das zugrunde liegende Pilotprogramm verlangt mehr als 200 GiB freien Platz zuzüglich 12 GiB Reserve sowie eine freie, geeignete RTX 4070. Die gewählte Konfiguration (128 x 1 x 16 = 2048 Tokens je Optimizer-Schritt, Ziel 262.144 Tokens) **muss vorab genau so** in der Freigabe enthalten sein und ist keine nachgewiesene ideale Einstellung.

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

`CPU_PREFLIGHT_FAIL` mit `failure_hint` bedeutet: erst diesen konkreten Blocker beheben. Fehlende oder ungeprüfte Unterlagen sind **kein** Grund, positive Review-Flags automatisch zu setzen. `CPU_PREFLIGHT_PASS_GPU_NOT_CHECKED` bedeutet nur, dass die echten nativen CPU-/Datensatz-/Freigabekonsistenzprüfungen und exakte Tokenprüfungen erfolgreich waren: menschliche Prüfung ist dadurch nicht unabhängig belegt und GPU-Speicher/Temperatur sind nicht getestet. Der eigentliche Trainer wiederholt alle nativen Prüfungen und verlangt anschließend die echten GPU-Schutzprüfungen.

**Nur nach einer tatsächlichen gültigen Freigabe und bestandenem Preflight, explizit auf deinem eigenen Rechner:**

```powershell
python -B frontier350/nextgen_v1/phoenix_weight_only_pilot.py @pilotArgs
```

Das ist der **reale Trainingsaufruf**, kein Beispiel für eine angeblich schon laufende Session. Ein Abbruch mit Fehler, fehlendem Speicher oder Temperaturüberschreitung ist ein fehlgeschlagener Versuch, kein Trainingserfolg. Der erste echte Checkpoint wird nur in `$out/resume.pt` gespeichert. Historische und neu verarbeitete Tokens nicht addieren. Die bislang nicht implementierte, separat zu genehmigende Erweiterung über den fixierten Stopp hinaus **nicht** durch Manipulation der Freigabe umgehen.

## Echten Fortschritt beobachten

In einem zweiten PowerShell-Fenster, nachdem das Training tatsächlich begonnen hat:

```powershell
Get-Content "$out/metrics.jsonl" -Tail 10 -Wait
```

Die Events `step` und `validation` zeigen unter anderem `new_tokens`, Loss und bereits vergangene Laufzeit; der Pilot behauptet keine gemessenen 60–70.000 Tokens/s. Erst mit echten Vorher-/Nachher-Antworten auf demselben unabhängig eingefrorenen Holdout ist eine Verbesserung der Chatqualität belegbar. Die vorhandene lokale Chatoberfläche kann den historischen Stand sofort testen; ein neues Modell darf nur nach wirklicher Prüfung als verbessert bezeichnet werden.
