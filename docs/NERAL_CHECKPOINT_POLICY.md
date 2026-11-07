# NERAL: bestes Validierungscheckpoint statt blind die letzten Gewichte

**Problem:** v6 und v6.1 liefen jeweils `120` Optimizer-Schritte mit
`eval_strategy="no"`, `save_strategy="no"`; die im Trainer
gehaltenen 100/140 Validierungsbeispiele wurden **erst nach
dem gesamten Lauf** gemessen. Ein mögliches besseres Zwischenmodell
konnte dadurch nicht automatisch gewählt werden.

## Implementierte optionale Forschungsfunktion

`scripts/neral_training_checkpoint_policy.py` enthält
`options_for_training(steps=120,interval=20)`.
Nur wenn der Nutzer beim **künftigen separaten Forschungsjob**
`--checkpoint-interval 20` ausdrücklich angibt, gelten:

- alle 20 Schritte automatische Validierung auf dem bereits
  getrennten, lokalen Validierungssatz;
- alle 20 Schritte Speicherung eines privaten Checkpoints;
- höchstens zwei Checkpoints aufbewahrt;
- Auswahl des Checkpoints mit **niedrigstem eval_loss**;
- `load_best_model_at_end=True`; der Trainer stellt
  vor `policy.save_pretrained` die Modellgewichte des
  ausgewählten Checkpoints wieder her;
- Prüfung von echtem privaten Pfad, tatsächlichem
  `checkpoint-{step}`, endlichem Validierungs-Loss und
  passender Intervallgrenze, bevor ein Adapter gespeichert wird.

**Unveränderter Standard:** Wer keinen Checkpoint-Parameter
angibt, erhält exakt `eval_strategy="no"` und
`save_strategy="no"`. Historische v6/v6.1-
Experimentkonfigurationen und ihre Checkpoint-SHAs
bleiben unverändert.

**Verfügbarkeit:** Der Parameter ist ausschließlich in der
lokalen experimentellen CLI `scripts.neral_v61_replay_train`
vorbereitet; es wird **kein neuer Trainingsworkflow gestartet**,
kein privater Checkpoint hochgeladen und **kein Adapter promotet**.

Beispiel für einen künftigen **separat autorisierten**
RTX-4070-Forschungsversuch (nur wenn andere Trainingsdaten
fachlich überprüft sind):

```powershell
python -m scripts.neral_v61_replay_train --mode gpu-train --out NEUER_PRIVATER_ORDNER --base PINNED_BASE --v5-adapter PINNED_V5 --authorize TRAIN_NERAL_V61_R16_120_REPLAY_EXPERIMENTAL --checkpoint-interval 20
```

Das `eval_loss`-Minimum liefert höchstens eine bessere
**in-domain**-Modellauswahl. Es schützt **nicht**
garantiert vor Regressionen auf neuen Aufgaben. Für eine
Releasefreigabe gelten unverändert unabhängige fachbereichs-
übergreifende Tests, SHA-Pinning, gedrosselte Promotion und
vor allem die bereits bestehende Modellquarantäne.

Den Checkpointmodus bitte nicht rückwirkend auf
die alten bereits abgeschlossenen v6/v6.1-Läufe anwenden.
