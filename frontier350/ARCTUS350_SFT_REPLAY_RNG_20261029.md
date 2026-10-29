# Arctus350: Replay-Zufallsstrom für kontrollierte SFT-Vergleiche

## Beobachtung

Der vorhandene Trainer `train_chat350.py` verwendet dasselbe
`random.Random(seed + 1)` für die Reihenfolge der SFT-Batches und die Wahl
von Replay-Datei und Offset. Verschiedene SFT-Sampler verbrauchen verschieden
viele Zufallszahlen. Bei gleichem Seed können dadurch **auch andere Replay-
Abschnitte** gezogen werden, obwohl nur die Batch-Methode untersucht werden
soll. Das wurde mit den vorhandenen Samplern und rein synthetischen Dateien
reproduziert; es ist keine Aussage über die tatsächliche Modellqualität.

## Änderung

`--rng-streams legacy` ist weiterhin der Standard: SFT und Replay teilen sich
denselben Generator und behalten die historische Zufallsfolge.
`--rng-streams independent` erzeugt zwei separat initialisierte Generatoren:
`seed + 1` für SFT und `seed + 2` für Replay. Auch ein
`human-balanced`-, `bucketed`- oder `compact`-Sampler verändert dann
nicht mehr den Replay-Zufallsstrom. Die bisherige separate Replay-Evaluation
mit `seed + 91` bleibt unverändert.

Für einen späteren Vergleich müssen **beide** Arme `--rng-streams independent`
und denselben Seed verwenden. Dieselben Replay-Ausschnitte setzen außerdem
identische, gleich sortierte Dateien mit identischem Inhalt und identischer
Länge sowie gleiche Replay-Häufigkeit, Batch- und Sequenzgröße voraus. Diese
Änderung prüft oder pinnt die Dateien nicht automatisch; dafür sind separat
verifizierte lokale Dateiprüfsummen nötig.

Der Modus steht in den Run-/Metrik-/Checkpointinformationen; bei getrennten
Streams wird zusätzlich der Replay-RNG-Zustand im Resume-Artefakt gespeichert.
Dies implementiert **keine Wiederaufnahme**: Der gespeicherte RNG-Zustand allein
enthält nicht alle Batch-Queues und garantiert kein exaktes Fortsetzen.

## Prüfung und Grenze

Sechs neue lokale CPU-Testmethoden vergleichen reale Batch-Sampler und die
vorhandene `replay_batch`-Funktion auf kleinen synthetischen Dateien; der
Tensor-Adapter ist simuliert. Weitere 16 bereits vorhandene Sampler-Tests
blieben grün. Es wurde kein Modell geladen, trainiert oder evaluiert.

Die Änderung entfernt einen einzelnen Vergleichsstörfaktor. Sie beweist weder
bessere Arctus-Antworten noch verhindert sie für sich genommen Vergessen.
Realer Nutzen, Geschwindigkeit und GPU-Verhalten bleiben ungemessen. Ein
späteres Training oder ein PC-GPU-Test erfordert die ausdrückliche Freigabe.
