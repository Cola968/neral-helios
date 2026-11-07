# NERAL v6: 600er-Testset aufnehmen, ohne falsche Unabhängigkeit zu behaupten

**Stand:** CPU-Prüfwerkzeug für *extern erstellte oder unabhängig redaktionell
geprüfte* Aufgaben. Der Code generiert **nicht** automatisch 600 wirklich neue
unbekannte Aufgaben, gibt kein Release frei und startet weder GPU noch Training.

## Was eine echte v6-Testfreigabe benötigt

Die Modellgewichte werden mit einer neuen **Release-Suite** aus 600 Aufgaben
bewertet. Eine zweite, frei nutzbare **Dev-Suite** enthält ebenfalls 600
Aufgaben und wird getrennt gehalten. Jede Suite enthält jeweils:

| Gebiet | Fälle | Antwortprüfung |
| --- | ---: | --- |
| Mathematik | 200 | unabhängige arithmetische/symbolische Oracle-Prüfung |
| Coding/Debugging | 200 | separate, versteckte Tests in isolierter Sandbox |
| Logik/Reasoning | 200 | nachvollziehbare externe Wahrheitstabellen/Oracles |

Jeder Bereich benötigt mindestens fünf wirklich unterschiedliche
Methodenfamilien, nicht bloß verschiedene Zahlen derselben Schablone.
Bestehende NERAL-EXT-400-Aufgaben dürfen nicht in die neue gesperrte Suite
kopiert werden. Öffentlich verfügbare Benchmarks können bereits in
Qwens ursprünglichem Training vorgekommen sein; eine 10-Gramm-Prüfung
beweist keine Unbekanntheit gegenüber vortrainierten Gewichten.

Nur die von Personen/externen Prüfern verantworteten Rohdateien dürfen als
Release-Daten betrachtet werden. Keine rohen privaten Testantworten
oder versteckten Python-Tests ins öffentliche GitHub-Repository hochladen.

## Private Dateien und feste Zeilenspalten

`v6-dev-600.jsonl` und `v6-release-600.jsonl`: jede Zeile hat genau
`id`, `domain`, `family`, `source_id`, `kind`, `prompt`,
`answer`, `tests_sha256`.

- `domain` ist `math`, `code` oder `reasoning`;
  die entsprechenden `kind`-Werte sind `numeric`, `python` oder
  `boolean`.
- `source_id` dokumentiert den **echten Ursprung** einer Aufgabe. Dieselbe
  ursprüngliche Aufgabe darf nicht über unterschiedlich formulierte Versionen
  sowohl in Dev als auch Release auftauchen.
- `tests_sha256` ist bei Codeaufgaben der SHA-256 der **separat und sicher**
  verwahrten Hidden-Unit-Tests, bei anderen Aufgaben `null`. Die
  Intake-Pipeline prüft nur die Form des Hashes; ob die Tests laufen und
  tatsächlich richtig sind, muss extern geprüft werden.
- `answer` enthält die unabhängige erwartete Antwort, darf nicht aus der
  aktuellen NERAL-Ausgabe übernommen werden.

**Alle** Trainingsdatensätze (v5 train/val, v6 train/val und spätere
ORPO-/DPO-Daten) müssen als lokale JSONL-Dateien übergeben werden.
Unterstützt werden `prompt` als String oder Chat-Liste und
`messages` mit genau einem `user`-Turn. Fehlt ein Trainingsbestand,
ist der spätere Audit unvollständig.

## CPU-Prüfung

```powershell
python -m unittest discover -s tests -p test_neral_v6_suite_intake.py -v
python -m scripts.neral_v6_suite_intake `
  --dev C:\private\v6-dev-600.jsonl `
  --release C:\private\v6-release-600.jsonl `
  --train-jsonl C:\private\v5-train.jsonl `
  --train-jsonl C:\private\v5-val.jsonl `
  --train-jsonl C:\private\v6-train.jsonl `
  --train-jsonl C:\private\v6-validation.jsonl `
  --out C:\private\v6-preflight-report.json
```

Weitere später entstandene Trainingsdateien jeweils ebenfalls mit
`--train-jsonl` hinzufügen und den Preflight **erneut** ausführen.
Das Tool verlangt 600+600 Fälle, 200 pro Bereich, Quell- und ID-Eindeutigkeit,
mindestens fünf Methodenfamilien je Gebiet, und verwirft wörtliche
normalisierte bzw. 10-Gramm-Überlappungen zwischen Dev/Release und
Trainingsdaten. Textähnlichkeit zwischen privaten Aufgaben darf nicht
durch neue IDs oder Leerzeichen versteckt werden.

Der `--out`-Pfad muss neu sein. Ein bestehendes Ergebnis wird nicht
überschrieben. Die Ausgabe enthält **nur SHA-256, Zählwerte und Warnflags**,
keine Aufgaben, Testdaten, Lösungen oder Modellantworten.
Das Ergebnis lautet ausdrücklich
`NERAL_V6_SUITE_INPUT_PREFLIGHT_ONLY` und
`release_validated=false`.

## Was der CPU-Preflight nicht kann

- Er **erstellt keine** 600 originellen unabhängigen Aufgaben und kann deren
  Quellen/Urheberschaft oder Bekanntheit im Pretraining nicht belegen.
- Er verifiziert keine semantischen Paraphrasen, Lizenzen, Antworten,
  Python-Sandbox oder tatsächlichen Test-Shas über die Struktur hinaus.
- Er **friert nicht selbst** die Daten in einer vorregistrierten und
  vorab unabhängig geprüften Suite ein und kann die Chronologie nicht
  überprüfen.
- Er garantiert nicht, dass alle zukünftigen Trainingsdaten vollständig
  in `--train-jsonl` übergeben wurden.

Erst nach fachlichem Spot-Check, versteckten echten Code-Tests,
vollständigem Dateninventar und manueller Prüfung:
Suite mit SHA aus dem Manifest **vor** der ersten Modellinferenz in
`GATE.md`/Vorregistrierungscommit verankern. Danach keine
Verwendung der Releasefälle für Datenmining, Hyperparametersuche oder
Fehlerkorrektur. Der vorhandene Statistikprüfer
`scripts/neral_release_gate.py` bleibt eine **separate Stufe**.
