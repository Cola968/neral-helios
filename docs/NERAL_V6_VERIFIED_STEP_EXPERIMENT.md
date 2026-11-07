# Trainingsvorbereitung: geprüfte Rechenschritte wieder in die Zieltexte

PR 365 bleibt die Messkorrektur und enthält diese Änderung nicht.
Diese Vorbereitung ist https://github.com/Cola968/neral-helios/pull/366,
Commit `4a160e42`, Zweig `cursor/verified-step-targets-f78c`.
Dies ist kein Training und kein neues Modellergebnis. Es gibt keine
erwartete Punktzahl.

## Was sich am Lernen ändert

Im bisherigen v6-Rezept ersetzt `build_experiment` die 128 geprüften
Gegenbeispiel-Antworten durch eine Kurzform. Die Kurzform behält die
Endzahl und streicht jedes `*` und `+`. Das steht in
`scripts/neral_v6_r16_gpu.py`: `_compact_trace` gegen die von
`verify_case` gesperrte Langform.

Beispiel, dieselbe Endzahl `358,26 EUR`:

- trainiert bisher: `first=2726623/100; with_fee=2734823/100; ...`
- geprüft und jetzt im Experiment: `(37351) * (100 - 27) / 100 = ...`
  und `(2726623/100) + (82) = ...`

Alle 128 Zeilen verlieren so die Rechenzeichen. Prompts, Endzahlen,
Lernrate `1e-5`, 120 Schritte, Rang 16 und die übrigen 600
Kompositionszeilen bleiben gleich. v6.1 (niedrigere Lernrate, 70/30-Replay)
wird nicht angefasst und nicht wiederholt.

Ob das die Rechenfähigkeit hebt, ist offen. Die längeren Zieltexte können
ausführlichere Antworten erzeugen. Ein reiner Formatgewinn zählt nicht.
Die Hypothese ist nur: der Optimierer sieht wieder die Operation, nicht
nur den Knotennamen.

## CPU-Nachweis

Die kompakte Fassung bleibt der Default. Das Experiment nutzt
`counterfactual_target=verified_steps`. Genau 128 Assistententexte
ändern sich, jede Endzahl bleibt, jedes neue Ziel enthält `*` und `+`.
Dev72, die neue 72er-Suite und das Kompositions-Holdout sind als
Prompts ausgeschlossen. Der Trainings-Hash der kompakten Fassung und der
Hash der wiederhergestellten Rechenschritte sind verschieden.

GPU-Start braucht die eigene Freigabe
`TRAIN_NERAL_V6_VERIFIED_STEPS_120_EXPERIMENTAL`. Die alte
Kompakt-Freigabe startet dieses Rezept nicht, und umgekehrt.

## Phase 1, bevor irgendjemand trainiert

Bestehende Adapter, kein neues Training. Dieselbe Aufgabe, dieselbe
Vorlage, dieselben drei Beispiele, greedy, 160 neue Tokens, für Qwen,
v5 und den vorhandenen v6-Adapter. Dafür gibt es
`--prompt-protocol matched_fewshot`. Ohne dieses Flag bleibt der alte
ungleiche Vergleich (nur Qwen sieht Beispiele).

Die Auswertung nennt jetzt vier Zähler, die zusammen die Suite ergeben:
`strict_correct`, `wrong_readable_number`, `unreadable`, `incomplete`.
Nur `wrong_readable_number` ist ein Rechenbefund. `unreadable` ist
Messung. Die alten Schlüssel `correct` und `ungraded_completed` bleiben
dieselben Zahlen wie `strict_correct` und `unreadable`. Der Parser ist
unverändert. 29/24/21 wird nicht neu bewertet. Diese Zähler sind
Instrumentierung, kein Fähigkeitsgewinn.

`phase2_training_gate` lehnt `historical` und `matched_zeroshot` ab.
Nur `matched_fewshot` kann `decision=ok` werden. `ok` heißt nicht
trainieren: `training_authorized` bleibt falsch, `claimed_score_gain`
bleibt falsch. Liegt im selben Bericht v6.1 bei den falschen lesbaren
Zahlen auf Höhe von v5 oder besser, lautet die Empfehlung
`skip_training`.

Ausgabe nur in einen neuen Ordner. Kein Überschreiben von v5, v6 oder
v6.1. Dieser CPU-Stand startet weder den Vergleich noch Phase 2.
Beides braucht eine ausdrückliche Freigabe.

`scripts.neral_v61_new_dev_eval --mode run` akzeptiert denselben Schalter
für den schon trainierten v6.1-Adapter. Liegt v6.1 unter
`matched_fewshot` bereits auf Höhe von v5, Phase 2 nicht starten.

## Phase 2, nur nach Freigabe

Workflow `.github/workflows/neral-v6-verified-steps-train.yml`.
Start nur manuell auf `main` nach dem Merge, mit der neuen Freigabe.
Checkout ist `main`. Ausgabe:

`%USERPROFILE%\NERAL-QLoRA-Research\v6-verified-steps-120-<Lauf>-<Versuch>\candidate-adapter`

Nicht `v6-r16-120-...` und nicht `v61-replay-120-...`. Basis ist das
versiegelte v5, nicht ein Überschreiben davon. 120 Schritte, Lernrate
`1e-5`, 728 Trainingszeilen, 100 Validierungszeilen. Passt eine Zeile
nicht in 768 Tokens, bricht der Lauf vor dem Speichern ab.

Stop und Rollback: nicht-endlicher Verlust, abweichende Schrittzahl,
oder ein Adapter-Hash gleich v5 bedeutet keinen Kandidaten. v5 bleibt
der Rückfall. Ein besserer Validierungsverlust ist kein Fähigkeitsbeleg.
Promotion ist in der Ausgabe hart auf falsch gesetzt.

Ein Screening ist bestanden nur, wenn unter `matched_fewshot` die
falschen lesbaren Zahlen gegen v5 nicht zunehmen, auf der
Validierung und auf einem noch nicht zum Trainieren benutzten Satz.
Die verbrauchten 72 und jede externe Endsuite bleiben draußen, bis
ein Kandidat festliegt. Auch ein besseres Screening ist kein Release.
