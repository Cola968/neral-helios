# Trainingsvorbereitung: geprüfte Rechenschritte wieder in die Zieltexte

PR 365 bleibt die Messkorrektur und enthält diese Änderung nicht.
Diese Vorbereitung ist https://github.com/Cola968/neral-helios/pull/366,
Zweig `cursor/verified-step-targets-f78c`.
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

## Längenbudget, kein Fähigkeitsbeweis

`tokenized_completion` nimmt eine Zeile nur an, wenn die Chat-Vorlage
als Prefix der vollen Sequenz tokenisiert, der Assistant-Anteil nicht
leer ist und `len(full) <= 768`. Sonst bricht Phase 2 ab, bevor ein
Adapter geschrieben wird. Die Vorlage ist `neral_hf_identity.CHAT_TEMPLATE`,
nicht die längere Tool-Vorlage in `tokenizer_config.json`.

Die Basis ist `Qwen/Qwen2.5-Coder-7B-Instruct`, Revision
`c03e6d358207e414f1eca0bb1891e29f1db0e242`. Für die Messung liegen nur
die öffentlichen Tokenizer-Dateien lokal, keine Gewichte:
`tokenizer.json`, `tokenizer_config.json`, `vocab.json`, `merges.txt`.
Ihre SHA-256 stehen in `PINNED_TOKENIZER_SHA256` und in
`docs/v6_verified_steps_token_audit.json`. Der Download kam vom
öffentlichen Hugging-Face-Endpunkt dieser Revision (`X-Repo-Commit`
gleich der Revision). Ein Lauf ohne diese Dateien lädt sie nicht nach.

Dieselbe Funktion `tokenized_completion` zählt jede unveränderte Zeile.
Von 828 Zeilen (600 Komposition, 100 Validierung, 128 Gegenbeispiele)
liegt keine über 768. Das Maximum ist 451, bei `segmented_fuel`.
`ordered_price` endet bei 327. Die Byte-Schranke bleibt eine Obergrenze:
70 der 128 Gegenbeispiele liegen dort über 768, Maximum 933, und in
keiner Zeile ist die Schranke kürzer als die echte Tokenzahl. Der
kleinste Abstand ist 426 Token. Die Schranke hat den Überlauf also
nicht belegt.

Verdichtung läuft nur, wenn die gewählte Länge über 768 liegt.
`length_source=exact` behält alle 128 Zieltexte wörtlich. Der
GPU-Einstieg lädt den Tokenizer aus `--base`, prüft die vier SHA-256
und nutzt diesen Pfad. `length_source=bound` bleibt der Pfad ohne
Tokenizer: dieselben 70 Zeilen verlieren Ballast bis die Schranke passt.
58 Preiszeilen bleiben wörtlich, 6 verlieren nur den Rundungssatz.
Alle 64 Treibstoffzeilen verlieren Einheiten und Knotennamen; 6 behalten
die Klammern, 42 verlieren sie, 16 verlieren zusätzlich die Leerzeichen
an den Rechenzeichen. Danach ist das Byte-Maximum 768, zwei Zeilen
liegen auf dem Budget. Tests, die
diese Zahlen prüfen, setzen `length_source=bound` und ändern sich
nicht, wenn zufällig ein Tokenizer auf der Maschine liegt. Fehlen die
gepinnnten Dateien, überspringt der Exakt-Test mit dieser Begründung.

Jede Verdichtung wertet jede Gleichung davor und danach aus. Erlaubt
sind nur ganze Zahlen, `+`, `-`, `*`, `/` und Klammern, über den
AST, ohne `eval`. Weicht ein Wert oder `FINAL_ANSWER` ab, bricht der
Lauf ab. Die 128 echten Zeilen bleiben unter dieser Prüfung gleich.
Ein künstliches `(2) * (3 + 4)` gegen `2 * 3 + 4` wird abgelehnt,
obwohl Ziffern und Rechenzeichen noch übereinstimmen.

Das ist eine Zulassungsprüfung, damit Phase 2 nicht an der Tokengrenze
abbricht. Es ist kein Beleg, dass die wörtlichen oder die verkürzten
Texte besser rechnen, und kein Fähigkeitsgewinn. Ob sich die Fähigkeit
ändert, bleibt offen, bis ein freigegebener `matched_fewshot`-Vergleich
das misst.

Der GPU-Vergleich mit `matched_fewshot` und das Phase-2-Training
bleiben hier blockiert. Beides braucht eine ausdrückliche Freigabe
für genau diesen Lauf. Diese CPU-Prüfung startet keines von beiden.

## Phase 1, bevor irgendjemand trainiert

Bestehende Adapter, kein neues Training. Dieselbe Aufgabe, dieselbe
Vorlage, dieselben drei Beispiele, greedy, 160 neue Tokens, für Qwen,
v5 und den vorhandenen v6-Adapter. Dafür gibt es
`--prompt-protocol matched_fewshot`. Ohne dieses Flag bleibt der alte
ungleiche Vergleich (nur Qwen sieht Beispiele).

Der gemeinsame GPU-Aufruf behält die injizierte `messages_factory` aus
dem Sechs-Arm-Lauf. Diese Factory ist die geprüfte Vorlage dieses
Laufs und wird nicht vom Protokoll umgeschrieben. Fehlt sie, gilt
`--prompt-protocol`. Der Standard bleibt `historical`.

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
`1e-5`, 728 Trainingszeilen, 100 Validierungszeilen. Der gemeinsame
Trainer kann seit main optionale Validierungs-Checkpoints wählen und
den besten Adapter vor dem Speichern zurücksetzen. Dieser Lauf übergibt
kein Intervall. Ohne Intervall bleiben `eval_strategy=no` und
`save_strategy=no`, also der bisherige letzte Schritt. Die CPU-Prüfung
oben verweigert das Bauen der Daten, wenn die gewählte Länge 768
übersteigt. Ohne gepinnten Tokenizer ist das die Byte-Schranke. Mit
dem Tokenizer aus `--base` ist es die echte Tokenzahl, und die
gemessenen Zeilen bleiben dann wörtlich. Der GPU-Lauf prüft danach
noch einmal mit `tokenized_completion` und speichert bei einem
Grenzfehler keinen Adapter.

Stop und Rollback: nicht-endlicher Verlust, abweichende Schrittzahl,
oder ein Adapter-Hash gleich v5 bedeutet keinen Kandidaten. v5 bleibt
der Rückfall. Ein besserer Validierungsverlust ist kein Fähigkeitsbeleg.
Promotion ist in der Ausgabe hart auf falsch gesetzt.

Ein Screening ist bestanden nur, wenn unter `matched_fewshot` die
falschen lesbaren Zahlen gegen v5 nicht zunehmen, auf der
Validierung und auf einem noch nicht zum Trainieren benutzten Satz.
Die verbrauchten 72 und jede externe Endsuite bleiben draußen, bis
ein Kandidat festliegt. Auch ein besseres Screening ist kein Release.
