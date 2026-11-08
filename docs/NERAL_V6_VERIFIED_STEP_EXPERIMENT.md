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
bleiben unausgeführt. Der Vergleich hat jetzt einen eigenen Einstieg,
`.github/workflows/neral-v6-slice-compare-gpu.yml`. Er startet nur
manuell auf `main` und nur mit einer eigenen Freigabe. Diese
CPU-Prüfung startet ihn nicht. Das Phase-2-Training bleibt ein
zweiter, späterer Lauf.

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
ein Kandidat festliegt. Der noch nicht benutzte Satz ist das
versiegelte Holdout weiter unten. Der Phase-2-Gate und die
Experimententscheidung dürfen seine Item-IDs nicht lesen.
Auch ein besseres Screening ist kein Release.

## Messschnitte, kein Lernschritt

Vor diesem Stand gab es für das Experiment keinen eingefrorenen,
disjunkten und getesteten Dreischnitt aus Retention, Transfer und
versiegeltem End-Holdout.

Was schon da war, ist nicht dieser Schnitt:

- Komposition hat 600 Trainings-, 100 Validierungs- und 100
  Holdout-Zeilen, fünf Familien, andere Schablonen je Split. Das
  Holdout ist per exaktem Prompt vom Training ausgeschlossen. Der
  Trainingsbau liest diese 100 Prompts, um dieselbe Zeile abzulehnen.
  Item-IDs hatte der Split nicht, und er war nicht als Retention
  beschriftet.
- Die frische Dev72 und die v6.1-72 liegen außerhalb der
  Trainingsfamilien. Sie sind damit transfer-ähnlich, aber verbrauchte
  Entwicklungsscreens. Die historische 29/24/21-Auszählung bleibt kein
  Fähigkeitsmaß und kein Transfer-Nachweis.
- Alle 128 Gegenbeispiele aus Seed `20261107` liegen im Training.
  Zurückgehaltene Indizes gibt es in diesem Seed nicht.

Neu ist nur die Messung, in `scripts/neral_v6_measurement_slices.py`
und `scripts/neral_v6_locked_holdout.py`. Kein Trainingsziel ändert
sich.

Die Zuordnung hängt nur von Familie und Index ab, nicht von einer
Punktzahl. Dieselben Funktionen liefern dieselben Fälle.

- Retention, 164 Fälle. Alle 100 Kompositions-Holdouts, Index 0 bis 19,
  gleiche Familie, andere Schablone. Dazu 64 Gegenbeispiele aus Seed
  `20261108`, Paarindex 0 bis 15, beide Varianten, `ordered_price` und
  `segmented_fuel`.
- Transfer, 64 Fälle. Acht Familien, die im Training, in den beiden
  verbrauchten Screens und im versiegelten Holdout nicht vorkommen.
  Die ersten vier bleiben `inclusive_range_sum`, `nesting_depth`,
  `inversion_pairs` und `running_maximum_changes`. Dazu kommen
  `digit_sum`, `adjacent_ascents`, `alternating_sum` und
  `capped_sum`. Je Index 0 bis 7. Klammertiefe steht in den Wörtern
  `open` und `close`, damit ein Satzzeichen-Filter die Fälle nicht
  zusammenfallen lässt. Die vier zusätzlichen Familien sind vor jedem
  GPU-Lauf dazugekommen: 32 Aufgaben können eine einseitige gepaarte
  Lücke unter 18,8 Prozentpunkten nicht signifikant machen.
- Versiegeltes Holdout, 96 Fälle, nur im zweiten Modul. Paarindex 16
  bis 31 derselben Mess-Seed, 64 Fälle. Dazu `odd_median`,
  `population_bitcount`, `distinct_values` und
  `parenthesized_product_sum`, je 8. Diese vier Familien sind nicht
  die Transfer-Familien, damit eine Auswahl auf dem Transfer-Schnitt
  nicht dieselben Aufgaben öffnet.

SHA-256 der kanonischen JSONL, sortierte Schlüssel:

- Retention `333bfcb4e92a9b9d6d27c2d9a6e6181c13f9c629a3a8d2bb35885a104a319054`
- Transfer `9604fe1910698e572d408e58e478f7446f393409d0997db3ab2bd99f769b48a2`
- Holdout `6332c37ec3927475aba83da312f7943064d7c400be516caaf1dcf085cf9c3270`

Der Vergleich in `scripts/neral_v6_slice_compare.py` ist hier nicht
gelaufen. Er braucht `matched_fewshot`, greedy, 160 neue Tokens und
dieselben Nachrichten für Qwen, v5 und v6. Ohne dieses Protokoll
bleibt der Gate bei `refuse`.

Exakte normalisierte Prompts schneiden sich nicht: die drei Schnitte
untereinander, und keiner mit Train, Validierung oder den beiden
Dev72. 10-Gramme gegen die Dev-Screens sind 0. Die acht Transfer-
Familien und die vier neuen Holdout-Familien haben auch gegen
Trainingsprompts und Trainingsantworten null 10-Gramme.

10-Gramme innerhalb derselben Familie trennen die Fälle nicht.
138 der 164 Retentionsprompts und alle 64 Gegenbeispiel-Holdouts
teilen ein 10-Gramm mit einem Trainingsprompt. Preis, Treibstoff und
die Kompositionsschablonen verwenden dieselben Wörter. Die Prompts
sind trotzdem nicht gleich. 74 der 100 Kompositions-Holdouts tragen
denselben Effekt.

`phase2_training_gate`, `experiment_decision`, `analyze_dev72` und
`audit_candidate_report` lehnen einen Bericht ab, der Holdout-IDs,
Holdout-Fälle oder ein Holdout-Aggregat enthält. Die Ablehnung
kopiert die IDs nicht ins Ergebnis. Diese Dateien importieren das
Holdout-Modul nicht. Der Trainingsbau importiert es nicht.
`sealed_holdout_counts` gibt nur die vier Zähler und Hashes zurück.
`training_authorized` und `claimed_score_gain` bleiben falsch.

Das ist eine Scorer- und Protokolländerung. Kein Training, kein
Adapter, kein Fähigkeitsgewinn. Ein grüner Test zeigt, dass der
Schnitt stabil ist, nicht dass ein Modell besser rechnet.

## Prüfung der Trainingsmischung

Geprüft wurde `build_experiment` für dieses Experiment: 600
Kompositionszeilen, 128 Gegenbeispiele, 100 Validierungszeilen.
Der Verified-Pfad lief mit `length_source=bound`. Das v6.1-Replay
ist eine andere Mischung und nicht dieser Lauf.

Gegen die frische Dev72 und die v6.1-72 ist die Überlappung exakt 0
und als 10-Gramm 0, in beide Richtungen. Exakt gegen das
Kompositions-Holdout: 0. Als 10-Gramm: 74 von 100, siehe oben.
Train gegen Validierung exakt: 0.

Zwischen `compact` und `verified_steps` ändern sich genau 128
Assistententexte. Alle stammen aus `candidate()` mit Seed
`20261107`. Die Kompaktform hat in keiner dieser 128 Zeilen ein `*`
oder `+`. Die von `verify_case` gesperrte Langform hat in allen 128
beides. Die Endzahl bleibt. Die 600 Kompositionszeilen und die 100
Validierungszeilen sind zwischen beiden Zielen gleich. Keine davon
trägt `first=` oder `a_per_min=`.

200 Trainingszeilen haben ein CHECK, das der Generator so schreibt,
mit `×` und `+`, beim Inventar auch mit `-`. 400 Zeilen sind nur die
Endantwort, weil `make_case` bei `index % 3 != 0` keine Prüfung
einsetzt. Das ist keine nachträgliche Streichung einer schon
geprüften Langform. Das Zeitzonen-CHECK nennt Minute und Tag, nicht
die Addition als Ausdruck. Ein `+` darin ist der Tagesversatz.

Die Trainingskonstruktion bleibt unverändert. Die 400
Antwort-only-Zeilen zu füllen würde die Behandlung über die 128
Ziele hinaus erweitern.

## Verwechslung, Beobachtung, belegte Ursache

Nicht als Ursache verwendet:

- Die historischen 29/24/21. Sie werden hier nicht neu bewertet.
- 10-Gramm-Treffer in derselben Familie. Sie zeigen gemeinsame
  Wörter, nicht denselben Prompt.
- Die 400 Antwort-only-Zeilen. Für sie gab es keine gestrichene
  Langform der Art der 128.

Beobachtet:

- Genau 128 Ziele verlieren `*` und `+` in der Kompaktfassung.
- Exakte Train/Dev-Überlappung ist 0. 10-Gramm gegen die Dev-Screens
  ist 0.
- 74 von 100 Kompositions-Holdout-Prompts teilen ein 10-Gramm mit
  dem Training.

Belegte Ursache der fehlenden Rechenzeichen: `_compact_trace` in
`scripts/neral_v6_r16_gpu.py`, nur auf diesen 128 Zeilen. Nicht
belegt ist, dass diese Streichung die historischen Dev72-Zahlen
erzeugt hat. Ein `matched_fewshot`-Vergleich, der das messen könnte,
ist nicht gelaufen.

## Vergleichspfad, noch nicht gelaufen

Kein bisheriger GPU-Einstieg bewertet Basis, v5 und v6 gemeinsam auf
Retention und Transfer unter `matched_fewshot`. Die Dev72-Workflows
messen verbrauchte 72er-Suiten. Der Sechs-Arm-Lauf misst v6.1, nicht
v6, und ebenfalls die verbrauchte Suite. `neral-v6-verified-steps-train.yml`
trainiert und bewertet diese Schnitte nicht.

Neu ist `scripts/neral_v6_slice_compare.py`. Der Lauf prüft die beiden
SHA-256 am Anfang, lädt das Holdout nicht, schreibt nur in einen neuen
Ordner und bricht ab, wenn die drei Arme nicht dieselben Nachrichten
und dasselbe Decoding-Objekt sehen: greedy, `do_sample=false`, 160
neue Tokens. Je Schnitt gibt es die vier Ausgänge
`strict_correct`, `wrong_readable_number`, `unreadable`,
`incomplete`. Je Paar gibt es den exakten McNemar-Wert, ein
Item-Bootstrap und ein Familien-Cluster-Bootstrap.
`training_authorized` und `claimed_score_gain` bleiben falsch.

Ein CPU-Trockenlauf mit einem deterministischen Stub hat Promptbau,
Few-Shot-Gleichheit, Decoding, Bewertung, Bericht und Gate durchlaufen.
Der Stub ist kein Modell. Seine Zähler sind kein Fähigkeitsergebnis.
Das Gate sagt dabei `decision=ok` und
`phase2_training=v61_comparison_not_in_report`. `ok` heißt nur, dass
das Protokoll zulässig ist.

Mit 64 Transfer-Aufgaben ist die kleinste einseitige Lücke mit
zweiseitigem exaktem p ≤ 0,05 weiter sechs Diskordanzen, also 9,4
Prozentpunkte, und nur wenn die Gegenrichtung null ist. Fünf gegen
null ergeben p = 0,0625. Eine Lücke um 15 Prozentpunkte bei etwa einem
Viertel diskordanter Paare braucht in der üblichen McNemar-Näherung
rund 85 unabhängige Aufgaben für etwa 80 Prozent Power. Die acht
Familien sind keine 64 unabhängigen Aufgaben. Ein Intervall, das null
enthält, ist kein Beleg für Gleichheit.

Die Reihenfolge bleibt Auswertung vor Training. Erst der
Nur-Lese-Vergleich zeigt, ob v6 unter gleichem Prompt auf der
Retention mehr falsch lesbare Zahlen hat als v5. Wenn nicht, ist das
Training nicht begründet. Der Trainings-Workflow bleibt dafür ein
eigener Lauf und ist in diesem Vorschlag nicht enthalten.
