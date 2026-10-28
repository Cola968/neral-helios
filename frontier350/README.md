# Neral 350M Frontier v1

This is the first intended public-release Neral model line.

## Product contract

- ~350M parameters, trained from scratch on Neral byte tokens.
- 10B token target for v1 pretraining, with resumable local RTX execution.
- 8K architectural context ceiling.
- No benchmark/test examples may enter training.
- All external data must be revision-pinned, hash-verified and license-admitted.
- Model checkpoints and corpora stay local; GitHub receives report-only artifacts.
- Public release requires the full frozen post-training gate.

## Hard release gate

- NIS2 >= 82
- RawMath exact >= 42%
- targeted math >= 80%
- code >= 78%
- German >= 96%
- reasoning >= 92%
- context recall = 100%
- free EOS >= 85%
- repeated trigram <= 2.5%
- UTF-8 >= 99.9%
- BPB@4095 <= 1.32

Stretch goals deliberately exceed the previous target: NIS2 >= 86, RawMath >= 50%, targeted math >= 88%, code >= 84%, reasoning >= 95%, EOS >= 92%, repetition <= 1.5%, BPB <= 1.25.

These are targets, not claims. No metric is reported until measured by the frozen evaluation stack.

## Training stages

The initial 10B pretraining run uses a single model lineage. Dataset construction should target a controlled mixture rather than sequentially creating named models:

1. General high-quality multilingual/educational text: 6.0B tokens
2. Math/science/problem-solving text: 1.4B
3. Code/technical text: 1.0B
4. Dialogue/instruction-rich text: 1.0B
5. German high-quality reserve: 0.4B
6. EOS/format/anti-repetition focused reserve: 0.2B

Mixture boundaries are data-routing targets. They do not create separate model families. All checkpoints remain `Neral 350M Frontier v1` steps.

## Post-pretraining

The 10B run is not sufficient by itself for publication. After pretraining:

- contamination audit
- instruction SFT using separately admitted human/curated conversations
- math/code transfer repair only if held-out dev metrics support it
- EOS/anti-repetition tuning
- frozen NIS2 / generation / math / code / context evaluation
- Android memory, latency, crash, thermal and offline tests
- safety/privacy red-team suite

Only a checkpoint satisfying every hard release gate can be labelled a release candidate.

## Lokale Chat-Oberfläche

[Chat im PC-Browser starten und optional vom Handy im WLAN zugreifen](CHAT350_BROWSER.md). Für tatsächliche native 350M-Antworten sind kompatibler lokaler Checkpoint, gesperrter Tokenizer und BF16-CUDA nötig. Ein getrennter Werkzeug-Vorschaumodus funktioniert ohne Modell. UI-/Werkzeugtests sind **keine** Modell-Benchmarks. Der 1.200-Schritte-Pilot bleibt abgelehnt.

## Entwicklungsdiagnostik vorhandener Generierungsberichte (Issue #142)

`chat350_dev_generation_diagnostics.py` liest ausschließlich bereits erzeugte
`eval_chat350.py`-JSON-Berichte. Es benötigt nur die Python-Standardbibliothek,
öffnet keine Checkpoints, Tokenizer oder Case-Suites und ruft keine GPU auf.
Nur Entwicklungs-/synthetische Berichte verwenden; finale Benchmarkfälle
gehören weder in diese Entwicklungsdiagnostik noch in eine Trainingsauswahl.

```sh
python -m frontier350.chat350_dev_generation_diagnostics \
  --report dev-1b.json --min-words 3 --out diagnostics.json
python -m frontier350.chat350_dev_generation_diagnostics \
  --report dev-1b.json --report dev-2b.json --report dev-sft.json \
  --out comparison.json
python -m unittest tests.test_chat350_dev_generation_diagnostics -v
```

Auch der direkte Skriptaufruf funktioniert. **Exitcode 2 ist bei aktuellen
Berichten beabsichtigt:** Das JSON enthält Teilbeobachtungen und eine konkrete
Diagnose fehlender Metadaten, aber keinen bestandenen Vergleich. Ungültige
Eingaben erzeugen `status: "invalid_report"`; die Ausgabe ersetzt auch ein
eventuell vorhandenes Ergebnis. Eine Eingabedatei darf nicht die Ausgabedatei
sein. Die CLI gibt keine Antworttexte, Prompts oder Parser-Eingaben aus.

### Tatsächlich vorhandenes Eingabeschema und Messgrenzen

Die aktuelle Evaluator-Ausgabe hat `checkpoint_sha256`, `tokenizer_sha256`,
`case_sha256`, `evaluation_contract: "segment_bpe_mask_specials_v1"`, `summary`
und `rows`. Jede Zeile enthält `id`, `category`, `text`, unveränderte
`generated_ids` und optional `expected_exact`. `summary` enthält `cases`,
`eos_fraction`, `mean_repeated_trigram_fraction`, `mean_distinct2_fraction`,
`mean_generated_tokens_before_eos`, `utf8_fraction`, `exact_instruction_pct`.
Ein unbekannter Vertrags-/Schema-Versionswert wird abgelehnt.

| Kennzahl | Beobachtung / Grenze |
| --- | --- |
| Empty | Gespeicherter decodierter Text ist nach NFC-/Whitespace-Normalisierung leer. |
| Very short | Weniger als `--min-words` Wörter, **einschließlich** leerer Antworten. Unicode-Buchstaben/Zahlen bilden Wörter; interne Apostrophe und Bindestriche bleiben im Wort. Umlaute werden vor dem Zählen NFC-normalisiert. Satzzeichen allein ergeben null Wörter, sind aber kein leerer Text. |
| Tokenlänge | Anzahl aller gespeicherten Token-IDs, einschließlich möglichem Stop-Token und Tokens hinter der abgeschnittenen Textgrenze. |
| Repetition | Vorhandene `repeated_ngram_fraction(ids, 3)` / `distinct_ngram_fraction(ids, 2)` auf **allen rohen IDs**. Diese Werte sind ausdrücklich nicht die EOS-bereinigten Inhaltsmetriken aus `summary`. Unter drei Tokens: Repetition 0; unter zwei Tokens: distinct2 1, entsprechend den vorhandenen Hilfen. |
| Exaktes / normalisiertes / Prefix-Echo | `not_measured`: Der letzte User-Text fehlt im Bericht. Kein Rückschluss aus einer ähnlich klingenden Antwort oder `expected_exact`. |
| EOS | `not_measured`: Die tatsächliche EOS-ID fehlt. `reported_eos` bewahrt nur den überlieferten `summary.eos_fraction` als `reported_unverified`, ohne erfundenen Zähler. |
| Max-new ohne EOS | `not_measured`: EOS-ID und Tokenlimit pro Fall fehlen; ein Kontextlimit oder eine abgeschnittene Antwort beweist kein ausgeschöpftes max-new-Budget. |

Der bestehende Evaluator kürzt `text` vor einem halluzinierten `\nUser:` und
behält die rohen Token-IDs. Text- und Tokenmetriken können deshalb bewusst
verschiedene Abschnitte beschreiben. Ob eine Grenze schon entfernt wurde, ist
`unknown`; aus gespeichertem Text lässt sich keine vollständige
Halluzinationsrate rekonstruieren. Noch ungekürzte `\nUser:`-Grenzen werden als
Vertragswiderspruch abgelehnt. Ein `Assistant:`-Header wird niemals entfernt,
um vermeintliche Echos zu erzeugen.

Die reine Funktion `measure_prompt_echo(prefix, text)` ist für synthetische
Regressionstests und die Integration in eine **nächste** Evaluator-Version
vorbereitet. Sie nutzt den bestehenden Chat-Prefix-Parser und die identische
Halluzinationsgrenze. Exakt bedeutet Zeichenidentität mit dem geparsten letzten
User-Inhalt; normalisiert bedeutet NFC plus vereinheitlichte Whitespace-Sequenzen.
Großschreibung, Satzzeichen, `ß` und Umlaute bleiben verschieden. Prefix-Echo
enthält volle Wiederholungen und beginnt mit dem **vollständigen** normalisierten
User-Text; `Ja` → `Jahr` zählt nicht. Keine Fuzzy- oder semantische Ähnlichkeit.
Die Bericht-CLI nimmt **keine** Originalprompts als zusätzliche Eingabe an.

### Ausgabe und Vergleichsvertrag

Das deterministische JSON (`diagnostics_schema_version: 1`) enthält
`status`, `valid_comparison: false`, `automatic_promotion: false`, `min_words`,
`provenance`, `missing_metadata`, `overall` und `by_category`.
Gemessene Raten haben `status`, `basis`, `numerator`, `denominator`, `rate`.
Längen und Repetition haben `sum`, `mean`, `basis`; ihr Nenner ist die jeweilige
Anzahl `cases`. Nicht gemessene Raten haben `numerator: null`, `denominator: 0`,
`rate: null` und einen Grund. Null bedeutet fehlende Messung, niemals null Fehler.
Prompt-, Antwort- und Zieltexte sowie Token-Arrays werden nicht kopiert.

Mehrere Berichte werden erst einzeln validiert und dann gegen **alle** anderen
auf identische Case-IDs, pro Fall identische Kategorien sowie identische
Case-Suite-/Tokenizer-SHA und `evaluation_contract` geprüft. Die Prüfergebnisse
stehen in `checked_identity`, einschließlich `observed_case_ids` und
`category_counts`. Jede Untergruppe bleibt in jedem Bericht sichtbar. Fehlende,
leere oder doppelte IDs, ungültige Typen/Hashes, NaN/Infinity (auch in Zusatzfeldern),
doppelte JSON-Schlüssel, widersprüchliche Fallzahlen, EOS-/Längen-Arithmetik und
Exact-/UTF-8-Summaries führen zum Abbruch. Unbekannte EOS-bereinigte
Repetitionssummaries können nur auf Typ, Endlichkeit und Wertebereich geprüft
werden; sie werden nicht als nachgerechnete Messungen ausgegeben.

Hashes sind **Herkunftsbehauptungen des Berichts**, keine Prüfung der Originaldateien.
Selbst identische beobachtete Case-Sets können gemeinsam unvollständig sein.
Ohne erwartete Case-IDs und Entwicklungsprovenienz bleibt Suite-Vollständigkeit
`unknown`. Ohne gespeicherte Generationseinstellungen ist ihre Gleichheit
unbelegt. Daher ist das Vergleichsergebnis aktueller Berichte immer
`status: "not_comparable"`, mit Exitcode 2 und ohne Rangfolge/Freigabe.

### Minimaler Vorschlag für die nächste Evaluator-Version (hier nicht geändert)

Eine explizit versionierte Erweiterung sollte beim Erzeugen der Berichte
zusätzlich festhalten:

1. Entwicklungs-/synthetische Suite-Provenienz, erwartete Case-IDs und Zuordnung
   zu Kategorien sowie die bisherigen unveränderten SHA-Fingerprints.
2. Tatsächliche EOS-ID, Generationseinstellungen (Decoderstrategie, Special-Token-
   Maske, Samplingparameter/Seed soweit anwendbar, Kontextlimit) und pro Fall
   `max_new_tokens`, Prompt-Tokenlänge und beobachteten Stopgrund.
3. Exakte, normalisierte und Prefix-Echo-Flags direkt am letzten User-Text vor
   dessen Verwerfen berechnen; dazu Normalisierungs-/Messversion und ein Flag
   für das Abschneiden halluzinierter User-Turns. So braucht der Offline-Bericht
   keine Rohprompts. Für eine unabhängige Neuberechnung der Echo-Flags wären
   zusätzliche zugriffsgeschützte Promptdaten nötig; ein Hash allein genügt nicht.

Diese Angaben sind ein Schema-Vorschlag, **keine** rückwirkend angenommenen
Felder alter Berichte. Das Diagnosemodul akzeptiert daraus keine erfundene
Vollständigkeit. Auf dem Format von `eval_chat350.py` am Branch-Ausgangspunkt
lassen sich Short/Empty und rohe Repetition messen, Echo/Stop jedoch nicht
vollständig getrennt nachrechnen und 1B/2B/SFT nicht gültig vergleichen.
Das belegt weder eine Verbesserung noch eine Verschlechterung der Modelle.
