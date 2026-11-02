# Phase 2.2: Benchmark-Verträge

## Genehmigte Nacharbeit am 2. November 2026

Der Betreiber hat Nacharbeit an diesem REVIEW_REQUIRED-Task ausdrücklich
freigegeben. Nach Stoppprüfung und unverändertem Remote-Head läuft der bestehende
Key regulär in Versuch 2. Keine neue Queue-ID umgeht eine Sperre.

Reproduzierter Defekt: Ein altes Importpaket mit bloßem Adapter-Hash wurde als
C/D akzeptiert, obwohl ID, Version und Integritätsprüfung fehlen. Der neue
Vergleichsvertrag 2 verlangt zusätzlich zu den bisherigen Settings:
`base_model_id`, `lora_adapter_id`, `lora_weights_version`, `adapter_hash`,
`adapter_integrity_verified`, `controller_active`, `context_control_active`.
A/B müssen alle Adapter-Identitäten und Hashes null und die Integritätsangabe
false halten. C/D brauchen eine nichtleere ID, Semver-Version, identische gültige
SHA-256-Werte und true; C und D müssen auch ID/Version teilen. Nur D aktiviert
Controller, Kontextsteuerung und controller_memory. Die öffentliche Funktion
`validate_condition_settings(name, setting)` prüft einzelne verfügbare Bedingungen,
damit ein Runner fehlende C/D sauber überspringen kann.

Ergebnisse müssen `completion_status`, `enabled_tools`, `used_tools` und
`tool_provenance` mitbringen. Jede Werkzeug-Provenienz enthält genau `tool_id`,
`call_index` (ab null, lückenlos) und `status`. Anzahl, aktivierte Werkzeuge und
genutzte IDs müssen übereinstimmen. A verweigert jede Werkzeugspur, auch bei
angeblicher Aufrufzahl null. Ein nicht vollständiger Abschlussstatus verhindert
volle Punktzahl. Nicht gemessene Tokenzahlen dürfen null bleiben; Aggregationen
weisen dann measured_cases aus, bei null Messungen auch Summe/Mittel/Maximum null.
Unbekannte Messwerte werden nicht in Nullen verwandelt.

Alte Run-Settings werden absichtlich nicht automatisch mit erfundenen
Integritätsangaben aufgefüllt. Manifest-Schema 1/2 und die alten acht Holdouts
werden nicht verändert. Berichtschema 2 nennt Vergleichsvertrag, Basismodell und
Adapter-Hash; die echte 20-%-Metrik und Regressionen bleiben erhalten.

CPU-TDD: Das alte Hash-only-Paket wurde zuerst unerwartet akzeptiert; aktualisierte
Positivfixtures und Abschlussstatusfälle schlugen vor der Implementierung fehl.
Ein weiterer RED-Test zeigte, dass unbekannte Tokenmessungen nicht darstellbar
waren. Aktuell **89 Evaluierungs-Vertragstests bestanden**. Separate vollständige
Windows-Modulsuite: **282 bestanden, 2 übersprungen, 2 abgewählt** (vorhandene
Symlink-/O_NOFOLLOW-Fixture-Voraussetzungen). Diese Zahlen überschneiden sich.

Importierte Integritäts-/Controller-Angaben sind überprüfte Vertragsbehauptungen,
kein Beweis tatsächlicher Modellinferenz oder Dateiintegrität. Der Runner muss
den Adapter lokal wirklich prüfen, statt diese Felder aus Nutzereingaben als
Wahrheit zu übernehmen. Synthetische Tests sind keine Qualitätsmessung. Keine
Inferenz, Downloads, GPU-/Trainingsläufe, Gewichtänderung, Zusatzkosten, Merges
oder Deployments; nachfolgende ältere Testergebnisse beziehen sich auf Linux.

Basis: PR #222, `c4c95d5ff2749cd544374f878621e84a2601b937`.
Dieses Paket ändert ausschließlich den vorhandenen Auswerter, seine Tests und
vorliegende Dokumentation. Keine Modellinferenz, kein Training, keine Änderung
an den acht eingefrorenen Pilotaufgaben und keine erfundenen Modellresultate.

## Behobene Defekte

Eine A-Ergebniszeile mit `tool_calls > 0` wurde bisher trotz werkzeugloser
Baseline akzeptiert. Der Import wird jetzt mit `ValueError` abgewiesen. A darf
keine Werkzeugkonfiguration oder berichtete Werkzeugausführung besitzen.

Tool-IDs benötigen eine explizite positive Version, beispielsweise `solve.v1`
oder `code_check.v1.2.0`. Unversionierte IDs und Version null werden abgewiesen.
Dies bindet eine deklarierte Version; es prüft keine tatsächlich installierten
Werkzeugartefakte oder Modellaufrufe.

## Schema 2

Schema 1 bleibt für bestehende eingefrorene Importverträge lesbar. Schema 2 nutzt
dieselben oberen Felder und ergänzt `split`: `train`, `dev` oder `final`.
Jede Aufgabe besitzt genau:

- `id`: eindeutig innerhalb des Manifests.
- `category`: wiederverwendbare Kategorie, zum Beispiel `coding`.
- `semantic_family`: deklarierte semantische Aufgabenfamilie; darf innerhalb
  derselben Partition mehrfach vorkommen.
- `prompt_sha256` und `rubric_sha256`: eingefrorene Inhaltszusagen;
  Prompt-Hashes bleiben innerhalb des Manifests eindeutig.
- `task_version`: explizite Version der Aufgabenstellung.
- `source`: genau `kind`, `reference`, `license`, jeweils nichtleere Strings.
- `review`: genau `status`, `reviewer`, `method`; Status muss `reviewed` sein.

DEV-Importe dürfen `external_unseen_assertion` und
`independent_reviewer_assertion` wahrheitsgemäß auf `false` setzen. Finale Importe
verlangen beide Zusagen als `true`. Das Programm beweist weder die Unabhängigkeit
noch die Wahrheit einer Zusage. Trainingspartitionen dürfen nur für den
Trennungscheck verwendet und nicht als Benchmark ausgewertet werden.

Der Bericht nennt `manifest_schema`, `split` und `category_counts`. Kategorien
sind frei deklarierbare Metadaten. Die Aufzählung beweist weder Vielfalt noch
Vollständigkeit der verlangten 30/25/20/15/10-Verteilung oder unabhängige Qualität.
Schema-1-Berichte haben `category_counts=null`, da Kategorien fehlen.
Der bestehende Bericht besitzt weiterhin seine eigene Berichtsschemaversion 1;
`manifest_schema` bezeichnet davon getrennt die Manifestversion.

## Explizite Migration und Trennungsprüfung

`migrate_manifest_v1(manifest, *, benchmark_version, split, task_metadata)` liefert
ein neues Objekt. Die neue Benchmarkversion muss vom alten Wert abweichen.
Der Aufrufer muss für jede alte ID explizit Kategorie, Aufgabenversion, Quelle
und Reviewdaten liefern. Ohne diese Informationen verweigert die Migration den
Vorgang, statt Herkunft oder Review zu erfinden. `semantic_group` wird zu
`semantic_family`; IDs und Prompt-/Rubrik-Hashes bleiben unverändert.
Bestehende Dateien werden nicht geschrieben oder überschrieben.

`validate_partition_separation(manifests)` prüft Schema-2-Manifeste gegen
deklarierte Familien und Prompt-Hashes. Dieselbe Familie oder derselbe Prompt
über verschiedene Partitionen ist ein Fehler. Unterschiedliche Namen und Hashes
können dennoch Paraphrasen oder semantische Verwandtschaft verbergen.
Die Rückgabe enthält deshalb stets `semantic_leakage_proven_absent=false`.
Der Check liest keine versiegelten Aufgabeninhalte und ersetzt weder
semantische Ähnlichkeitsprüfung noch unabhängiges Review. Er sollte nicht als
kompatibler Ersatz für das Arctus-Stagingaudit aus PR #208 bezeichnet werden.

## TDD und tatsächliche Nachweise

Vor der Implementierung: 11 beabsichtigte Regressionen fehlgeschlagen,
50 bestehende beziehungsweise negative Fälle bestanden. Darunter waren
A-Werkzeugzähler, fehlende Toolversionen, der gültige Schema-2-Import und die
fehlenden Migrations-/Partitionseinstiege.
Ein ergänzender RED-Lauf zeigte vier unkontrollierte `TypeError` bei Listen oder
Objekten als `split`. Auch diese Eingaben erzeugen jetzt kontrollierte
`ValueError`; Listen/Objekte als Reviewstatus werden ebenfalls geprüft.

Nach der Korrektur: 121 ausgewählte Tests bestanden. Auswahl:
`tests/neral_control/test_evaluation.py`, `tests/test_neral_blind_comparison.py`,
`tests/test_neral_project_sft.py`, `tests/test_neral_qlora_pilot.py` und
`tests/test_neral_adapter_candidate.py`. Davon 71 Auswertertests.
Alle neuen Benchmarkdaten in den Tests sind ausdrücklich synthetische
Vertragsfixtures und keine tatsächlich erzeugten Modellantworten.

## Grenzen

Die vier Bedingungen müssen weiterhin vollständig und mit identischen
Kerneinstellungen importiert werden. C und D verlangen einen Adapterhash;
sein Vorhandensein in einem Import beweist keine real geladenen Gewichte.
Die 20-%-Formel, ihr Nenner und die Erfassung von Regressionen bleiben erhalten.
Kein Qualitätsgewinn oder Release wird durch diese Vertragskorrekturen behauptet.

Gleiche Budgets pro Aufruf binden noch nicht die gesamten Modellaufruf-/Token-
oder Zeitbudgets, Hardware, Runtime, Tokenizer, Kontext oder Gedächtnissnapshot.
Der Bericht nennt diese Vergleichbarkeitsgrenze ausdrücklich. Die noch nötige
reale 100-Aufgaben-Sammlung, unabhängige finale Bewertung und ein tatsächlicher
Adaptervergleich sind separate Arbeitspakete. Windows-/GPU-Abnahmen wurden hier
nicht ausgeführt.

Der zusätzlich ausgeführte ungefilterte Repositorylauf ergab auf diesem
Worktree **877 bestanden, 2 übersprungen, 5 fehlgeschlagen**, außerdem eine
bestehende `SyntaxWarning` in `benchmark_gsm8k50.py:32`. Die fünf Fehler stammen
wie im unveränderten Ausgangsstand von fehlenden Daten/Checkpoints:

- `tests/test_curriculum_v2.py::test_v2_splits_and_hashes`
- `tests/test_curriculum_v2.py::test_pilot02_lineage`
- `tests/test_growth.py::test_40m_exact_count_and_inherited_weights_unchanged`
- `tests/test_regressions.py::test_curriculum_disjoint_and_masked`
- `tests/test_regressions.py::test_original_untouched_and_parameter_count`

Es wurden keine fehlenden Daten oder Gewichte zum Verbergen dieser Fehler
erzeugt. Die ausgewählten 121 Tests überlappen mit diesem Gesamtlauf; ihre Zahlen
werden nicht addiert. Vollständige Logs während der Ausführung:
`/tmp/neral-phase22-benchmark-red.txt`,
`/tmp/neral-phase22-benchmark-malformed-red.txt`,
`/tmp/neral-phase22-benchmark-regressions.txt` und
`/tmp/neral-phase22-benchmark-full.txt`.
