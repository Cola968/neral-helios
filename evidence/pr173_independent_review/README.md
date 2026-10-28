# Unabhängige Prüfung von PR #173 und optionale Arctus350-Vorbereitung

Geprüfter unveränderter Head: `11a7f9cfb98aa342d3b80921fa23afe44ff06228`,
Basis `main` `4e4dca3da0d8bc0d3ee5ec697edb7575cf9aa278`.
PR: https://github.com/Cola968/neral-helios/pull/173

Der Review las Diff und beteiligte Implementierungen selbst und führte Tests
in einem vollständigen Checkout aus. Der unveränderte Head wurde zusätzlich in
`/workspace/pr173-baseline` ausgecheckt und vollständig getestet. Die Zahlen des
alten PR-Berichts wurden nicht ungeprüft übernommen. Kein Merge.

## Konkrete Befunde und Korrekturen

| Priorität | Befund / Auswirkung | Korrektur / Nachweis |
|---|---|---|
| P2 | `sourceLabel` zeigte gespeicherte `trained-generation`-Antworten mit `source: tool:calculator` oder `no-model` als eigenes Modell an. | Widersprüchliche Quellen bleiben unverifiziert; `provenance-red.txt` → `provenance-green.txt`, final 20 Node-Fälle. |
| P2 | Tokenisierung/Kontextprüfung erfolgte vor Erwerb der Modellsperre. Ein gleichzeitiger Modellwechsel konnte Prompt und Modellkonfiguration verschiedenen Runtimes zuordnen. Bei zwei Tokenizerfamilien besonders relevant. | Tokenisierung und Kontextprüfung unter derselben Sperre wie Inferenz; Fehler geben sie frei. `tokenizer-race-red.txt`, danach gezielte und vollständige Tests grün bis auf die bekannten fehlenden Assets. |
| P2 | Nach Abbruch gesetztes Event unterdrückte bei anschließendem Inferenzfehler die terminale Stream-Fehlermeldung. Bei voller Queue konnte auch `done` verloren gehen. | Terminale Ergebnisse/Fehler unabhängig vom Stoppsignal einreihen, bei gestopptem vollem Stream alte Fortschrittsframes verwerfen. `cancel-error-red.txt` → `cancel-error-green.txt`, zusätzlicher Backpressure-/Freigabetest. |
| P2 | Modellkontext unterschied Checkpoints nur nach wiederverwendbarer ID. Andere Gewichte unter derselben ID konnten alten Kontext übernehmen. | Antwort-/Speicher-Provenienz umfasst Checkpoint-SHA; Kontext bei bekanntem SHA nur aus passenden eigenen vollständigen Antworten. `checkpoint-context-red.txt`, danach Node grün. |

Die Sperr- und Stream-Probleme waren bereits im bestehenden Runtime-Unterbau
vorhanden; sie werden hier als Integrations-/Zuverlässigkeitsbefunde dokumentiert,
nicht als ausschließlich durch PR #173 neu verursachte Fehler.

Zusätzlich für die optionale Integration: BPE-Kontext achtet weiter auf das
4096-Byte-API-Limit (`bpe-byte-budget-red.txt`), Anzeige verwendet Token statt
falscher Byte-Token-Angaben, Starter erkennt 349.825.152 Parameter, CI erfasst die
Adapterquellen und Tests. Keine Änderungen an Modellarchitektur oder Gewichten.

## Ausgeführte Validierung

| Prüfung | Ergebnis | Protokoll |
|---|---|---|
| Unveränderter Head, App-Suite | 113 bestanden | `baseline.txt` |
| Unveränderter Head, vollständiges pytest | 451 bestanden, 5 fehlgeschlagen, 1 Skip | `baseline-full-pytest.txt` |
| Überarbeitete Fassung, vollständiges pytest | 468 bestanden, dieselben 5 fehlgeschlagen, 1 Skip | `full-pytest-final.txt` |
| App-/Adapter-Suite vor den beiden letzten Streamfällen | 128 bestanden | `app-final.txt` |
| Stream-/Adapter-Suite einschließlich letzter Fälle | 35 bestanden | `stream-final.txt` |
| JavaScript, echte Testfälle direkt ausgeführt | 20 bestanden | `node-direct-final.txt` |
| Browser: bestehende Oberfläche | bestanden | `browser-existing.txt` |
| Browser: Sync, Streaming, Stoppen, Fehler mit expliziter Test-API | bestanden | `browser-workspace.txt` |
| Browser: echter Flask-Einstieg, leere Assets, kein Modell | bestanden | `browser-real-flask.txt` |
| Fehlende Probe-Dateien | Exit 1, `FileNotFoundError`, kein Modell geladen | `probe-missing-inputs.json` |

Protokolle sind nur um abschließende Leerzeichen bereinigt.
`git diff --check`, Node-Syntax und Python-Kompilierung erfolgreich.
Die unabhängige Gegenprüfung der neuen Anbindung fand keinen wichtigen offenen
Codebefund; die öffentliche API begrenzt Top-k weiterhin auf 260, obwohl der
Arctus-Adapter intern mehr unterstützt. Das ist in der Anleitung dokumentiert.

Die fünf Fehler sind **in beiden vollständigen Läufen identisch**:

- `tests/test_curriculum_v2.py::test_v2_splits_and_hashes` — `curriculum_v2`
- `tests/test_curriculum_v2.py::test_pilot02_lineage` — `export/pilot_02`
- `tests/test_growth.py::test_40m_exact_count_and_inherited_weights_unchanged` — `export/neral40m`
- `tests/test_regressions.py::test_curriculum_disjoint_and_masked` — `curriculum`
- `tests/test_regressions.py::test_original_untouched_and_parameter_count` — `seed/neral20m.pt`

Kein grüner Gesamtlauf behauptet. Die Umgebung hatte zusätzlich eine externe
pytest-Plugin-Deprecation-Warnung; der erste Vollvergleich meldete außerdem eine
bestehende Escape-Syntaxwarnung in `benchmark_gsm8k50.py`.

Umgebung: Python 3.12, torch 2.8.0+cpu, Flask 3.1.3, NumPy 2.3.5,
pytest 9.2.0, tokenizers 0.22.1, waitress 3.0.2. Projekt-Pins wurden nicht auf diese
Testumgebung umgestellt. Browser: Playwright/Chromium headless unter Linux.
Abhängigkeiten und Browser wurden installiert, **keine Modell-/Tokenizerassets**.

Initiale Versuche mit globalem `HELIOS_PREPARATION_MODE=1` und ohne Socketrechte
waren Umgebungsfehler; der dokumentierte Vergleich lief ohne diese Teststörung
und mit Socketrechten. Der echte Browserstart scheiterte zunächst an fehlendem
`waitress`, danach bestand der wiederholte Lauf. Ältere Zwischenprotokolle sind
keine Abnahme des aktuellen Stands.

## Nicht erfüllbare reale Checkpoint-Abnahme

Der Nutzer lieferte den Repository-/PR-Link und ausschließlich den bereits
bekannten **Tokenizer-Hash**, keine Dateien, keinen Checkpoint-Pfad und keinen
Checkpoint-Hash. Der Auftragsteil „mit dem erhaltenen Checkpoint testen“ bleibt
**offen**. Siehe `real-checkpoint-status.json`.

Die 14 Adaptertests verwenden ein winziges synthetisches Modell bzw. gezielte
Fehlerfixtures; der positive `/api/model`-Test ersetzt ausschließlich das Laden
der Dateien. Keiner beweist erfolgreiche Deserialisierung oder Kompatibilität
des echten 349.825.152-Parameter-Checkpoints. Auch die Browserläufe führen kein
350M-Modell aus. Kein Qualitäts-/Geschwindigkeits-/RAM-/GPU-Nachweis, keine
Windows-Abnahme in dieser Umgebung, keine Modellfreigabe.

Vorbereitet sind der strikt lokale CPU-Adapter mit gesperrtem BPE-/Konfigurations-
Vertrag, ausdrücklichem Laden, Herkunft und Hash sowie ein expliziter Prüf-Befehl
für echte Dateien. Bedienung, genaue Ladebedingungen, Abbruchgrenzen und Schritte
zur offenen Abnahme: [`docs/ARCTUS350_FLASK_OPTIONAL.md`](../../docs/ARCTUS350_FLASK_OPTIONAL.md).
