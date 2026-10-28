# Arctus350: echte All-Turn-Holdout-Sanierung (28.10.2026)

**Originale NeralTalk-v3-Daten wurden nicht verändert.** Dieser Branch ist experimentell / DRAFT und hat nichts nach `main` gemerged.

## Gemessene Ausgangslage

[Read-only Audit #40716889080](https://github.com/Cola968/neral-helios/actions/runs/40716889080): 11.775 Train-, 1.533 Val-, 943 Testdialoge. Der alte Gate in `chat350_data.validate_prompt_holdouts` vergleicht nur den **ersten** User-Turn. Der unabhängige Read-only Audit fand **14 cross-split-Paare identischer späterer User-Fragen**. Zusätzlich 231.669 **nicht-exakte** Zahlenvorlagen-Paare, die aus legitimen Rechenvarianten stammen können; **keine automatische Löschung** solcher Paare.

## Konkrete Änderungen

- `chat350_data.validate_prompt_holdouts` prüft **alle User-Turns** auf NFKC/casefold/Whitespace-normalisierte exakte Überschneidungen über train/val/test. Bei Treffer Abbruch vor SFT. Cross-split-Promptinhalt erscheint nicht im Fehlertext.
- `chat350_split_sanitizer` erstellt nur lokal einen **neuen** Datensatz. Test erhält höchste Priorität, Val die zweite, Training die dritte. Konkurrierende niedriger priorisierte Dialoge werden entfernt; Ursprung bleibt SHA-überprüfbar unverändert. Keine kosmetischen/numerischen Vorlagen werden automatisch verworfen.
- CPU-Tests erzwingen Fehlverhalten im RED-Status und verifizieren die Behandlung später User-Turns, Test-Priorität, Numerik und unveränderliche Quellen.

## Ausgeführt und belegt

- RED All-Turn-Gate: [#40717299331](https://github.com/Cola968/neral-helios/actions/runs/40717299331) schlug mit den neuen Tests am ursprünglichen First-Turn-Gate fehl; GREEN: [#40717402400](https://github.com/Cola968/neral-helios/actions/runs/40717402400).
- RED Sanitizer: [#40717590305](https://github.com/Cola968/neral-helios/actions/runs/40717590305); GREEN: [#40717669984](https://github.com/Cola968/neral-helios/actions/runs/40717669984).
- Echte lokal vorbereitete Daten: [#40717899967](https://github.com/Cola968/neral-helios/actions/runs/40717899967): **Train 11.775 → 11.769 (-6), Val 1.533 unverändert, Test 943 unverändert**.
- Original-Datensatzmanifest-SHA: `b9c525dc6a5a7cd2ff1a2f8d2e90ef7f137d3bfd36d73fb218f88d57a1f00676`.
- Neues abgeleitetes, ausschließlich lokales Datensatzmanifest-SHA: `fb52786aefe2c8ffce39e6c72039249a5c999403d231773141abd8291e599f31`.
- Der bereinigte Datensatz liegt auf dem **eigenen PC** im neuen `runs/neraltalk350-sanitized-allturn_40717899967`; GitHub erhält nur den aggregierten Report ohne Originaldialoge.

## Was diese Sanierung NICHT beweist

Sie belegt **keine semantische Unabhängigkeit**; numerisch veränderte Aufgaben sowie Paraphrasen, Quellen-/Lizenzfragen und unabhängige externe Holdouts erfordern weitere Prüfungen. Ein niedrigerer Val-Loss beweist keine reale Chatqualität. Keine Änderung bestehender Checkpoints, kein Modelltraining durch diesen Branch, keine NIS2-Bewertung, keine Promotion oder Releasefreigabe.
