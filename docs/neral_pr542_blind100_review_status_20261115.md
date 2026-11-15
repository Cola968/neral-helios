# NERAL PR #542 — Prospective100: Stand der Beweissicherung (15.11.2026)

**Dies ist kein Release-Schein und keine neue GPU-Messung.**

## Unveränderliche Ausgangslage

- Getesteter PR-Quellcode: `737413f03b347676108d507ff3fe7693975dd269`.
- Historischer Generator: `8c9c58b4cdc7ffda1fb3392a6e6277b59fae203f`.
- Autorisierter LF-Korpus-SHA-256: `74cf83ce3e9b3644b306556438a3e51373589ffdaf254dc84abb74e08fd04258`.
- Ursprünglicher, nicht lokalisierbarer Hash `3feca67d...b9a7b280d` wurde **vor** dem GPU-Lauf transparent abgelöst.
- GPU-Run: [#43709097701](https://github.com/Cola968/neral-helios/actions/runs/43709097701), erfolgreich. Kein Merge.
- Striktes Originalergebnis: Qwen 2/100, NERAL 12/100; Netto +10; 0 **streng formatierte** Regressionen; 10/25 Z3-verifizierte Logikfreigaben. Diese Werte sind **keine unabhängig validierte semantische Regressionsmessung**.

## Bekannter Bewertungsfehler und Nachbewertung

Der ursprüngliche Scorer akzeptierte nur eine ganze Zahl beziehungsweise eine eng formatierte Final-Answer-Zeile. Das direkte Qwen bekam kein entsprechendes verpflichtendes Ausgabeformat. Plausibel richtige Antworten in natürlicher Sprache wurden daher möglicherweise als falsch gewertet. Die absoluten Punktzahlen und die 0-Regressions-Meldung können dadurch erheblich verzerrt sein.

Die Originalantworten lagen ausschließlich in `RUNNER_TEMP/.../private-results.jsonl`. Das auf GitHub hochgeladene Artefakt enthält **nur skalare Kennzahlen**.

Nur-Lese-CPU-Läufe:
- [#43712518143](https://github.com/Cola968/neral-helios/actions/runs/43712518143): 77 feste Dateipfade, keine Rohdatei.
- [#43716581326](https://github.com/Cola968/neral-helios/actions/runs/43716581326): 307.721 Einträge, keine Rohdatei; nach 150 Sekunden unvollständig (33 nicht lesbare Einträge).
- [#43717104485](https://github.com/Cola968/neral-helios/actions/runs/43717104485): Priorität korrigiert, 343.803 Einträge, keine Rohdatei; nach 150 Sekunden unvollständig (42 nicht lesbare Einträge).

**Nicht nachgewiesen:** vollständige Abwesenheit auf allen Laufwerken. **Nachgewiesen:** keine gültige Originaldatei in den untersuchten Bereichen; der originale Inferenzlauf kann derzeit nicht offline neu bewertet werden.

## Isolierte CPU-Arbeiten

- `scripts/neral_pr542_offline_regrade_saved_raw.py`: derselbe feste Abschlusszahl-Extraktor für beide Modellarme; benötigt vorhandene Originalantworten; keine neue Inferenz.
- `scripts/neral_pr542_recovery_readonly.py`: Namenssuche mit Grenze und expliziter Kennzeichnung unvollständiger Durchläufe; Unit-Test für korrekte Suchpriorität nach zuerst fehlgeschlagenem Test.
- `tests/test_neral_pr542_raw_recovery_cpu.py`: fünf synthetische CPU-Tests (auf Windows vier erfolgreich, einer mangels Symlink-Berechtigung übersprungen; nach Prioritätsfix in Cloud 5/5).
- `scripts/neral_pr542_release_admission_cpu.py`: sperrt die Release-Beurteilung bei fehlendem Original-Rohdatenreview, unklaren Antworten, Herkunftskonflikten oder nachgewiesenen Regressionen.
- `tests/test_neral_pr542_release_admission_cpu.py`: acht synthetische Tests erfolgreich; RED-Fehler für unvalidierte Typen und ungünstige Suchpriorität wurden vor den Fixes reproduziert.

**Aktueller CPU-Entscheid:** `admission=blocked`, Grund `raw_evidence_missing`; `automatic_merge_allowed=false`.

## Nicht in den 100 Aufgaben getestet

Der isolierte Modus `nl_schema_consensus=true` ist laut `neral_local_chat_api.py` (Commit `737413f0`) ausdrücklich inkompatibel mit `experimental_tir` beziehungsweise `dual_routing`. Der Python-Sandbox-Pfad wurde nicht mitgemessen. Eine hypothetische kombinierte Gesamtpunktzahl ist unbekannt.

## Bedingungen für den nächsten **neuen** prospektiven Benchmark

1. Neuen unabhängigen Korpus vor jeglicher Anpassung versiegeln und einmalig testen; den verbrauchten 100er-Satz nicht als neuen Blindtest ausgeben.
2. Identische, vorher festgelegte Antwortkonvention beziehungsweise unabhängige Auswertung für Qwen und NERAL; Ambiguitäten blind manuell adjudizieren.
3. Rohantworten ausschließlich lokal und dauerhaft außerhalb von `RUNNER_TEMP` gesichert aufbewahren, inklusive SHA256-Manifests. Erfolg erst bestätigen, nachdem gespeicherte Originaldatei wieder lesbar und validiert ist.
4. QLoRA+Z3 und Sandbox+Z3 als getrennte Ablationsarme und optional kombinierte Route testen; vorab CPU-Regressions- und Provenienztests.
5. Kein `main`-Merge oder Release allein aufgrund der formal grünen #43709097701-Actions.

Diese Dokumentation enthält bewusst **keine** Aufgaben, Goldwerte oder Modell-Rohantworten.
