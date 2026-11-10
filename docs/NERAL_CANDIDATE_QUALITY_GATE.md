# NERAL: Kandidatenprüfung vor Gewichtsfreigabe

Status: experimentelles **Read-only-Tool** für Qwen2.5-Coder-7B-Instruct und einen separat erzeugten NERAL-LoRA-Adapter. Noch nicht mit Helios/Release-Automation verbunden. **Kein Training, kein Modellstart, keine Änderung an Gewichtedateien und kein nachgewiesener Fähigkeitsgewinn.**

## Ziel

Der bisherige 86-Beispiel-Trainingsdatensatz aus Draft #234 macht ein späteres QLoRA-Experiment möglich, beweist aber keine gelernte Verbesserung. Dieses Werkzeug verhindert, dass ein möglicher Adapter nur wegen einer guten Gesamtzahl als fertig eingestuft wird, obwohl vorher gelöste Aufgaben regressieren oder er aus einem anderen Modell stammt.

Das Werkzeug liest einen separat geprüften, unbekannten **FINAL-Holdout** und zwei extern bewertete Antwortinventare unter identischen Bedingungen. Es lehnt inkonsistente/fehlende Fälle, unvollständige Volltreffer, ungestützte Behauptungen und nicht passende Datei-Hashes ab.

## Aufruf

Ein unabhängiges FINAL-Inventar und die tatsächlich bewerteten Antworten müssen schon existieren. Die 100 bisherigen DEV-Aufgaben aus #228 sind **kein** neuer FINAL-Holdout und werden ausdrücklich abgelehnt.

python scripts/neral_candidate_quality_gate.py --final-manifest FINAL.json --baseline BASE.json --candidate ADAPTER.json --adapter-file adapter_model.safetensors

Es gibt nur zwei Entscheidungen:

- REJECT: Vergleich formal ungültig bzw. empirische Qualität unter dem strengen Anfangsgate. Bei formal falschen Daten Abbruch mit Exitcode 2.
- MANUAL_REVIEW_REQUIRED: empirisches, **noch nicht authentifiziertes** Vergleichsprotokoll erfüllt die Schwellen. Exitcode 0 ist **keine** Freigabe zur Aktivierung oder zum Merge.

Das CLI führt keine externen Befehle aus, liest keine Modelldaten ins neuronale Laufzeitsystem und schreibt keine Dateien.

## Exakte Eingaben

Das FINAL-Manifest enthält die Felder schema=1, benchmark_version, split=final, independent_review_assertion=true und tasks (100–500). Jede Aufgabe enthält id, category, prompt_sha256 und rubric_sha256. Erlaubte Kategorien: coding, math, reasoning, instruction, reliability; mindestens zehn Aufgaben je Kategorie. Aufgabe, Prompt-Hash und Rubrik-Hash müssen eindeutig sein. Ein umbenannter DEV-Datensatz ist damit **nicht** tatsächlich unabhängig.

Jeder Run enthält schema=1, role (baseline/candidate), base_model_id, base_revision (40 hex), base_snapshot_sha256, tokenizer_sha256, prompt_template_sha256, decoding_sha256, tools_sha256, benchmark_sha256, adapter_sha256 und cases. Die verglichenen Einstellungen müssen bytegleich sein. Baseline hat keinen Adapter; Kandidat muss den echten SHA-256 der vorliegenden lokalen Safetensors-Adapterdatei tragen. Datei darf kein Symlink und maximal 512 MiB groß sein. Für jeden Task müssen id, prompt_sha256, rubric_sha256, answer_sha256, status (full/partial/fail), complete (Boolean), unsupported_claims (Ganzzahl) und reviewer angegeben sein.

Die Scorer- und Reviewerangaben sind **nicht kryptografisch authentifiziert**. Externe Prüfung des Antworttexts, des tatsächlichen Modells, der Semantik und des bislang unbekannten Testinventars bleibt zwingend.

## Strenge Anfangsschwelle

- Mindestens fünf Prozentpunkte mehr vollständig gelöste Aufgaben und mindestens fünf zusätzliche Erfolge.
- Kein bisher vollständig gelöster Fall darf regressieren. Keine Kategorie darf schlechter werden.
- Keine Zunahme von ungestützten Aussagen oder nicht vollständig beendeten Antworten.
- Einseitiger exakter gepaarter McNemar-/Vorzeichentest mit p höchstens 0,05.

Dieser bewusst sehr konservative Gate-Entwurf kann nützliche Forschungsadapter ablehnen. Solche Kandidaten dürfen experimentell weiterentwickelt werden, aber nicht automatisch in NERAL freigegeben werden.

## Sicherheitsgrenzen

Der CLI ist ein Prüfwerkzeug, **keine technische globale Schreibsperre**. Andere Prozesse können weiterhin Dateien verändern; später muss das Gate in eine geprüfte, fail-closed Release-Pipeline integriert werden. Auch der behauptete unabhängige Review wird hier nicht verifiziert. Nicht wiederholt am finalen Holdout optimieren: Mehrfachtests und nachträgliche Auswahl verfälschen die Aussagekraft.

Die bisherige Freigabe für maximal drei QLoRA-Optimizerschritte gilt ausschließlich für einen einmaligen technischen Pilotlauf und nicht für fortgesetztes Training, autonome Änderungen oder automatische Aktivierung.

**10. November 2026:** Wenn keine echte A/B-Qualitätsverbesserung gemessen ist, bleibt die korrekte Aussage „kein gelernter Modellgewinn nachgewiesen“. NERAL kann dann nur die nachweisbaren System- und Werkzeugverbesserungen separat ausweisen.

Tests: python -m unittest discover -s tests -p test_neral_candidate_quality_gate.py -v (16 synthetische CPU-Vertragstests; keine Qwen-Inferenz).
