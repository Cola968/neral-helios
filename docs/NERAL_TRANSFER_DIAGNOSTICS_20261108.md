# NERAL: Fehlerdiagnose und Transfer-Sperre (08.11.2026)

## Was diese Änderung tatsächlich leistet

- Das bisherige dev72-Ergebnis bleibt **verbraucht**. Der neue, nur manuell auf `main` auslösbare Runner-Job liest ausschließlich die privaten Originalantworten aus `sixarm-parity-42470178641-1`. Er validiert die ursprüngliche sechsarmige Manifest- und Antwortstruktur und erzeugt nur aggregierte Zähler. **Kein Training, keine neuen Gewichte, keine GPU-Inferenz, keine automatische Freigabe.**
- Bei identischem Zero-/Three-Shot-Protokoll zählt er getrennt: richtige Zahl, auswertbare aber falsche Zahl, nicht eindeutig auswertbare Antwort, unvollständige Antwort. Zusätzlich werden die gepaarten Gewinner/Verlierer und kleine, explizit diagnostische Familienaggregate ausgewiesen.
- Ein allgemeiner `pilot_gate` akzeptiert erst nach unabhängiger Bewertung **164 Retention- und 64 Transferfälle** (Format: JSONL, je `id, family, baseline_correct, candidate_correct`), verweigert Dopplungen und fehlende Daten. Das sind die im noch offenen PR #366 vorgeschlagenen Split-Größen; diese Änderung erzeugt selbst keine neuen Aufgaben und behauptet keine Unabhängigkeit.
- Konservative, ausdrücklich **experimentelle** Hürden für eine *manuelle Review* sind: Transfer mindestens +6 netto von 64, exakter gepaarter McNemar-Wert ≤ 0,05, Retention nicht schlechter als −2 von 164 und pro Retention-Familie nicht schlechter als −2. Diese Schwellen sind eine lokale Entscheidungsregel, **kein bewiesener Schutz gegen Catastrophic Forgetting**.

## Welche Schlüsse weiterhin verboten sind

Ein passendes Ergebnis kann ausschließlich `eligible_for_review=true` liefern. Die Ausgabe setzt **immer** `external_freshness_verified=false`, `grader_independence_verified=false`, `training_authorized=false`, `release_validated=false` und `adapter_promoted=false`. Erst unabhängige Quellenprüfung, End-Holdout, Identitäts-/Decoderprüfung und menschliche Entscheidung können über spätere Freigaben informieren.

Die 72 alten Aufgaben sind keine unabhängige Generalisierungsprüfung. `wrong_readable` belegt eine falsche ausgelesene ganze Zahl, aber allein noch keinen mathematischen Ursachenmechanismus. Die 24 Python-Fälle messen Code-Tracing, keine Codegenerierung.

## Ausführung

CPU-Test in CI (Ubuntu, Python 3.12):

```powershell
python -B -m unittest discover -s tests -p test_neral_transfer_diagnostics.py -v
```

Nach Merge: GitHub Actions → `NERAL regression error audit and transfer pilot (CPU only)` → `Run workflow`, Branch `main`, Eingabe exakt `RUN_NERAL_ERROR_AUDIT_READONLY`. Die einzige private Arbeit ist eine kurze CPU-Auswertung auf dem lokalen Runner, anschließend `neral-dev72-sixarm-errors-aggregate-only` als Artefakt. Nicht auf dem Pull-Request-Branch auf Self-Hosted-Runnern ausführen.

Beispiel für künftige extern geprüfte Split-Dateien:

```json
{"id":"ret_001","family":"arithmetic_transform","baseline_correct":true,"candidate_correct":true}
```

```powershell
python -B -m scripts.neral_transfer_diagnostics --mode pilot_gate --retention retention.jsonl --transfer transfer.jsonl --out new_review_only.json
```

Die JSONL-Dateien dürfen niemals aus den historischen dev72-Ergebnissen oder aus dem Training erzeugt werden. Ein wiederverwendeter Test macht keine Behauptung über OOD-Leistung zulässig.
