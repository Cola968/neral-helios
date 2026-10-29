# ARCTUS350 – Präregistrierter A/B-Forschungsvertrag v1

**Status 29.10.2026:** CPU-Code fertig und durch echte Github-Actions-Tests überprüft. **Weder trainiert noch Arctus-Gewichte verbessert.** Ergänzung zu [Research Issue #179](https://github.com/Cola968/neral-helios/issues/179), integrativ auf PR #186 und dem exakten Statistik-Gate PR #187.

## Warum dieser Baustein erforderlich ist

In einem 350M-Modell mit begrenztem Trainingsbudget zählt jeder Experimentdurchlauf. Wenn man gleichzeitig Lernrate, Daten und Rechenmenge verändert, ist nachher nicht bekannt, *warum* eine Fähigkeit besser/schlechter geworden ist. Nachträgliches Auswählen vieler zufälliger kleiner DEV-Gewinne überfittet die Entwicklungsprüfung. Eine korrekte **vorher eingefrorene** Fragestellung ist billiger als weitere ungezielte SFT-Läufe.

Wissenschaftlicher Anknüpfungspunkt: [SmolLM2 Paper, Abschnitt 3.1 und 6](https://arxiv.org/html/2502.02737). Dort wurden vergleichbare Daten-Ablationen bei gleicher Modellkonfiguration, Hyperparametern und Tokenmenge durchgeführt. SmolLM2-360M erhielt ca. vier Billionen Tokens, **kein** kurzfristig erreichbares oder kostenfreies Trainingsziel für Arctus350. Anders als ein leistungsfähiges fremdes Modell behalten wir die eigenen nativen Arctus-Gewichte.

## Was `arctus350_ablation_contract.py` tatsächlich leistet

- Akzeptiert nur den eigenen Modelltyp `neral-arctus350` und die Originalarchitektur `frontier350-v1`.
- Pinnt formal den ursprünglichen Checkpoint-SHA-256, den **Original-16K-BPE-Tokenizer**, Trainings- und VAL-Manifest-Digests, Benchmark-IDs, Seed, exakt gleiche geplante Optimizer-Schritte, Sequenzlänge, Mikrobatch, Akkumulation und geplante Assistant-Loss-Tokenobergrenze.
- Genau **ein** veränderter Trainingsparameter zwischen Baseline und Kandidat: `peak_lr`, `eos_weight` oder `sampling_mode`.
- Verweigert unbekannte Eingabefelder, unvollständige/duplizierte JSON-Felder, NaN/Infinity, negative Budgets, fremde Modellarchitekturen, beliebige weitere Vergleichsrunden und `evaluation_split=test` bzw. Zugriff auf geschützte TEST/NIS2.
- Erzeugt einen deterministischen SHA-256-Fingerprint des vollständigen Vorhabens und die eindeutige Kennzeichnung `FROZEN_PLAN_NOT_RUN`; niemals einen Modellscore oder eine Trainingsfreigabe.
- CLI-Ausgabe wird **nicht überschrieben**. Keine Downloads, Netzwerkaufrufe, Torch-Abhängigkeiten, GPU-Zugriffe oder geplanten Hintergrundarbeiten.

## Anwendung – erst nach lokaler Verfügbarkeit echter Ausgangsdaten

Die Input-JSON-Datei enthält:

| Bereich | Erforderlich |
|---|---|
| `model_family` | `neral-arctus350` |
| `architecture` | `frontier350-v1` |
| `source_checkpoint_sha256` | Aus **lokaler Originaldatei** neu gemessene Checkpoint-SHA |
| `tokenizer_sha256` | Gelockter Originaltokenizer; **keine** andere Tokenisierung |
| `train_manifest_sha256`, `val_manifest_sha256`, `benchmark_sha256` | Vom tatsächlich geprüften unveränderten Korpus bzw. der DEV-Suite |
| `evaluation_split`, `protected_test_access` | Nur `val` und `false` |
| `seed`, `budget` | Eindeutiger Seed, **gleiches** Arbeitspensum beider Arme |
| `baseline`, `candidate` | Jeweils `peak_lr`, `eos_weight`, `sampling_mode`, genau **eine** Änderung |
| `primary_metric` | Vorab `heldout_exact_match` |
| `secondary_metrics`, `intent` | Explizite zusätzliche Metriken und begründete Forschungshypothese |

Beispielaufruf **rein CPU, noch kein Training**:

```bash
python -m frontier350.arctus350_ablation_contract \
  --manifest locally_verified_ablation_plan.json \
  --out frozen_arctus_study.json
```

Die komplette strenge Eingabeform ist über `tests/test_arctus350_ablation_contract.py::fixture` als *synthetisches* Regressionsbeispiel nachvollziehbar. Die SHA-Zeichenfolgen dort sind **absichtlich nicht echt** und dürfen nicht für Trainingsfreigaben eingesetzt werden.

## Empfohlener erster späterer Test auf einer realen RTX 4070

**Hypothese:** `sampling_mode=compact` reduziert Padding auf dem realen Korpus gegenüber `bucketed`, ohne dass held-out Genauigkeit oder EOS sinken. Basis ist die echte vorhandene SFT-Pipeline `frontier350/train_chat350.py`, die den optionalen Compact-Modus bereits unterstützt (PR #181/#186). Aber: Auch bei identischer Schrittzahl können die Reihenfolge, tokengewichteten Updates, **tatsächlichen** Assistant-Zieltoken, VRAM-Zeiten und FLOPs abweichen. Erst echte Run-Metriken zeigen, ob die Methode schnellere, günstigere Modellfortschritte erlaubt.

**Gates vor GPU-Start:** Ausgangscheckpoint und BPE read-only verifiziert; alle Daten-/Suite-Hashes auf realen lokalen Dateien gemessen; Split-Leak-Audit (#182), Supervision-Mask-Audit (#183), gepaarte Native-Evidenz (#180), exakter Statistik-Gate (#187). Fehlende Messungen bleiben `NOT_MEASURED`; Test/NIS2 versiegelt.

**Grenzen:** Das Tool beweist *nur* die formale Prüfbarkeit des Vertrags, weder die Existenz noch die Echtheit der genannten Dateien. Es kann echte Korpusqualität, Leaks, gleiche tatsächliche Tokenmengen, unabhängige Fallauswahl, Rohmodellleistung oder den Modellfortschritt **nicht** bestätigen. Von `FROZEN_PLAN_NOT_RUN` geht keinerlei automatischer Trainings-/Veröffentlichungsauftrag aus.

**Kosten-/Freigabepolicy:** 0 € zusätzliche direkte Ausgaben bevorzugt, absolute Obergrenze 25 € ohne Strom. Kein kostenpflichtiger Hugging-Face-Job, kein Auto-Merge, keine privaten Gewichte oder Korpora in die Cloud und **kein RTX-4070-Start** ohne neue ausdrückliche Nutzerfreigabe.
