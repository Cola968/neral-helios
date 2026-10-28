# Arctus350: schreibgeschützter Struktur-/Template-Audit

## Warum?

`chat350_data.validate_prompt_holdouts` erkennt bereits **exakt normalisierte erste
Nutzerfragen** über `train`, `val`, `test` hinweg, aber nicht jede Ziffernvariation
oder Wiederholung in späteren Nutzerturns. Das hier ist eine **zusätzliche,
heuristische Review-Prüfung**, keine Änderung am bestehenden Trainings-Gate.

## Ausführung (nur CPU, kein PyTorch, keine Checkpoint-Dateien)

```bash
python -m frontier350.chat350_template_leakage_audit \
  --root /PFAD/ZU/LOKALEN/NERALTALK350_DATEN \
  --out /PFAD/ZU/NEUEM/REPORT.json
python -m unittest tests.test_chat350_template_leakage_audit -v
```

`--root` erwartet vorhandene, lokal verwaltete `train.jsonl`, `val.jsonl`,
`test.jsonl` mit dem bestehenden NeralTalk-Prompt-/Message-Schema. Report-Datei
muss neu sein. Es wird nichts heruntergeladen, übertragen oder im Datenbestand
verändert; es werden nur aggregierte Zählungen veröffentlicht. Lokale Daten
bleiben lokal. Kein GitHub-Action-Lauf führt diesen Audit auf realen Nutzerdaten
aus; CI prüft ausschließlich synthetische Daten.

## Metriken und Interpretation

- `exact_pairs`: Anzahl von Paaren identischer normalisierter Nutzerturns in
  **verschiedenen** Splits, einschließlich späterer Turns; jede Kombination von
  Vorkommen zählt als Paar.
- `numeric_template_only_pairs`: Split-übergreifende Paare *verschiedener*
  normalisierter Fragen, die erst nach Ersetzen von Zahlenfolgen durch `<num>`
  dieselbe Vorlage ergeben. Ein exakter Treffer wird nicht doppelt gezählt.
- `later_turn_*_pairs`: Teilmenge der jeweiligen Paare, bei denen mindestens
  eine Frage aus einem User-Turn nach dem ersten stammt.
- `status=review_required` fordert eine *manuelle Prüfung* möglicher
  Überschneidungen. `no_structural_overlap_detected` besagt **nicht**,
  dass Daten semantisch unabhängig sind.

Die Normalisierung verwendet Unicode NFKC, `casefold`, Whitespace-Zusammenfassung
und für die Template-Heuristik eine Unicode-Ziffernerkennung. Ziffernänderungen
können die Semantik ändern (z. B. eine Matheaufgabe!); eine Übereinstimmung
beweist keine Kontamination. Reine Umformulierungen, Paraphrasen, gemeinsame
Antworttexte und versteckte externe Holdouts werden nicht erfasst.

## Bezug zum Gemini-Bericht (28.10.2026)

Der Bericht verwendet den historischen Code-Commit `bf9f638f...`. Im aktuellen
Review-Branch #140 wurden segmentierte BPE-Generierung, BOS/PAD/SEP-Maskierung,
restricted Checkpoint-Laden und Best-Checkpoint-Auswertung schon korrigiert.
`--precision-mode fp32-master` ist ebenfalls vorbereitet, aber der tatsächliche
VRAM-Bedarf und die Modellverbesserung sind **NOT_MEASURED**. Die Behauptung,
dass der echte 350M-SFT-Checkpoint nur bestimmte Layer aktualisiert hat,
benötigt den *read-only* Vergleich der Originalgewichte; synthetische BF16-
Rundungsbeispiele genügen dafür nicht. CPU-Vollgenerierung eines 350M-Modells
über viele Fälle ist nicht seriös auf wenige Sekunden festzulegen.

**Keine Modellgewichte, Trainingseinstellungen, Datasets, KI-Fähigkeitsscores,
Freigabeentscheidung oder bestehende Branches werden durch diesen Audit geändert.**
