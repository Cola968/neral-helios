# NERAL v6: kompositioneller Transfer, isolierter Kandidat

Stand 07.11.2026. **Noch kein v6-Modell und keine Trainingsfreigabe.** Dieser
Schritt erzeugt einen neuen, deterministischen und rechnerisch prüfbaren
SFT-Kandidaten für unbekannte Kombinationen von Rechenschritten.

## Warum

v5 erzielte 91/320 gegenüber 32/320 bei vertrauten Aufgabenfamilien, aber
13/120 gegenüber 13/120 bei einer anders formulierten synthetischen
Mathematikprobe. Das deutet auf eine ausgeprägte Formatanpassung bei
schwächerem rechnerischen Transfer hin. Es ist noch keine isolierte
Ursachenanalyse und keine unabhängige Leistungsbewertung.

## Neuer Kandidat

`scripts/neral_v6_compositional.py` generiert **800 Originalaufgaben**:
600 Training, 100 Validation, 100 getrennte synthetische Holdout-Aufgaben.

Die fünf Familien kombinieren jeweils mehrere Operationen:

- Prozentfolge mit dazwischenliegender fester Gebühr (Reihenfolge ist relevant)
- Treibstoffmasse aus zwei Raten, zwei Abschnitten, Reserve und Bodenverbrauch
- Zeitzonen samt Reisezeit, Zwischenstopp und Tageswechsel
- Lagerbestand mit Packungsgröße, Lieferung, Rückgabe und Ausschuss
- Gewichtete Mischkonzentration mit Verdünnung

Werte, Operationsfolgen, Aufgaben- und Antwortformate sind deterministisch,
Antworten werden rechnerisch mit `Fraction` und kaufmännischer Cent-Rundung
erzeugt. Train/Val/Holdout verwenden getrennte Seeds und Formulierungen.
Ein Teil der Aufgaben enthält `CHECK` und `FINAL_ANSWER`, sonst nur die
Endantwort. Diese synthetischen Varianten sind **noch keine vollständig
neuen Aufgabentypen und kein unabhängiger externer Benchmark**.

## Erstellen und gegen Manipulation prüfen

```powershell
python -B -m scripts.neral_v6_compositional --out C:\neral-v6-candidate
python -B -m scripts.neral_v6_export --source C:\neral-v6-candidate --out C:\neral-v6-trainer-export
python -m unittest discover -s tests -p test_neral_v6_compositional.py -v
python -m unittest discover -s tests -p test_neral_v6_export.py -v
```

Das Exportskript rekonstruiert das vollständige Originalmaterial und
verifiziert die Dateien Byte für Byte. Exportiert werden **nur train/validation**
ins bisherige QLoRA-JSONL-Format und dazu die Prüfmanifeste.
`holdout.jsonl` bleibt ausschließlich im Kandidatenordner und darf nie
für Training, DPO-Paarbildung oder Prompt-Auswahl benutzt werden.

Das Manifest enthält `training_allowed=false`,
`human_review_approved=false`, `tokenizer_verified=false` und
`training_performed=false`. Trainierbare Modelle und neue Gewichte werden
hier weder geladen noch erstellt.

## Tatsächlicher Weg zu besserer Generalisierung

1. Den laufenden 400-Fälle-EXT-Benchmark abschließen. Für die transferrelevanten
   numerischen A/B-Fälle nicht nur Qwen-Basis, sondern **Qwen mit Few-Shot**
   gegen NERAL v5 vergleichen. Python-MBPP ohne Sandbox bleibt unbewertet.
2. Fehler in *falsche Lösung*, *falsches Format*, *unvollständig* und
   *keine eindeutige Endantwort* trennen. **Keine** EXT-/OOD-Prompts, Antworten
   oder Variationen derselben Aufgaben ins Training übertragen.
3. Die oben erzeugten neuen Trainingsfamilien mit weiteren Originalaufgaben
   und kontrolliertem Coding-Korpus mischen. Vor jeder GPU-Runde:
   menschlicher Qualitätsreview, SHA- und Überlappungsprüfung, echter lokaler
   Qwen-Tokenizer und Completion-Mask-Preflight. Kein stiller Merge in v5.
4. Erst anschließend einen separaten experimentellen v6-QLoRA-Adapter
   trainieren, mit **einem** bewusst geänderten Faktor im Vergleich zu v5.
   Keine automatische Promotion nach einem guten Teilbenchmark.
5. Auf vor Trainingsbeginn gesperrten Testfällen die paarweisen
   Verbesserungen *und Regressionen* messen. Zusätzlich einen unabhängigen
   Coding-Benchmark mit Sandbox und eine andere echte Mathematikquelle
   prüfen. Aufgabenfamilien-, Sprach- und Schwierigkeitsmix berichten.

**Erfolgskriterium:** Ein stabiler, reproduzierbarer Vorsprung gegenüber
Qwen+Few-Shot auf noch nicht genutzten Aufgaben, ohne deutliche Regressionen
in bisherigen Stärken. Zielgrößen wie +5 oder +10 Prozentpunkte sind
Planungsziele, keine Vorhersagen. Ein positiver Score auf einem synthetischen
Holdout allein reicht nicht.

## Inferenzoptimierung ist getrennt

Batching, neue Tokenlimits und Speculative Decoding können die Auswertung
beschleunigen, **nicht die trainierten Fähigkeiten verbessern**.
Batchgrößen 1/2/4 zunächst mit VRAM- und Zeitmessungen prüfen. Niemals
Antworten bei 64/96 Tokens als vollständig werten, wenn sie abgeschnitten
wurden. Bei Code nicht an Zeilenumbrüchen stoppen. Assisted Decoding ist
nicht automatisch mit Batching kombinierbar und erfordert passende Modelle.
