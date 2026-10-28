# Arctus 350M — unabhängiger Claude-/Gemini-Review, abgeglichen 28.10.2026

**Scope:** Code-Lesereview; keine reale 2B-Checkpoint-Inferenz, kein Training, keine bezahlten Aufrufe. Von den beiden Fremdreviews zitierte Zeilen gehören zum historischen Stand `bf9f638f6c45311c175c3f154d7c62a7952b8465`; dieser PR enthält inzwischen neue Fixes, sodass Zeilennummern abweichen.

## Entscheidungen nach Gegenprüfung

| Review-Hinweis | Einstufung | Aktion |
|---|---|---|
| `train_chat350.py`: BF16-Modellparameter plus AdamW bei kleiner Lernrate | **Nachgewiesene Konstruktion / Qualitätseffekt ungemessen** | Separaten Precision-/VRAM-Pilot auf RTX 4070 vorbereiten. Gemini behauptet feste 84,5 % eingefrorene Gewichte und sichere 7 GiB VRAM; beides ohne echte Messung nicht garantieren. Keine ungeprüfte Hauptgewichts-/Optimizer-Umstellung. |
| `eval_chat350.py`: Ganztext-BPE vs. segmentierte SFT-Prompts | **Nachgewiesener Paritätsbruch** | **Behoben** mit `chat350_eval_contract.case_prefix_ids`. Alle 12 Dev- und 30 Capability-Prefix-Formate passen zur strikten Parsergrammatik. Der Check ersetzt keine Inferenzmessung. |
| `eval_chat350.py`: BOS/PAD/SEP konnten generiert werden | **Nachgewiesene Diskrepanz zur Chat-Runtime** | **Behoben**: Greedy-Logits vor argmax maskieren; EOS bleibt erlaubt. Roh-token-IDs bleiben im Bericht. |
| Erfundene `\nUser:`-Fortsetzung kann Integer-/Exact-Scoring verfälschen | **Plausibles Messproblem** | **Behoben**: angezeigten/scorbaren Antworttext vor dem neuen User-Turn abschneiden; ursprüngliche Tokenfolge bleibt für EOS/Repetitionsmetriken erhalten. Evaluation `evaluation_contract=segment_bpe_mask_specials_v1` markiert. Die alte Score-Suite muss mit beiden Checkpoints erneut ausgewertet werden. |
| `eval_chat350_slices.py`: `torch.load(weights_only=False)` | **Echter Deserialisierungs-Risikopfad** | **Behoben**: restricted `secure_checkpoint350.load_checkpoint` verwenden. |
| `compare_chat350_slices.py`: nur `overall` oder ausgefallene lange Beispiele können `slice_no_regression=True` liefern | **Nachgewiesene unzureichende Abdeckung** | **Behoben**: mindestens eine auswertbare Untergruppe und keine übersprungenen langen Beispiele erforderlich. Das Gate lehnt bewusst zu kleine/invollständige Berichte ab. |
| Frühere `score_chat350_capabilities.py` überschreibt doppelte IDs, toleriert negatives Kategorien-Delta | **Nachgewiesen** | Bereits im selben Draft-PR #140 zuvor behoben, zusammen mit Frozen-Suite-SHA und endlichen Prozentwerten. |
| Bei jedem vorherigen Assistant-Turn fehle EOS, deshalb zwingend fehlerhaft | **Format-/Trainingsdesign-Frage, nicht automatisch Bug** | Gesprächspfade werden als ein einziges konkateniertes Transkript trainiert; das EOS nur nach dem letzten Assistant ist eine mögliche Turn-Semantik. Eine Umstellung auf EOS *nach jedem Turn* ändert Label-Ziele, Inferenztemplate und DPO-Daten gleichzeitig. Nicht ohne separaten kontrollierten Vergleich/Kompatibilitätsprüfung. |
| `chat350_knowledge.py` indexiere immer noch Text jenseits der 2400-Zeichen-Sichtgrenze | **Veralteter Befund** | Bereits am aktuellen Branch korrigiert: `visible_excerpt = excerpt[:2400]` und `tokens = _terms(visible_excerpt)`. Ein vorhandener Regressionstest deckt das ab. |
| Lockerung der strikten Trigramm-Schranke auf pauschale +5 Prozentpunkte | **Hypothese, schwächt Sicherheitsgate** | **Nicht übernommen.** Für faire Längenvergleiche zuerst EOS, Tokenlänge und geeignete Untergruppen dokumentieren; eine permissive Grenze muss gesondert begründet und getestet werden. |
| `best_chat.pt` wird nach answer-NLL/Teacher-Forced-EOS gewählt, `summary.json` kann letzten Step zeigen | **Nachgewiesener Auswahl-/Berichtskonflikt** | **Offen.** Vor Änderung getrennte Baseline- und Kandidatenmetrik-Semantik planen, Tests schreiben und einen Pilot auf bestehender Checkpoint-Linie auswerten. Eine einfache NLL-Verbesserung ist keine Release-Freigabe. |
| Batchweise SFT-NLL mit EOS-Weight ≠ globale Validation-NLL | **Nachgewiesene Ziel-/Metrik-Differenz** | **Offen.** Tokengewichtete Gradientenakkumulation nur per separatem A/B-Experiment einführen; ändern wir nicht blind zusammen mit FP32 und EOS, sonst ist keine Ursache isolierbar. |
| Rückgang 3/30 → 2/30 | **Ein einzelner Fall bei kleiner Stichprobe** | Kein statistisch gesicherter Fähigkeitsverlust. Strikte Release-Gates können dennoch bewusst Vorsicht verlangen. Vor weiterem Training den historischen Lauf mit gleichem korrigiertem Scorer neu messen. |

## Umsetzung in diesem PR

- `frontier350/chat350_eval_contract.py`: strikt kanonische User-/Assistant-Prefixe segmentiert tokenisieren; Spezialtoken-IDs validieren; gefälschte User-Turns aus scorbarem Text fernhalten.
- `frontier350/eval_chat350.py`: dieselbe historische Tokenisierungssemantik wie die SFT-Runtime; BOS/PAD/SEP maskiert; Version des Evaluationsvertrags in JSON-Bericht.
- `frontier350/compare_chat350.py`: keine unbemerkte Vermischung von alten und neuen Evaluationsverträgen.
- `frontier350/eval_chat350_slices.py`: restricted Checkpoint-Load.
- `frontier350/compare_chat350_slices.py`: Slice-Review benötigt echte auswertbare Untergruppen und vollständig berücksichtigte Zeilen.
- `tests/test_arctus350_gemini_review_regressions.py`: reine Python-Regressionsfälle ohne GPU.
- Unverändert: 350M-Foundation-Gewichte und Pretraining, SFT-Optimizer, OASST-/NeralTalk-Korpora, Benchmark-Testfragen, Ausgabe von `main`.

## Grenzen

Neue und frühere Checker-Tests sind **synthetische Logiktests**. Es gibt hier keine gemessene 2B-Checkpoint-Loss-Kurve, kein GPU-Diagnoseprotokoll, keinen gemessenen VRAM-Peak und keinen glaubwürdig aktualisierten NIS2-Wert. Keine Qualitätsverbesserung des Rohmodells behaupten. Vor den nächsten Trainingsjobs zuerst aktuelle 1B-SAFE- und 2B-Foundation-Identität, SFT-Historie, echte Antworten und eingefrorene Suite prüfen; dabei den neuen Evaluationsvertrag **für beide** zu vergleichenden Checkpoints verwenden.

**Budget:** Null direkte Ausgaben; 25 € insgesamt erlaubt, Haushaltsstrom ausgeschlossen. Die Veröffentlichung / Übernahme bleibt manuell.
