# NERAL Qwen3.5-9B — unabhängige 100er-Ablation und Dual-Routing (12.11.2026)

## Quellen und Prüfumfang

- Ursprünglicher gepaarter RTX-4070-Lauf: [#43242062250](https://github.com/Cola968/neral-helios/actions/runs/43242062250), [100 Fallpaare als JSON](https://github.com/Cola968/neral-helios/actions/runs/43242062250/artifacts/13956491402).
- Neuer GPU-Ablation-Lauf: [#43247170855](https://github.com/Cola968/neral-helios/actions/runs/43247170855), [vollständige Ablation-JSON und Hashes](https://github.com/Cola968/neral-helios/actions/runs/43247170855/artifacts/13958401906). CPU- und GPU-Job: erfolgreich, 14 fokussierte CPU-Tests.
- Exakter getesteter Code-Commit: `2d498d7fcdf241caf1d4491f1dfee7e8d30c80c6`; kein Merge nach `main` oder Produktumstellung.
- Unveränderter PR-#537-Guard: Git-Blob `b87c594ee54a20b2209b6bf176bc0aacfe256ce8`.
- Eingefrorene 100 Aufgaben SHA-256: `ecff6f8e88c2c0f6c1f6a7d2bc77f1c6f0f94dfa047f035f4d5157d458c02560`.
- Trainings-SHA-256: `2a6c2219f4c52365be8a3a60bd080877c5d3fecef37371024d8936e31b18ec79`.
- Beide Modell-Digests verifiziert: Kontrolle `ed6a74343db547257a9a78150d647f33fbf25db93d87b8e3d80273b116fa7bb9`; QLoRA `9c45488f99cc446821f2008f74a4320a1de58a2f5382a68db889eadc8660dda8`.
- Identische ursprüngliche Systemnachricht, Tokenlimit 2048, Kontext 4096, Temperatur 0, per-case Seed. **Die früheren 29/100 wurden aus den 100 gespeicherten Einzelresultaten übernommen, nicht in diesem Lauf neu generiert.** Modellantworten wurden ohne Training auf derselben RTX 4070 ausgewertet.

## Vier Arme auf denselben 100 Aufgaben

| Kategorie | Kontrolle, raw | QLoRA, raw (kein Werkzeug) | Bisheriges QLoRA+PR537 | QLoRA-Tools + Basis-Raw-Fallback |
|---|---:|---:|---:|---:|
| Mathematik | 11/25 | 8/25 | 8/25 | 11/25 |
| Logik | 0/25 | 1/25 | 1/25 | 0/25 |
| Python-Programmierung | 5/25 | 5/25 | 19/25 | 19/25 |
| Gemischt | 13/25 | 12/25 | 12/25 | 13/25 |
| **Gesamt** | **29/100** | **26/100** | **40/100** | **43/100** |

Gepaarte Veränderungen gegenüber den 29/100 der gespeicherten Kontrolle:

- Reines QLoRA: 2 Gewinne, **5 Verluste**, netto −3, exakter zweiseitiger McNemar **p=0,453125**. *Keine Evidenz einer allgemeinen Gewichtsverbesserung.*
- Bisheriges QLoRA+PR537: 16 Gewinne, 5 Verluste, netto +11, **40/100** (siehe ursprünglichen Lauf).
- **Dual-Routing:** 14 Gewinne, **0 Verluste**, netto +14, 43/100; nomineller exakter McNemar **p=0,0001220703125**. Achtung: Die Architektur wurde nach Einblick in die alten 100 Resultate gewählt; dies ist eine erneut gemessene Entwicklungsstichprobe, kein unabhängiger Blindtest.

In der neuen Dual-Routing-Messung wurden **15 unabhängig verifizierte Python-Sandbox-Resultate** übernommen, alle 15 korrekt (14 zusätzliche Treffer gegenüber der Kontrolle). In den übrigen **85 Fällen** kam die Antwort aus dem unveränderten untrainierten Raw-Qwen. Für alle 85 stimmte die neue Fallback-*Korrektheit* mit den 85 früheren Kontroll-Korrektheitswerten überein. Es ist keine Aussage, dass sämtliche Modellantworttexte byteidentisch waren.

Reines QLoRA erreichte bei Programmierung **5/25**, also **keine Verbesserung** gegenüber der Kontrolle. Für die 14 zusätzlichen Programmier-Treffer war die unabhängig verifizierte Ausführung des vorgegebenen Python-Programms entscheidend, nicht ein nachgewiesener Intelligenzgewinn des Adapters.

## Fünf ursprüngliche Regressionen und erneute Diagnosen

Die ursprünglichen Einzelfallprotokolle protokollierten den Korrektheitsstatus, **nicht** die falsche Zahl. Die folgende Zahlenspalte stammt deshalb aus dem jetzt erneut ausgeführten Raw-QLoRA-Lauf mit demselben Prompt/Decoder/Seed.

| Fall-ID | Aufgabe | Referenz | QLoRA-Raw, erneuter GPU-Lauf | Basis-Raw im neuen Dual-Routing |
|---|---|---:|---:|---:|
| mathematics-08 | Aus 10 Karten 5 ohne Reihenfolge wählen | 252 | **10** | **252** |
| mathematics-18 | Aus 12 Karten 3 ohne Reihenfolge wählen | 220 | **496** | **220** |
| mathematics-23 | Aus 12 Karten 5 ohne Reihenfolge wählen | 792 | **12** | **792** |
| mixed-05 | 34.000 m/s × 161 s | 5.474.000 | **5.274.000** | **5.474.000** |
| mixed-07 | kinetische Energie ½ × 34 × 30² | 15.300 | **1.530** | **15.300** |

Bei **allen fünf** wurde im neuen Lauf einmal eine modellbasierte Schema-Extraktion versucht, mit `invalid_input` verworfen, **kein Tool zur exakten Berechnung** ausgelöst (`tool_calls_triggered=0`) und auf Raw zurückgefallen. Die erste und dritte Kombinatorikantwort entsprechen nur der Anzahl der angebotenen Karten; bei mixed-05 liegt ein Multiplikationsfehler von 200.000, bei mixed-07 ein Faktor-10-Fehler vor. Der genaue interne Grund für den Output 496 bei mathematics-18 ist nicht nachweisbar. Das sind beobachtete Fehlerbilder, keine Behauptung über verborgene Modell-Rechenschritte.

## Dual-Routing — kritischer Implementierungsunterschied

Ein Hook nur im `choose_guarded_tir_reply(..., raw_respond=base)` aus PR #537 genügt nicht. Die Funktion `run_experimental_tir(..., respond=...)` führt gelegentlich **selbst** `base_fallback()` aus, das `respond(original_messages)` aufruft. Der Guard sieht dann `base_fallback_unverified` und übernimmt die bereits erzeugte Antwort.

Deshalb verwendet der geprüfte separate Evaluator einen zentralen Modell-Dispatcher:

- `respond(turns)`: Wenn `turns` strukturell **genau dem Originaldialog** entspricht, nur den untrainierten Kontroll-Tag benutzen. Alle veränderten TIR-Tool-Planungsdialoge nutzen QLoRA.
- `extract_constraints(turns)`: Modellextraktionen und maximal eine Reparatur immer über QLoRA.
- `choose_guarded_tir_reply(..., raw_respond=raw_control)`: Auch zusätzliche Guard-Fallbacks gehen an das Kontrollmodell.
- Verifizierte Antwort aus explizitem Nutzercode oder explizitem formalem Schema bleibt bestehen. Modellgenerierte, bloß solver-akzeptierte natürliche Sprachübersetzungen erhalten keinen Verifikationsstatus.
- Die Implementierung erzwingt für alle Raw-Fallbacks **genau einen Aufruf des untrainierten Kontrollmodells**; CPU-Tests decken internen Controller-Fallback, Guard-Fallback und verifizierten Tool-Pfad ab.

Diese Route ist derzeit **nur im separaten Prüfscript** `scripts/neral_generalization_100_ablation.py` implementiert, nicht als Standard der produktiven lokalen API geschaltet. Keine bestehenden Ollama-Modelle oder Gewichte wurden verändert.

## Grenzen und Empfehlung

Dies sind 100 öffentlich erzeugte, numerisch exakt bewertete Entwicklungsaufgaben. Der Vergleich wurde nach Betrachtung des ersten 100er-Laufs um Dual-Routing erweitert; deswegen nicht als unabhängige statistische Bestätigung auf unberührten Daten vermarkten. Die 29/100-Kontrolle wurde aus der ersten Messung übernommen. Die neuen 85 Dual-Raw-Fallbacks stimmen in der *Korrektheit*, nicht nachweislich im vollständigen Antworttext, mit der Kontrollmessung überein. Die Antwortformatierung war in allen 100 Raw-Aufgaben je Modell numerisch gültig, daher ist die geringe Trefferquote kein bloßer Freitext-Parsingfehler.

**Technische Entscheidung:** Das unveränderte Qwen für allgemeine Raw-Antworten beibehalten; QLoRA höchstens für spezialisierte Tool-/Schema-Vorschläge verwenden; unabhängig verifizierte Nutzer-Code-Ausführung priorisieren. PR #537 bleibt experimentell. Kein weiteres Training auf dem 100er-Korpus, keine Anpassung von Heuristiken an diese Fälle. Vor Produktionsfreigabe ein tatsächlich unabhängiges Blindtest-Gate und Latenz-/Robustheitsprüfungen durchführen.
