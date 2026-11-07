# NERAL-EXT vom 7. November 2026: diagnostisches Ergebnis

Der [Lauf 42395097203](https://github.com/Cola968/neral-helios/actions/runs/42395097203)
endete um **19:37 Uhr Europe/Berlin** erfolgreich. Gemessen wurden drei
sequenzielle Inferenzläufe, kein Training. Ausgewertet wird ausschließlich das
öffentlich hochgeladene
[Scalar-Artefakt 13585524327](https://github.com/Cola968/neral-helios/actions/runs/42395097203/artifacts/13585524327).
Private Aufgaben oder Modellantworten werden hier nicht veröffentlicht.

## Identität der Messung

- Ausgeführter Code: `2f14fcfe10ed719951d786ea778100d77ba0faf6`.
- Foundation: `c03e6d358207e414f1eca0bb1891e29f1db0e242`.
- Adapter-SHA-256: `2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5`.
- Datensatz-SHA-256: `47258a911ce0beb96635a34137a634556f364457487fb18a160a892302124cb1`.
- ZIP-SHA-256 laut GitHub: `ebd8469ae2c7c47ebab551c27d7b708e6fe691b835d23a76aa36099dc2be9759`.

## Was die Zähler tatsächlich zeigen

| Zähler im ursprünglichen Bericht | Qwen | Qwen mit Beispielen | NERAL v5 |
| --- | ---: | ---: | ---: |
| Semantisch korrekt, ursprünglicher Gesamtzähler | 59/400 | 84/400 | 91/400 |
| Strikt korrektes Format und Ergebnis | 58/400 | 81/400 | 90/400 |
| Gültiges Antwortformat | 94/400 | 194/400 | 235/400 |
| Numerisch bewertbare Transferaufgaben aus A+B | 46/250 | 66/250 | 69/250 |

Die letzte Zeile rekonstruiert die A- und B-Treffer aus dem unveränderten
Artefakt; die 50 funktional **unbewerteten** Python-Fälle im Stratum A zählen
nicht zum numerischen Nenner. Die übrigen 100 Fälle aus C und D sind getrennte
Format-/Robustheitsdiagnosen. Der Gesamtzähler ist deshalb keine
domänenübergreifende Coding-/Reasoning-Erfolgsquote.

Der numerische Vorsprung gegenüber der stärkeren Few-Shot-Baseline beträgt
**3/250 = 1,2 Prozentpunkte**. Es gibt 12 Verbesserungen und 9 Regressionen.
Der gewöhnliche exakte zweiseitige McNemar ergibt dafür explorativ
`p=0.6636238098144531`. Für den ursprünglichen Gesamtzähler sind es 21
Verbesserungen und 14 Regressionen (`p=0.31050465907901525`). Diese Werte
belegen keinen zuverlässigen allgemeinen Leistungsvorsprung.

## Einschränkungen

- Der Datensatz enthält öffentliche Aufgaben und synthetische Familien;
  Unbekanntheit gegenüber Qwens Vortraining ist nicht belegt.
- Der aktuelle Datensatzgenerator kann identische Aufgaben mehrfach erzeugen.
  Gewöhnliche fallweise Signifikanzwerte sind deshalb zusätzlich nur
  explorativ. Duplikatdiagnosen neuer Auswerter ändern den alten Datensatz
  und seine gespeicherten Antworten nicht.
- Weniger abgeschnittene Antworten und mehr gültige Formate sind echte
  Verhaltensunterschiede; daraus folgt kein ebenso großer Zuwachs an
  fachlicher Korrektheit.
- Der Lauf nutzte den früheren Auswerter. Neue Felder aus späteren Commits
  sind hier als Rekonstruktion benannt, nicht als damals ausgeführte Prüfung.
- Keine Adapter-Promotion und kein bestandenes Release-Gate.

## Konsequenz für die Entwicklung

Das Ergebnis rechtfertigt weder ein pauschales Weitertrainieren derselben
Schablonen noch ein Versprechen von +10 Prozentpunkten. Der nächste
Trainingskandidat muss andere Lösungsstrukturen und überprüfbare
Zwischenschritte enthalten. Er wird ausschließlich aus neuen Trainingsaufgaben
gebaut. EXT-/Release-Aufgaben und ihre Antworten bleiben ausgeschlossen.
Erst Datenreview, Tokenizer-/Completion-Mask-Preflight und getrennte
Kontaminationsprüfungen erlauben einen anschließenden experimentellen Lauf.

Code-Agenten benötigen außerdem frische Evidenz: Ein bestandener Test darf
nicht weitergelten, nachdem die geprüften Quelldateien geändert wurden oder
eine spätere Wiederholungsprüfung fehlgeschlagen ist.
