# NERAL: Gleichungsdiagnose und ausgewogene Core-/Mathematik-Mischung

Stand: 6. November 2026. **Keine neuen Gewichte trainiert und keine Freigabe für die Produktion.**

## Belastbare Resultate

- 80 Mathematik-Entwicklungsaufgaben: Qwen 65/80, alter Rank-32-Adapter 67/80, Mathe-v2 Rank16 69/80. Kein unabhängiger Release-Holdout.
- 36 zusätzliche Entwicklungsaufgaben: alle drei 28/36, davon 16/24 Mathematik und 12/12 einfache exakte Text-Anweisungen. Diese prüfen weder Code noch das gesamte Systemverhalten.
- Acht getrennte lineare Gleichungen: v2 2/8. **8/8 waren gültige reine Zahlenantworten**. Falsches Ausgabeformat ist damit *nicht* der beobachtete Fehler. Der vermutete Nutzen expliziter Zwischenschritte muss separat getestet werden.

### Sechs gescheiterte Gleichungen aus dem ursprünglichen 80er-Lauf

| ID | Gleichung | Sollwert | Format |
|---|---|---:|---|
| novel-041 | 6x - 32 = 286 | 53 | Zahl, aber falsch |
| novel-043 | 2x + 118 = 134 | 8 | Zahl, aber falsch |
| novel-044 | 14x + 5 = 439 | 31 | Zahl, aber falsch |
| novel-045 | 9x + 20 = 191 | 19 | Zahl, aber falsch |
| novel-046 | 15x + 68 = 683 | 41 | Zahl, aber falsch |
| novel-048 | 13x + 96 = 642 | 42 | Zahl, aber falsch |

Die Fälle novel-042 (x=-12) und novel-047 (x=69) waren richtig. Die bereits gespeicherten 80er-Laufdaten enthalten nur die Korrektheits- und Formatflags sowie SHA der Antworten, **nicht** die sechs numerischen Fehlwerte. Diese dürfen nicht geraten werden. Ein expliziter Zweischritt-Promptvergleich mit denselben Modellgewichten wäre eine zusätzliche Diagnose, aber kein Training.

## Geprüfter lokaler Trainingskandidat (v3, noch nicht trainiert)

`scripts/neral_math_v3_balanced_pack.py` erzeugt deterministisch **384** Zeilen aus dem unveränderten lokalen SHA-gesicherten v2-Datensatz und 100 unabhängig nachgerechneten neuen Gleichungen:

| Bereich | Anzahl |
|---|---:|
| Eigene NERAL-Core-SFT | 86 |
| Historische Mathematik mit Erklärungen | 40 |
| Bereits geprüfte Gleichungen | 70 |
| Neue Gleichungen (75 Zahl, 25 auf Anweisung mit Umformungsschritt) | 100 |
| Andere Rechenfamilien | 88 |

Dies erhöht den Anteil der ursprünglichen 86 NERAL-Core-Beispiele von **11,6 %** im alten 744er-Pack auf **22,4 %** (nicht durch Duplikate). Der Gleichungsanteil steigt auf 170/384. Neue Gleichungen erlauben auch negative Koeffizienten und drei Formfamilien. Jede numerische Lösung wird aus dem angezeigten Term unabhängig mit rationaler Arithmetik überprüft; Mutations- und Reproduzierbarkeitstests müssen bestehen. Die alten Review- und Quellen-Hashes werden für jede übernommene Zeile erhalten; die Gesamtmanifestdatei muss vom unveränderten Original-QLoRA-Trainer erneut akzeptiert werden.

Der Workflow `neral-math-v3-balanced-audit.yml` ist **nur** zur lokalen Datenkontrolle freigegeben. Er schreibt neue Dateien in ein separates Verzeichnis, verändert die v2-Daten nicht und lädt nach GitHub ausschließlich Zahlen und Prüfsummen hoch. Keine externe API, keine GPU-Inferenz oder Trainingsschritte.

## Rang 16 und Rang 32: fairer nächster Versuch

Der bestehende Rank-32-Pilot bewies **80.740.352 trainierbare LoRA-Parameter**; das entspricht ungefähr einem Prozent der Qwen-7B-plus-Adapter-Parameter. Mathe-v2 verwendete dagegen Rank 16. Daher lässt sich aus den früheren Ergebnissen **nicht** ableiten, ob Rank 32 besser als Rank 16 wäre: Daten, Lernschritte und Lernraten unterschieden sich.

Ein fairer künftiger Vergleich verwendet **denselben eingefrorenen v3-Datensatz**, dieselben Schritte, dieselbe Lernrate, feste Seeds, einen getrennten Validation-Split und identische A/B-Messkriterien für Rank 16 und Rank 32. Vor einem Rank-32-Lauf sind eine gesondert freigegebene Traineränderung (bisher ist Rank 32 auf 3 Schritte begrenzt), ausreichender RTX-4070-VRAM sowie getrennte Adapter-Verzeichnisse erforderlich. Kein stilles Hochdrehen der Schrittzahl und kein automatischer Merge.

**Release-Gate:** mindestens formatgetreue Antworten wie v2 (15/16 ursprüngliche Aufgaben), bessere Gleichungswerte ohne Einbruch bei den 36 zusätzlichen Aufgaben, plus ein neuer unabhängiger Holdout für Coding/NERAL-Core/Systemverhalten. Erst danach darf eine Aktivierung diskutiert werden. Alle Werte bisher gelten nur als Entwicklungsdiagnose.
