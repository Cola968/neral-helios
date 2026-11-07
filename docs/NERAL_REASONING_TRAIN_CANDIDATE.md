# Kleiner Trainingskandidat für überprüfbare Rechenschritte

`scripts/neral_reasoning_train_candidate.py` erzeugt ausschließlich neue
Trainingsdaten: **128 Aufgaben in 64 Gegenbeispielpaaren**, keine Validation,
keinen Test und kein Modell. Bestehende Generatoren und gesperrte Fälle werden
nicht eingelesen oder verändert. Die Daten sind synthetisch; bessere
Generalisierung ist damit noch nicht nachgewiesen.

Jedes Paar teilt dieselben Zahlen und dieselbe Formulierung. Eine relevante
Regel ändert sich, weshalb beide Antworten unterschiedlich sein müssen:

- Preisrechnung: Rabatt → Gebühr → Aufschlag gegenüber Aufschlag → Gebühr → Rabatt.
  Die feste Gebühr verhindert, dass beide Reihenfolgen zufällig kommutieren.
- Treibstoffplanung: Zwei Abschnitte mit unterschiedlichen kg/h-Raten;
  die Reserve nutzt einmal Rate A und einmal Rate B. Beide kg/min-Umrechnungen
  und beide Gesamtergebnisse sind absichtlich nicht ganzzahlig.

Jede Antwort enthält explizite Zwischenrechnungen unter `CHECK` und eine
`FINAL_ANSWER`. Der Generator rechnet mit `Fraction`. Ein separater
70-stelliger `Decimal`-Interpreter prüft jeden Knoten des Operationsgraphen;
Preisrundung erfolgt erst am Ende mit `ROUND_HALF_UP`. Eine zusätzliche
ganzzahlige Kreuzproduktprüfung erzwingt exakte Zwischenwerte, auch unterhalb
der Decimal-Replay-Toleranz von `1e-55` bei periodischen Dezimalzahlen.
Zusätzlich prüfen Tests eine anders faktorisierte
Preisformel, die Treibstoffgleichung per ganzzahliger Kreuzmultiplikation,
einen halben Cent und manipulierte Schritte, Graphen, Antworten und Texte.

## Lokal erzeugen

```powershell
python -B -m scripts.neral_reasoning_train_candidate --out C:\private\neral-reasoning-train-candidate
python -B -m unittest discover -s tests -p test_neral_reasoning_train_candidate.py -v
```

Der Zielordner muss neu sein. `train.jsonl` enthält nur `messages` im
bisherigen Drei-Turn-Trainerformat. `proofs.jsonl` hält die Rechengraphen,
Fakten und Paar-IDs getrennt. Das Manifest dokumentiert Datei-/Quell-SHAs,
Seed und ungeprüfte Voraussetzungen. Es erstellt kein Review-Manifest für
den alten Trainer und umgeht keine dessen Beschränkungen.

Vor einer späteren Trainingsfreigabe müssen alle offenen Gates ausdrücklich
erledigt werden:

1. Menschlicher Bedeutungs- und Einheitenreview, insbesondere der deutschen
   Aufgabenformulierungen; reine Reproduzierbarkeit genügt nicht.
2. Echter lokaler Qwen-Tokenizer mit dem verwendeten Chat-Template:
   vollständige Sequenzlängen und Completion-Masken prüfen. Kein stilles Kürzen
   auf 512 Tokens; maximale Länge ist hier noch nicht verifiziert.
3. Vollständiges Trainingsinventar gegen Dev-/Releasequellen prüfen,
   einschließlich semantischer Dubletten. Die neue Seedwahl beweist keine
   Unabhängigkeit; gesperrte Antworten bleiben außerhalb des Trainings.
4. Unabhängige Dev- und Releasefälle vor weiterer Modellinferenz einfrieren.
   Die 128 Trainingsfälle und ihre Zahlenvarianten sind kein Leistungstest.
5. Separater experimenteller Adapter und ausdrückliche lokale Trainingsfreigabe;
   keine automatische Integration oder Promotion.

Alle entsprechenden Freigabefelder bleiben `false`. Zwei Familien und vier
Einleitungssätze pro Familie sind ein kontrollierbarer Baustein, kein breiter
Reasoning-Korpus. Der Nutzen der Paare und Zwischenschritte muss später gegen
eine gleich große Kontrollmenge bei gleichem Token-/Optimizerbudget gemessen
werden. Mehr Details im Zieltext erhöhen das Tokenbudget; ein Vergleich nur
nach Aufgabenanzahl wäre daher irreführend.

Die bisherigen 480 Optimizer-Schritte dürfen nicht unverändert auf diese
128 Fälle angewendet werden: Mit Batchgröße 1 und vier Akkumulationen wären
das 15 Epochen. Eine Epoche dieses Kandidaten umfasst 32 Optimizer-Schritte.
Für die Datenablation zunächst Rang 16, Lernrate `5e-5`, NF4/BF16 und die
bestehenden Decodingbedingungen konstant halten. Das ist eine kontrollierte
Ausgangsbedingung, keine Behauptung optimaler Hyperparameter.
