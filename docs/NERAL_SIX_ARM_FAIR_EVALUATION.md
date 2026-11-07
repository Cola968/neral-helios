# NERAL: fairer Gewichtseffekt durch sechs getrennte Testarme

## Vorheriger Fehler und seine Konsequenz

Historische interne 72er-Vergleiche: Qwen-Basis erhielt **drei**
eingefrorene Few-Shots, NERAL v5/v6.1 dagegen **null**.
Das beantwortet die praktische Frage: Welcher der beiden
unterschiedlich konfigurierten Chat-Modi liefert mehr strikt
parsebare Endantworten? **NICHT** die wissenschaftliche
Frage: Verbessert v5/v6.1 die Gewichte bei identischen Eingaben?

Das lässt sich durch reine Summenzähler nicht nachträglich korrigieren.
`scripts/neral_six_arm_eval.py` stellt den korrekten experimentellen
Vertrag für sechs künftige Testarme bereit:

| Modellgewicht | Nullshot | Drei identische Shots |
|---|---|---|
| Qwen 2.5 Coder 7B Foundation | `qwen_zero` | `qwen_three` |
| v5 LoRA auf gleicher Foundation | `v5_zero` | `v5_three` |
| v6.1 LoRA auf gleicher Foundation | `v61_zero` | `v61_three` |

Für jede Bedingung sind Systemprompt, Userprompt, Few-Shots,
Tokenizer, Generationseinstellungen und Grader **gleich**.
`match_shot_pair()` verweigert in Code eine als
Gewichtseffekt deklarierte Gegenüberstellung
`qwen_three` vs. `v5_zero`.

Der CPU-Vertrag berechnet für einen **neuen** 72er-Diagnosedatensatz
nur die korrekt gepaarten Modellvergleiche und pro Domäne die
Summen und nicht auswertbaren Finals. Er ist
**kein** Release-Screen, keine Trainingserlaubnis, kein
Modell-GPU-Test und keine automatische Promotion.

Die alten dev72-Sätze sind bereits verbraucht. Für einen
neuen wissenschaftlichen Leistungsvergleich sind
neue, vor der Ausführung eingefrorene und unabhängig
fachlich geprüfte Orakel erforderlich, später das
vorregistrierte 600er-Gate. Coding-Generierung braucht
ausführbare Hidden Tests; Code-Trace reicht nicht.

In diesem PR werden keinerlei LoRA-Gewichte, Trainingsdaten,
CUDA-Prozesse oder laufende Eval-Rezepte verändert.
