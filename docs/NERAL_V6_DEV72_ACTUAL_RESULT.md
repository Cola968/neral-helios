# NERAL v6: tatsächlicher 72er Entwicklungstest, 7. November 2026

Quelle: https://github.com/Cola968/neral-helios/actions/runs/42409458549
Der Lauf ist erfolgreich abgeschlossen. Originalantworten bleiben auf dem privaten Windows-PC; der öffentliche GitHub-Bericht enthält nur aggregierte Zahlen.

Pinned Testdatei: 4b9a302a4cdb07b1e9ed3601cc1382a200ef4e4048a3afac29e72a4324f902dd
Bericht-JSON SHA-256: 56634140f3a6582dbf8702978c810dcf840f0e2f091fd6dced8ddd5e376619f0
v5 Adapter SHA-256: 2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5
v6 Adapter SHA-256: 8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd

## Nachgewiesene Scores

| Bereich | Qwen + 3 Few-Shots | NERAL v5 | NERAL v6 |
| --- | ---: | ---: | ---: |
| Alle 72 | 29 | 24 | 21 |
| Mathematik (24) | 9 | 7 | 4 |
| Aussagenlogik (24) | 15 | 14 | 14 |
| Python-Code-Tracing (24) | 5 | 3 | 3 |
| Nicht eindeutig auswertbare komplette Antworten | 0 | 7 | 19 |
| Durch Tokenbudget unvollständige Antworten | 0 | 0 | 0 |

Paarweise gegen v5: v6 verbessert 0 Fälle und regressiert 3; der
explorative exakte McNemar-Wert ist p = 0.25. Gegen Qwen+Few-Shot
verbessert v6 2 Fälle und regressiert 10; p = 0.03857421875.
Diese kleine korrelierte Entwicklungssuite wurde nicht unabhängig
vorgegeben. Der p-Wert darf nicht als allgemeiner Fähigkeitsbeweis
oder Release-Validierung dargestellt werden.

## Technische Diagnose

v6 hat gegenüber v5 **12 zusätzliche nicht eindeutig bewertbare
komplette Antworten**. Daraus folgt nicht unmittelbar, dass
zwölf zusätzliche Lösungsversuche inhaltlich falsch waren.
Die sichere Auswertung erkennt bei v6 weniger eindeutige Endwerte.
Im Bereich Mathematik sinkt die Zahl sicher richtiger Lösungen um drei.

Die bisherigen Trainingsdaten und die neue 120-Schritt-QLoRA-Fortsetzung
konnten bislang keinen übertragbaren Kompetenzgewinn belegen.
NERAL v6 soll **nicht automatisch in Helios aktiviert** und nicht
als intelligenter als v5 oder Qwen beworben werden.

## Ab sofort geltende Entscheidung

1. v5 als bisherige Vergleichs- und Rollback-Variante erhalten.
2. v6 nur als getrennten experimentellen Adapter behandeln.
3. Den CPU-only Replay-Kandidaten aus PR #353 nur zur Vorbereitung
   verifizierter train-only Fehler und v5-Fähigkeitserhalt nutzen.
   Die 72er-Dev-Suite und geschützte Release-Aufgaben bleiben außen vor.
4. Die 19 nicht eindeutig auswertbaren v6-Antworten und 7 v5-Antworten
   lokal untersuchen: Endwert-Extraktion, unklare Formatierung,
   tatsächliche Rechenfehler und Reihenfolge der Aussagen getrennt zählen.
5. Vor einem weiteren GPU-Training nur anhand des eingefrorenen,
   familiengetrennten neuen Vergleichs entscheiden. Ein höherer
   Trainings-Loss bzw. niedrigerer Loss allein ist kein Ziel.
6. Der neue CPU-Gate aus scripts/neral_v6_dev72_regression_gate.py
   verweigert ein positives Entwicklungssignal bei v6-Rückschritten.
   Auch ein bestandenes Gate erlaubt NIE selbstständig ein Release.

Das ist eine dokumentierte Regressionsanalyse, keine neue Modellversion.
