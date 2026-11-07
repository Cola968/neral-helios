# NERAL-EXT: Evidenzentscheidung am 7. November 2026

Diese Notiz trennt GPU-Messung, nachträgliche CPU-Diagnosen, Hinweise ohne
bestätigte Richtigkeit und weiterhin gesperrte Funktionsbewertungen.
Keine neuen Gewichte, kein Training, kein Release-Claim.

## Provenienz

- Ursprünglicher gepaarter Lauf: https://github.com/Cola968/neral-helios/actions/runs/42395097203
- Dataset-SHA256: 47258a911ce0beb96635a34137a634556f364457487fb18a160a892302124cb1
- v5-Adapter-SHA256: 2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5
- Reiner CPU-Audit (exakt gespeicherte Antworten):
  https://github.com/Cola968/neral-helios/actions/runs/42405161499
- Erweiterter CPU-Audit (nur Zahlensignale, KEIN zusätzliches Grading):
  https://github.com/Cola968/neral-helios/actions/runs/42406971044
- Reale synthetische Docker-Smoketests: PR #347. Funktionaler Grader:
  PR #346. Kein privater Modelloutput wurde in GitHub-Artefakten veröffentlicht.

## Gepaarter GPU-Vergleich (verifizierte Resultate)

| Kategorie | Qwen Basis | Qwen + 3 Beispiele | NERAL v5 |
| --- | ---: | ---: | ---: |
| Alle 400, ursprüngliche numerische Semantik-Zählung | 59 | 84 | 91 |
| Stratum A, 150 numerische Aufgaben | 2 | 16 | 18 |
| Stratum B, 100 numerische Aufgaben | 44 | 50 | 51 |
| A+B, nur 250 numerisch bewertbare Fälle | 46 | 66 | 69 |
| Stratum C, 50 bekannte Familien | 13 | 15 | 21 |
| Stratum D, 50 Robustheitsfälle | 0 | 3 | 1 |
| Gültiges Antwortformat von 400 | 94 | 194 | 235 |

Der korrekte, numerische A+B-Abstand beträgt NERAL gegenüber Qwen+Few-Shot
3/250 = +1,2 Prozentpunkte. Der explorative gepaarte Test wurde als
p=0,6636238098144531 rekonstruiert; keine nachgewiesene generelle Verbesserung.
50 MBPP-Aufgaben aus Stratum A haben bislang keinen funktional verifizierten
Korrektheitsstatus. Die ursprüngliche 400er-Gesamtzahl ist deshalb keine
vollständige domänenübergreifende Accuracy.

## Diagnose der Antwortbeendigung

Der unveränderte GPU-Lauf speicherte die Texte bei complete=false als leer.
Darum kann kein Grader die letzten Rechenschritte oder eine hypothetisch
richtige Fortsetzung rekonstruieren.

| Zähler | Qwen Basis | Qwen+Few-Shot | NERAL v5 |
| --- | ---: | ---: | ---: |
| Insgesamt unvollständig von 400 | 126 | 12 | 2 |
| In A numerisch unvollständig von 150 | 73 | 8 | 0 |
| In A numerisch vollständig, aber konservativ unbewertbar | 75 | 79 | 72 |
| In A: letzte Zahl stimmt mit Referenz überein (UNBESTÄTIGT) | 51 | 26 | 18 |
| In A: letzte Zahl weicht ab (UNBESTÄTIGT) | 23 | 53 | 54 |
| In A: keine letzte Zahl vorhanden | 1 | 0 | 0 |
| Durch sicheren neuen Endwert-Extraktor nachträglich gerettet | 0 | 0 | 0 |

Die letzte Zahl ist keine belastbare Endantwort: Zwischenwerte, Zitate,
Stichproben-Zufallstreffer und falsche Rechenketten können exakt dieselbe
Zahl enthalten. Deshalb verändern diese Signale die Korrektheitszähler
NICHT. Es ist nicht bewiesen, dass Qwen 51 zusätzliche Aufgaben lösen
konnte, ebenso wenig, dass NERAL diese 18 Aufgaben tatsächlich korrekt
löste.

Die starke Differenz bei abgeschnittenen Antworten erklärt einen erheblichen
Teil des Vorteils bei gültigen Ausgabeformaten. Sie beweist keine neue
Schlussfolgerungsfähigkeit der Adaptergewichte.

## Programmierbewertung (Block 3)

Ein echter Docker-only-MBPP-Tester wurde implementiert und seine
Docker-Isolation auf einem GitHub-hosted Linux-Runner anhand
synthetischer richtiger, falscher, frühzeitig beendeter und
netzwerkabhängiger Programme erfolgreich getestet.

Der Versuch, die originalen 50x3 MBPP-Rohantworten lokal auszuwerten, ist
am Windows-Runner https://github.com/Cola968/neral-helios/actions/runs/42406352615
korrekt fail-closed gestoppt. Ursache: Docker ist auf dem Rechner nicht
installiert bzw. nicht im Runner-PATH; PowerShell meldete
CommandNotFoundException. Daher gibt es weiterhin KEINE echten MBPP-Scores.
Kein Ausweichen auf unsichere native Windows-Subprozesse.

Die lokale, private Stichprobe von 20 Stratum-A-Antworten pro Variante
wurde durch den CPU-Audit zum manuellen Lesen vorbereitet. Die hier
vorliegenden Berichte sind automatisiert; sie ersetzen keine wirkliche
menschliche Prüfung der Rohantworten.

## Nächste Entscheidung

1. Vor einem DPO-/ORPO-Lauf: 20 Originalantworten pro Modell manuell
   prüfen; insbesondere die als letzte Zahl übereinstimmenden,
   aber formal unbewertbaren Antworten. Ohne diese Prüfung keine
   neue angebliche Gesamt-Accuracy veröffentlichen.
2. Windows-Linux-Docker sicher einrichten und das vorhandene manuelle
   MBPP-Scoring wiederholen. Die Originalantworten müssen lokal bleiben.
   Danach Coding-Testpassrate mit eigenem Nenner und Censoring ausweisen.
3. Die bisherige NERAL-THINK-Implementierung ist ein Werkzeug-Layer, vor
   allem für einfache Mathematik. Ein besserer Gesamtagent ist nicht
   gleichbedeutend mit einem besseren LoRA-Grundmodell. THINK erst auf
   frischen, nicht fürs Training genutzten Aufgaben und mit Latenzmessung
   getrennt von reinem v5-Greedy evaluieren.
4. Kein weiteres Schablonen-SFT auf dem verbrauchten EXT-Datensatz.
   Neue Trainingsbeispiele dürfen ausschließlich train-only generiert,
   überprüft und von geschützten Holdouts getrennt werden.
5. Vor einem neuen Datensatz: Rechenfehler, Ablauf-Tokenlimit,
   tatsächliches EOS, mehrdeutige Textantworten und fehlerhafte
   Python-Programme getrennt erfassen. Rohantworten in einem nächsten
   bewusst privaten Lauf auch bei max-token-Abbruch lokal bewahren.

Das konkrete Ergebnis dieses Arbeitsschritts ist bessere Messinstrumentierung,
keine bewiesene intelligente Modellverbesserung. Die Entscheidung für v6
darf nicht auf nachträglich optimistischen Auswerter-Heuristiken beruhen.
