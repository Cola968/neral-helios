# NERAL LOW/HIGH: Qualitätsprüfung statt bloßer Rechenzeit

Stand: 8. November 2026. **HIGH ist noch nicht als qualitativ besser als LOW nachgewiesen.** Dieser Entwicklungscheck misst beide Modi, ohne den Qwen-Basisgewichten eine Änderung zuzuschreiben.

## Was verglichen wird

Dasselbe lokal installierte und unveränderte `qwen3:8b` erhält exakt dieselbe Nutzerfrage, einschließlich identischer Endformat-Anweisung. Die beiden direkten NERAL-1.0-Chatwege sind:

- LOW: `POST /api/neral/chat/generate`, direkt und ohne Qwen-internes Thinking.
- HIGH: `POST /api/neral/chat/think`, Qwen-internes Thinking und optional eine zweite kritische Überarbeitung; exakte NERAL-Werkzeuge nur bei sicher erkannten Aufgaben.

LOW/HIGH werden pro Aufgabe abwechselnd zuerst gestartet, um Reihenfolgeeffekte teilweise zu begrenzen. Bei jedem Aufruf müssen Modellkennung und Herkunft unverändert sein, der lokale Adapter muss ausdrücklich `adapter_loaded=false` melden. Nur die letzte eindeutig ausgewiesene ganzzahlige Antwort wird mit einem deterministischen Ergebnis verglichen. Nicht lesbare, zu lange, ungültige, unvollständige und fehlgeschlagene Antworten zählen **nicht** als richtig.

**Das ist bewusst ein Systemvergleich**, kein fairer Gewichtsvergleich mit gleichem Token-/Werkzeugbudget. HIGH erhält mehr Token, Rechenzeit und gegebenenfalls lokale Hilfswerkzeuge. Tool-assistierte Antworten werden gesondert gezählt.

## Entwicklungskriterium (vor der Auswertung festgelegt)

Die vollständig ausgeführte **72-Aufgaben-Entwicklungssuite** mit 12 Methodenfamilien aus Mathematik, Logik und Python-Ausgabe muss gleichzeitig erfüllen:

1. Mindestens **+5 Prozentpunkte** mehr korrekt gelöste Aufgaben durch HIGH.
2. Exakter gepaarter McNemar-Test **p ≤ 0,05**, keine bloß länger wirkenden Antworten.
3. Keine negative Nettoveränderung in einer der drei Domänen.

Ein 12- oder 24-Fälle-Diagnoselauf kann diese Gate-Bedingungen **nicht** erfüllen und darf nicht als allgemeiner Erfolg ausgegeben werden. Der Auswerter prüft auch Fälle, die LOW richtig und HIGH falsch löst. Jede solche Verschlechterung zählt gegen HIGH.

Diese 72 Aufgaben wurden bereits für frühere Entwicklungsvergleiche veröffentlicht und verwendet. Sie sind deshalb **kein** versiegelter, vorher ungesehener Release-Benchmark. Auch ein bestandenes Entwicklungs-Gate erzeugt ausdrücklich `release_validated=false`, `neural_improvement_proven=false` und `general_reasoning_superiority_proven=false`.

Die nächste Aussage „HIGH ist allgemein besser“ braucht eine neue, eingefrorene, **unbekannte** Suite mit unabhängig bewerteten offenen Fragen, echten Coding-Aufgaben und dem gleichen gepaarten LOW/HIGH-Protokoll. Die Ergebnisse dürfen nicht zum Nachjustieren des Gates verwendet werden.

## Lokal ausführen – keine Cloud-KI

Das folgende Verfahren ist für das **vollständige GitHub-Repository**, nicht für das reduzierte Vorschau-ZIP. Ollama und `qwen3:8b` müssen bereits installiert sein. Der ausführende Python-Interpreter benötigt die vorhandenen lokalen Web-App-Abhängigkeiten.

```powershell
python -m scripts.neral_low_high_quality --count 12 --out .\local-high-pilot.json
```

Für den vollständigen Entwicklungsvergleich, nur wenn die RTX 4070 frei ist und kein Training läuft:

```powershell
python -m scripts.neral_low_high_quality --count 72 --out .\local-high-72.json
```

Die Ausführung startet **keinen** Modell-Download, kein Training und keine Cloud-Inferenz. Sie arbeitet ausschließlich über die lokale Ollama-Schnittstelle und die vorhandene HTTP-Schicht. Sie kann auf der RTX 4070 erheblich länger dauern als LOW. Der Nutzer muss den Befehl bewusst ausführen; vor automatischer Veröffentlichung gibt es eine separate PC-Abnahme. Die Grenzen des HIGH-Backends sind pro Modellaufruf gesetzt, jedoch keine harte Unterbrechung einer bereits laufenden Modellgenerierung.

Standardmäßig schreibt das Skript nur einen **neuen aggregierten JSON-Bericht** ohne Fragen oder Modellantworten. Bestehende Dateien werden nicht überschrieben. Reportinhalte sind Kennzahlen, SHA-256 von Suite und lokalem Modell, Latenzen, Tool-Nutzung und methodische Grenzen. **Niemals** rohe Aufgaben oder generierte Antworten als öffentliches Actions-Artefakt hochladen.

## Warum HIGH nicht automatisch gewinnt

Ein zweiter Modellaufruf kann den ersten Entwurf auch verschlechtern. Auf exakt verifizierbaren Teilaufgaben nutzt HIGH den unabhängigen Rechner zur Korrektur; bei offenen Fragen besteht bisher nur eine Modell-Selbstprüfung. Das ist keine externe Wahrheitskontrolle. Bei einer nachgewiesenen Regression zuerst den betroffenen Aufgabentyp analysieren und die Prüf- bzw. Auswahlstrategie korrigieren; keine willkürlichen nachträglichen Änderung von Testfällen oder Bewertungsregeln. Ziel ist weniger Fehler, nicht größere Antworten.

CPU-Vertragstests:

```powershell
python -m unittest discover -s tests -p test_neral_low_high_quality.py -v
```
