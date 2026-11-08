# NERAL THINK – endliche Logik nachvollziehbar prüfen (08.11.2026)

## Abgrenzung

Dies ist **eine neue NERAL-Systemfähigkeit**, nicht ein Gewichts- oder ein LoRA-Intelligenzsprung. Der lokale Qwen-basierte Generator darf eine Antwort vorschlagen; NERAL THINK kann sie bei vier **streng begrenzten** Aufgabenarten unabhängig prüfen. Bei einem nachweisbaren Fehler liefert das Werkzeug seine eigene richtige Zahl. Dabei bleibt `model_generated=false`, `verified=true`, `verification_method=finite_logic_exact`, `tool_used=neral_logic_exact`. Auch eine korrekte Modellantwort wird nur wegen der unabhängigen Werkzeugprüfung als verifiziert markiert. Die generierten Gewichte bleiben unverändert.

## Implementiert

- XOR-Parität für 2–12 Bits (auch die alte Vier-Schalter-Vorlage).
- Erreichbarkeit in gerichteten Graphen, inklusive Zyklen, begrenzt auf Knoten 0–31 und höchstens 64 Kanten; die erreichbaren Knoten erscheinen als strukturiertes Zertifikat.
- Exhaustive Prüfung aller acht booleschen Belegungen von A/B/C unter den explizit unterstützten XOR-/OR-/NOT-Bedingungen und einer Maximalsumme; Zertifikat mit den erfüllenden Belegungen.
- Drei alternative gerichtete Wege mit nichtnegativen gewichteten Kanten; Zertifikat enthält die drei Pfadkosten und den minimalen Weg.

Jedes Werkzeug akzeptiert nur exakt erlaubte deutsche/englische Aufgabenmuster, optional mit genau einem festgelegten Ausgabesuffix. Zusätzliche Sätze, Codeaufrufe, Shell-Anweisungen, unzulässige Zahlbereiche oder mehrteilige Anfragen werden **nicht** automatisch durch das Werkzeug beantwortet. Es gibt kein `eval`, `exec`, generiertes Python, eine Netzwerksuche oder Datenweitergabe.

## Testdesign und ehrliche Messung

Die Entwicklungstests prüfen:

- 24 Logikfälle des **bereits verwendeten** 72er-Datensatzes (nur Regression, keine unbekannte Generalisierung).
- Je 64 neu erzeugte, reproduzierbare Zahlen-/Graphvarianten für vier Familien, also 256 zusätzliche systematische Werkzeugfälle. Diese Variationen sind keine unabhängig ausgewählte Endsuite und keine breite Prüfung der natürlichen Sprache.
- Absichtlich falsche Stub-Antworten am THINK-Endpunkt: Für die verbrauchte 72er-Suite werden nach der Änderung 24 Mathe-, 24 Logik- und 18 Python-Tracing-Aufgaben durch Werkzeuge beantwortet, insgesamt 66/72. Das misst **Werkzeugabdeckung** bei einem Fake-Modell, nicht die Trefferquote der tatsächlichen Modellgewichte.
- Negative Fälle: ungültige Bits, zu große Knoten, ungeprüfte Zusatzanweisungen, mehrteilige Prompts, unzulässige boolesche Grenzen und negative Pfadkosten.
- Zusätzlich HTTP-Integration mit korrekten Angaben zur Herkunft und Maschinen-Zertifikat.

## Was NERAL von Qwen unterscheidet

In NERAL THINK ist das lokale Sprachmodell nicht die letzte Instanz über Wahr oder Falsch. Der von NERAL kontrollierte Logik-Solver kann ausgewählte Problemklassen deterministisch lösen und seine Rechenschritte als strukturiertes Zertifikat bereitstellen. Das ist eine **Architektur- und Produktdifferenz**, nicht der Nachweis besserer Basisgewichte als Qwen.

Für eine tatsächliche Modell-Differenz sind weitere Schritte erforderlich: Neue, extern geprüfte Logik-Transferaufgaben, prompt- und decoder-gleiche Qwen-vs.-LoRA-Vergleiche **ohne Werkzeuge**, Retention-Gate gegen Vergessen, dann erst vorsichtiges gezieltes Training. Der bereits implementierte Review-Gate in `scripts/neral_transfer_diagnostics.py` darf weiterhin keinen Adapter automatisch aktivieren.

## Grenzen

NERAL versteht damit noch **nicht beliebige Logikrätsel**. Allgemeine Beweissuche, Weltwissen, Mehrdeutigkeit, offene Argumentationsaufgaben und komplexes mehrstufiges Planen bleiben modellabhängig und unverifiziert. Kein Ergebnis darf als „66/72 Modellleistung“ oder „NERAL schlägt Qwen“ veröffentlicht werden.
