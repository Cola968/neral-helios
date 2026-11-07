# NERAL THINK: mehr überprüfbares Python-Codeverständnis ohne Code-Ausführung

Erweiterung des vorhandenen konservativen AST-Interpreters
`neral_agent_python_trace.py` (07.11.2026).

Ziel: Das Modell kann bei geeigneten Codeverständnisfragen seine
Antwort mit einem deterministischen, begrenzten Python-AST-Interpreter
kontrollieren lassen, statt den eigenen Ratewert zu bestätigen.
Es wird **kein von Nutzer oder Modell gelieferter Python-Code
ausgeführt**. `eval`, `exec`, `compile`, Imports,
Netzwerk, Dateisystem und Subprozesse bleiben unzugänglich.

Neu unterstützt:

- Mutable `list`-Aliasreferenzen (z. B. `a=[4,7]; b=a`).
- Geprüfte `list.append`-Anweisungen, Index-Schreibzugriffe und
  `+=` auf vorhandene Listenelemente.
- Genau einstellige, begrenzte `{f(x) for x in numbers if test(x)}`
  Mengenkomprehensionen, deren Elemente Ganzzahlen sind.
- Geprüfte `sum(expression for index,value in enumerate(list) if ...)`
  auf einer endlichen Liste.
- Keine List-/Generator-Methodenausführung außerhalb der
  explizit überprüften AST-Syntax. Komprehensionszielvariablen
  werden nach der Auswertung wiederhergestellt.

Bestehende Limits gelten weiter: maximal 3000 Zeichen Quelltext,
180 AST-Knoten, 1200 Berechnungsschritte, 128 Elemente und
ein eindeutiger Integer-`print`. Alle neuen Tests prüfen sowohl
korrekte Ergebnisse auf unabhängig kontrollierten Quelltexten
als auch Ablehnung von `import`, `__import__`,
gefährlichen Attributen, mehrteiligen Generatoren,
zu großen Ranges/Collections sowie unzulässiger Listenmutation.

Diese Methode verbessert die **Systemfähigkeit mit THINK und
Verifikator**. Sie beweist keine neue selbstständig erlernte
Codingfähigkeit der zugrundeliegenden 7B-Gewichte und darf nicht
als solche beworben werden.

Rückrollbar: Ein einziger isolierter Modul-/Tests-Change.
Das vorherige v5 und die noch nicht freigegebenen v6/v6.1-Adapter
werden nicht verändert.
