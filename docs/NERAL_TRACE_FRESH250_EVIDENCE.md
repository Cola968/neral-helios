# NERAL THINK: unabhängige CPU-Funktionstests mit 250 Python-Aufgaben

**Was tatsächlich geprüft wird:** der bereits im Hauptbranch vorhandene
beschränkte AST-Interpreter `neral_agent_python_trace.py` und dessen
Verknüpfung mit `neral_think.py`. Dieses Experiment enthält **keine**
neue Qwen-/NERAL-GPU-Inferenz, keine neue LoRA und keinen Beweis für einen
Gewichtstrainingsdurchbruch.

## Testdesign

`tests/test_neral_python_trace_fresh250.py` erzeugt deterministisch
250 unterschiedliche Python-Programme unabhängig von den verbrauchten
72er-Entwicklungsprompts und ihren Modellantworten. Jeweils 50 betreffen
Listen-Slices und Summen, Wörterbuch-Updates, verschachtelte Zählschleifen
mit `continue`, String-Slices mit `count` sowie Bedingungen und boolesche
Verknüpfungen.

Ein separater CPU-Rechenausdruck liefert die erwartete ganze Zahl. Der
eingereichte Programmtext wird ausschließlich per AST interpretiert,
**niemals direkt ausgeführt**. Jeder positive Fall verlangt
`exact=true`, `method=python_trace_ast`, `source_executed=false` und
den korrekten Endwert. Zusätzliche Tests verweigern dynamische Ausführung,
Imports, Dateien, zu große Schleifen, beliebige Funktionen und fremde
Anweisungen in der Nutzerfrage.

Der THINK-Integrationstest gibt absichtlich falsche, simulierte
Modellantworten zurück. Die korrigierte Zahl muss dem Werkzeug
zugeordnet bleiben (`model_generated=false`,
`tool_used=neral_python_trace_static`). Kein Ergebnis darf als bessere
neuronale Gewichtsleistung gezählt werden.

## Nachweise und Grenzen

GitHub Actions `.github/workflows/neral-think-ci.yml` prüft die neue
Suite **unter Windows und Linux**, zusammen mit den vorhandenen THINK-
Regressionstests und Flask-/UI-Verträgen.

Beobachteter technischer Fortschritt ist höhere Testabdeckung für einen
überprüfbaren Teil des Codeverständnisses. Die erfassten Programme
ähneln nicht im Wortlaut der dev72-Aufgaben, teilen aber grundlegende
Programmierkonstrukte; sie sind keine extern unabhängige KI-Suite.
Eine Bewertung von tatsächlichem Codegenerieren, Debugging,
Neuheitsgeneralisation der trainierten Gewichte, Sicherheit außerhalb
des zugelassenen AST-Subsets oder menschlicher Programmierleistung
ist dadurch **nicht** möglich.

Für einen Gewichts-Intelligenzsprung bleibt der nächste Schritt die
vorregistrierte frische GPU-Evaluation von v6.1 gegen v5 und Qwen mit
identischen Inferenzkonfigurationen und überprüfbarer Fehleranalyse.
