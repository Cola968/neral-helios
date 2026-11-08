# NERAL v7: Offline-Vergleich neuer Basismodelle

**Zweck:** Vor weiteren QLoRA-Trainings eine stärkere lokale
Ausgangsbasis auswählen. Weder Modellqualität noch ein Intelligenzsprung
werden vor dem echten Lauf behauptet.

## Vergleich

`scripts/neral_v7_foundation_tournament.py` erzeugt 72 deterministische,
originale Entwicklungsaufgaben aus **12 Methodenfamilien**: Mathematik
(Binomialkoeffizienten, modulare Inverse, gewichtete Stellenwertziffern,
exklusive Vielfache), Reasoning (Implikation und XOR, Pfadanzahlen in
DAGs, taktbasierte Ankunftsoptimierung, Reihenfolgen mit Nebenbedingungen)
sowie Python-Codeverständnis (Dictionary-Akkumulation, gewichtete Slices,
verschachtelte Breaks, Zeichenkettenfilter).

Die Programme werden als Text behandelt. **Generierter oder vorgelegter
Python-Code wird niemals ausgeführt.** Die korrekten Integer-Ergebnisse
entstehen aus separaten, vertrauenswürdigen Python-Rechenoperationen.
Die Scorefunktion akzeptiert ausschließlich **eine eindeutige letzte
`FINAL_ANSWER: <integer>`-Zeile**. Mehrdeutige Formate zählen
`ungraded`, eindeutig falsche Zahlen `wrong_final`, abgebrochene
Ausgaben `incomplete`. Damit ist die Bewertungsregel überprüfbar.

## Lokale Modelle und Datenschutz

`--mode inventory` ruft nur
`http://127.0.0.1:11434/api/tags` auf. Es wird niemals ein
`ollama pull`, `/api/pull`, Remote-API, eine Proxyverbindung oder ein
Redirect ausgeführt. Ausgewählte Modell-Tags müssen **exakt** schon lokal
installiert sein und zu `qwen2.5-coder:`, `qwen3:` oder
`qwen3.5:` gehören. Die Digests werden vor und nach dem Lauf geprüft.

Es gibt jeweils dieselben 72 Fragen, denselben Systemprompt,
`temperature=0`, `num_ctx=4096` sowie dasselbe
`num_predict`-Budget für jedes Modell. **OFF** verwendet 512
Generierungstokens, **ON** 2048 und den Ollama-`think`-Parameter.
Beide Einstellungen müssen **getrennt** berichtet werden. Nicht alle
Modelle unterstützen denselben Denkmodus: Wenn der lokale Ollama-Dienst
diese Anfrage ablehnt, schlägt der Lauf fehl. Ein Modell kann den
Denkparameter aber auch ignorieren; die bloße Einstellung `thinking=on`
ist deshalb **kein Nachweis**, dass alle Modelle denselben Denkmodus
benutzt haben. Der erste Vergleich sollte `thinking=off` verwenden.

Beispiel für die Ausführung am lokalen Windows-PC:

```powershell
python -m scripts.neral_v7_foundation_tournament --mode inventory
python -m scripts.neral_v7_foundation_tournament --mode compare --models qwen3:8b,qwen3.5:9b --thinking off --out C:\private\neral-v7-foundation
```

*Die Modell-Tags im Beispiel sind keine Aussage darüber, was bereits
installiert ist. Die Inventory-Ausgabe ist maßgeblich.*

Alle Aufgaben, Antworten und Rohtexte werden als private Dateien
**außerhalb** des Repositories gespeichert. Nur die aggregierte
`scalar-summary.json` ist für GitHub-Artefakte vorgesehen.
Fehler, Digestabweichungen oder unterbrochene Läufe dürfen **keine**
erfolgreiche Summary erzeugen. Keine Gewichtsdatei wird geändert,
geladenes Modell wird nicht trainiert, kein Kandidat wird automatisch
in NERAL geladen.

## GitHub Workflow

Auf `main` den manuellen Workflow
`.github/workflows/neral-v7-foundation-local-gpu.yml` ausführen.
Eingaben: `authorize=BENCHMARK_NERAL_V7_LOCAL_FOUNDATIONS_ONLY`,
`installed_models` als kommagetrennte Liste von zwei bis drei
**vorhandenen** Qwen-Tags, `thinking=off` für den ersten Lauf.

Der Job belegt ausschließlich den lokalen Windows-GPU-Runner, prüft
zuerst CPU-Kontrakte und Ollama-Inventory, bricht bei GPU-Konkurrenz ab
und veröffentlicht nur die aggregierte Ergebnisdatei. Dauer hängt
vom lokal geladenen Modell ab. Eine CUDA-GPU allein garantiert keine
vollständige GPU-Auslagerung von Ollama-Modellen; besonders
Latenzmessungen sind entsprechend zu prüfen.

## NERAL-Release-Grenzen

Dies ist ein **öffentlich dokumentierter, synthetischer
Entwicklungsbenchmark**, keine verdeckte unabhängige Releasesuite.
Codeverständnis ist nicht gleich Codegenerierung. Es gibt weder versteckte
Unit-Tests für neu generierten Code noch eine unabhängig begutachtete
Aufgabensammlung. Modell- und Quantisierungsunterschiede sowie
Modell-Laufzeit- und Prompt-Template-Effekte können einen
Vergleich zwischen Basismodellen beeinflussen.

Der Test soll nur helfen, eine mögliche stärkere Offline-Basis zu
identifizieren. Erst danach werden ein mit eigenen Trainingsaufgaben
aufgesetzter NERAL-v7-Adapter und dessen **same-base A/B-Test mit und
ohne LoRA** sinnvoll. Das etablierte 600-Fälle-Release-Gate gilt
weiter; diese 72 Aufgaben ersetzen es nicht.
