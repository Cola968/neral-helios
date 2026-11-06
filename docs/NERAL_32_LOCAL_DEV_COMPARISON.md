# NERAL – 32 zusätzliche Entwicklungsaufgaben für Qwen-vs.-LoRA

**Ziel:** Einen ehrlichen, reproduzierbaren Vergleich der bereits vorhandenen
lokalen Qwen2.5-Coder-7B-Basis und des experimentellen NERAL-Rank32-Adapters
ermöglichen. Das ist **kein neues Training**, keine Modellaktivierung und
kein unabhängiger Blindtest.

## Aufgaben

32 neu verfasste, klar abgegrenzte Entwicklungsaufgaben:
- 8 Arithmetik, Brüche, Prozentrechnung und lineare Gleichungen
- 8 SQL-Klauseln und JOIN-Entscheidungen
- 8 Python-Funktionen, nur statischer AST-Exact-Match
- 8 Agenten-Werkzeugentscheidungen, nur einzelnes freigegebenes JSON

Auswertungsmodul: scripts/neral_generalization_probe.py

**Grenzen:** Exact-Match beim Python-AST prüft syntaktische
Strukturgleichheit, **nicht** semantisch gleichwertige alternative Lösungen.
Es wird niemals generierter Code ausgeführt. Die 32 Fälle sind
projektintern und öffentlich in GitHub, daher kein verblindeter oder
unabhängiger Holdout und keine ausreichende Statistik für einen
allgemeinen Qualitätsvorsprung.

## GPU-Vergleich

Workflow: .github/workflows/neral-32-local-comparison.yml

Nur auf main manuell freizugeben mit dem exakten Wert
RUN_NERAL_LOCAL_32_DEV_COMPARE.

Der Windows-Runner benötigt:
- Das SHA-verifizierte lokale Qwen2.5-Coder-7B-Modell;
- den vorhandenen **experimentellen** NERAL-Rank32-LoRA-Adapter;
- den ursprünglichen, unverändert gepinnten Trainer und lokalen
  NF4/BF16-Laufzeit-Stack auf der RTX 4070;
- mindestens 10.000 MiB freien GPU-Speicher.

Qwen-Basis und NERAL-Adapter werden mit identischen Prompts
und dem exakt selben Qwen-Modell geprüft; für die Basis wird der
Adapter vorübergehend mit disable_adapter() ausgeschaltet.
Die Ausgabe ist deterministisch (temperature 0; do_sample=False,
160 neue Token maximal). Es gibt keine Download-/Cloud-Fallbacks.

**Privat bleiben** sämtliche Rohantworten, Eingaben, Daten,
Modellgewichte, Adapter und sonstige PC-Dateien. GitHub erhält nach
vollständigem Erfolg nur den kleinen geprüften JSON-Bericht mit
32er-Scores, Teilbereichszählern und Modell-/Adapter-Prüfsummen.
Ein Fehlschlag kann keine Adaptergewichte überschreiben oder
einen Release aktivieren.

## Entwicklungskontrolle

    python -m pytest -q tests/test_neral_generalization_probe.py tests/test_neral_generalization_gpu.py

Erst wenn der Job auf dem lokalen Windows-PC wirklich erfolgreich
durchgelaufen ist, darf ein Vergleichsergebnis angegeben werden.
Positive Zahlen allein beweisen keinen robusten, unabhängigen
Modellgewinn. Für eine Release-Freigabe fehlen weiterhin größere
Blindtests, tatsächlich ausgeführte Codingtests in einer Sandbox
und reproduzierbare Computer-Use-Aufgaben.
