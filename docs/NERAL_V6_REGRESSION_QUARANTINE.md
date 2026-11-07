# NERAL: automatische Regressionssperre für neue Adapter

**Stand: 7. November 2026. Keine automatische Freigabe eines neuen Modells.**

## Anlass und Befund

Beim [echten 72-Fälle-GPU-Vergleich #42409458549](https://github.com/Cola968/neral-helios/actions/runs/42409458549) erreichte Qwen+3 Few-Shots 29/72, NERAL v5 24/72 und NERAL v6 nur 21/72. Bei den vollständig generierten Antworten waren 0/72 (Qwen), 7/72 (v5) bzw. 19/72 (v6) nicht eindeutig auswertbar. Die Entwicklungsfamilien bestanden aus 24 Mathematik-, 24 Logik- und 24 Python-Code-Trace-Aufgaben; Code **schreiben** wurde nicht geprüft. Dieser Test ist eine kleine, nicht unabhängige Entwicklungssuite. Der Unterschied v6 zu v5 war in diesem Test nicht statistisch belastbar. Das rechtfertigt **keine** Behauptung, dass v6 allgemein dümmer ist, jedoch eine vorsorgliche Entwicklungsquarantäne.

**Explizit blockierter Adapter:** SHA-256 `8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd`. Dieser experimentelle v6-Adapter darf wegen der beobachteten Regression nicht zum Standard erklärt werden. Die alte v5-Datei wird nicht überschrieben.

## Neue Prüfstufe nach jedem 72er-Vergleich

Die Auswertung in `scripts/neral_v6_fresh_dev_eval.py --mode combine` schreibt künftig automatisch `regression_screen` in das **private** Summen-JSON. Das neue Modul `scripts/neral_v6_regression_gate.py` prüft:

- Vorhandensein von Qwen+Few-Shot und v5 als Referenzen, unveränderte Datenfamilien und Suite, Modell-Hashes, Aufgaben-/Domänen-Zähler und rechnerisch stimmige gepaarte Verbesserungen und Regressionen.
- Neues Modell muss in der Gesamtzahl **beide** Referenzen strikt übertreffen; kein Fachbereich darf hinter einer Referenz zurückfallen.
- Abgeschnittene und nicht eindeutig auswertbare Antworten dürfen gegenüber keiner Referenz zunehmen.
- Der SHA-256 des bekannten gescheiterten v6 bleibt **auch bei vermeintlich besserem Report** gesperrt.
- Fehlende, manipulierbare oder widersprüchliche Reports werden fail-closed abgewiesen.

**Befehle auf dem eigenen PC:**

```powershell
python -m scripts.neral_v6_regression_gate --report C:\private\v6-new-dev72\summary.json
python -m scripts.neral_v6_regression_gate --report C:\private\v6-new-dev72\summary.json --require-screen-pass
```

Der zweite Befehl bricht mit Fehlercode ab, falls ein bekannter oder gemessener Rückschritt vorliegt. Das ist ein **technischer Stopp für Workflows, die diese Sperre aufrufen**, keine Behauptung, dass künftig jeder Adapter-Loader automatisch verhindert ist. Das Development-Screen-Resultat führt niemals zum automatischen Deployment: `release_authorized=false` bleibt auch bei bestandener Vorprüfung.

## Ehrliche Grenzen / Freigabepolitik

Die derzeitige Entwicklungs-Suite ist verbraucht. Ihre Fälle dürfen zur Diagnose und als Regressionstest verwendet werden, aber **nicht** zur unabhängigen Releasefreigabe. Der rohe 72er-Report darf nur mit vertrauenswürdiger Herkunft bewertet werden; Hashes und Zähler belegen nicht eigenständig, dass ein Modell die Aufgaben gelöst hat.

Vor einem späteren v6+/v7-Release gelten weiter die vorregistrierten Regeln aus [GATE.md](../GATE.md): neue externe Aufgaben, 600 unabhängige Fälle, begrenzte Domänenregression, statistische Schranken, Sicherheitsabnahme, alter Adapter als Referenz und gesonderte Freigabe. Gegen künftige Fehlversuche gibt es keine absolute Garantie – wohl aber ein verbindliches **Nicht-Freigeben** bei bekanntem Versagen. Tests auf Windows und Linux unter `.github/workflows/neral-v6-dev72-cpu-ci.yml` prüfen auch die Sperrlogik. Es wurde **kein** GPU-Training, kein kostenpflichtiger Cloud-Dienst und kein Adapter-Merge gestartet.
