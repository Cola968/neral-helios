# NERAL V2: freier Helios-Code-Loop (Entwicklungsstand)

**Ausgangslage:** V2 baut auf PR #561 auf, der noch gemerged werden muss. Dieser Branch verändert `main` nicht.

## Funktionsweise
- Zunächst bleiben die bisherigen unabhängigen Vollquellenbeweise aktiv.
- Freie Rechen-, Logik- und Planungsaufgaben werden heuristisch erkannt (kein starrer Aufgabenparser). Helios erzeugt einen einzelnen Python-Codeblock, der ausschließlich in Docker ausgeführt werden darf.
- Bei Fehlern gibt es höchstens **zwei** Skriptreparaturen. Fehlt Docker, wird kein Host-Python-Fallback ausgeführt.
- Erfolgreiches Docker-`stdout` wird wortgetreu angezeigt; eine zusätzliche Helios-Erklärung ist ausdrücklich **nicht unabhängig verifiziert** (`verified: false`, `sandbox_executed: true`).
- Alle anderen Chat-Aufgaben können weiterhin regulär über Luna/Terra/Helios beantwortet werden.

## Solver-Container vorbereiten
Das bisherige `python:3.11-alpine` enthält `sympy` und `z3` nicht zwangsläufig. Deshalb gibt es einen **separaten, ausdrücklich zu installierenden** Container. Einmalig auf dem eigenen PC im Repository ausführen:

```powershell
docker build --file docker/neral-code-v2.Dockerfile --tag neral-python-tools:v2 .
```

Danach **für den NERAL-Serverprozess** `NERAL_CODE_SANDBOX_V2=1` setzen und den Dienst kontrolliert neu starten. Erst wenn das Image lokal existiert und der Server mit dieser Umgebungsvariable läuft, werden `sympy` und `z3` aus dem vorinstallierten Image bereitgestellt. Es findet beim einzelnen Code-Lauf kein Image-Download statt (`--pull never`), und Docker bleibt ohne Netzwerk, mit Read-only-Dateisystem, reduzierten Berechtigungen sowie Zeit-/Speicherlimits. Bei fehlendem Image ist das Ergebnis ausdrücklich nicht ausgeführt.

## Aussagekraft der Tests
Die fünf CPU-Regressionstests verwenden einen **Fake-Helios** und einen **Fake-Docker-Callback**, der die Codeübergabe kontrolliert. Sie beweisen korrekte Einbindung, begrenzte Retries und ehrliche Metadaten, **nicht** die tatsächliche Lösungsquote eines unbekannten 9B-Modells. Eine reale fünfteilige Abnahme auf RTX 4070 und dem laufenden Server steht aus.

**Keine universelle mathematische Garantie:** Modellgenerierter Code kann mit falschen Annahmen erfolgreich durchlaufen. Für ein echtes `verified: true` ist weiterhin eine unabhängige, vollständige Aufgabenprüfung nötig.

## Reale Fünf-Aufgaben-Abnahme vor Merge
Wenn PR #561 gemerged ist, der V2-Stand isoliert installiert und der eigene Drei-Modell-Dienst auf Port 9690 aktiv ist, lokal ausführen:

```powershell
python -B scripts/neral_v2_live5.py
```

Das Skript erstellt fünf im Lauf frisch parametrisierte Aufgaben, sendet sie mit einem kurzlebigen Testkonto an den echten lokalen Chat, kontrolliert `sandbox_executed`, `verified: false` und die exakten numerischen Ergebnisse und versucht, das Testkonto abschließend zu löschen. Der Quellcode der fünf Aufgaben und die erwarteten Werte werden nicht an GitHub übertragen. Dieser reale Test wurde noch **nicht** ausgeführt.
