# NERAL Assistant – lokaler 9B-Pilot (RTX 4070)

**Status:** Der Code für NERAL Assistant und die eigene App-Oberfläche ist als
Entwicklungs-Pull-Request vorhanden. **Kein Leistungsversprechen und keine
Produktfreigabe.** Der Runner hat am 27.10.2026 die RTX 4070 (12.282 MiB)
erkannt. Ein erneuter Runner-Test bestätigte am 27.10.2026,
dass **Ollama erreichbar und Qwen3.5-9B installiert** ist. Ein **einziger
nativer Smoke-Test** lieferte auf die Aufforderung, exakt BER zu antworten,
korrekt `BER`. Vollständige NERAL-App-End-to-End-Tests mit echtem Modell
und aussagekräftige Qualitätsmessungen stehen weiterhin aus.

## Was NERAL ist – und was nicht

**NERAL Assistant** ist die eigene NERAL-Anwendung mit eigener Oberfläche,
lokalem Gesprächsverlauf, Sicherheitsprüfungen und einem eingeschränkten
Rechenwerkzeug. Die erste leistungsfähige Modellbasis ist **Qwen3.5-9B**
(Open-Weight-Modell von Qwen/Alibaba unter Apache-2.0), *nicht* ein vollständig
von uns von Grund auf trainiertes NERAL-Modell. Eigene **Arctus-350M-Gewichte**
bleiben unberührt. Ein eigener NERAL-Adapter kann später getrennt entwickelt
und hinsichtlich Lizenz, Datenherkunft und Leistungsgewinn evaluiert werden.

Offizielle Modellquelle: https://huggingface.co/Qwen/Qwen3.5-9B
Alternative für kleinere Grafikspeicher: https://huggingface.co/Qwen/Qwen3.5-4B
Ollama-Modelle: https://ollama.com/library/qwen3.5

Apache-2.0 erfordert insbesondere Erhalt der jeweiligen Lizenz-/Copyright-
Hinweise bei Weitergabe und angemessene Kennzeichnung eigener Änderungen.
Auch Drittanbieter-Quantisierungen und verwendete Trainingsdaten gesondert
lizenzrechtlich prüfen.

## Installation auf dem eigenen Windows-PC (manuell)

1. Installiere **Ollama** über https://ollama.com/download.
2. Prüfe in PowerShell `ollama --version`. Starte den lokalen Dienst,
   falls er nicht bereits läuft.
3. **Nur wenn du den Download ausdrücklich willst**, lade die Modellgewichte:

```powershell
ollama pull qwen3.5:9b
ollama list
```

Die Ollama-Modellauflistung gibt für den 9B-Quantisierungsdownload ungefähr
6,6 GB Modellgröße an. Zusätzlich benötigt Inferenz Speicher für Kontext,
KV-Cache und Laufzeit. Auf 12 GB VRAM wird daher zunächst
`num_ctx=4096` begrenzt. Verfügbare VRAM-Menge, tatsächliche Auslastung und
Antwortgeschwindigkeit sind noch zu messen.

4. Installiere die NERAL-Projektabhängigkeiten wie in `REPRODUCE.md`
   beschrieben. Falls nötig, installiere im Projekt-Python-Environment
   `pip install -r requirements.txt`. Die App verwendet CPU-PyTorch;
   **Ollama übernimmt die GPU-Inferenz separat**.
5. Starte die NERAL-App aus dem **aktuellen Branch** in PowerShell:

```powershell
$env:NERAL_ASSISTANT_ENABLED = "1"
$env:NERAL_ASSISTANT_MODEL = "qwen3.5:9b"
$env:NERAL_ASSISTANT_THINK = "0"
python app.py
```

Öffne http://127.0.0.1:9680, wähle **✦ NERAL Assistant** und schreibe
eine Nachricht. Der normale `Chat`-Modus bleibt dein eigenes,
experimentelles NERAL/Helios-Gewicht; beide Modi sind strikt getrennt.

Für einen kleineren Vergleich `qwen3.5:4b` manuell installieren und
`NERAL_ASSISTANT_MODEL=qwen3.5:4b` setzen.

## NERAL Assistant mit einem Doppelklick starten (Windows)

Wenn Python und die NERAL-Projektabhängigkeiten eingerichtet sind, kann die
App ab diesem Branch mit **`Start-NERAL-Assistant.cmd`** gestartet werden.

- Der Starter prüft, ob Ollama erreichbar ist und `qwen3.5:9b` bereits
  installiert wurde. Er lädt **nichts** automatisch herunter.
- Er startet nur die NERAL-App des aktuellen Projektordners und aktiviert
  ausschließlich den separat ausgewiesenen Modus **✦ NERAL Assistant**.
- Er öffnet automatisch `http://127.0.0.1:9680` im Browser.
- Falls eine alte NERAL-Version bereits Port 9680 belegt, meldet er diesen
  Konflikt und startet **nicht** die falsche Version. Die alte App schließen
  und den Starter erneut ausführen; persönliche Daten nicht löschen.
- Die Eigenmodelllinie Arctus/NERAL bleibt unverändert.

**Wichtig:** Du musst den Projektstand aus dem Entwicklungsbranch
`feature/neral-assistant-local-reasoning-20261027` benutzen, nicht die
ältere Hauptversion. Sollten Python-Module fehlen, musst du einmalig
im passenden Python-Environment `python -m pip install -r requirements.txt`
und CPU-PyTorch installieren:

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Diese Installation ist **separat vom Ollama-Modell** und nicht automatisch.
Eine normale NERAL-Assistant-Antwort kommt im Chatfenster, nicht im Terminal.
Aktuell erfolgt die Ausgabe nach vollständiger Berechnung und nicht
Token für Token. Der Prototyp ist ausschließlich lokal unter localhost
erreichbar. Eine öffentliche Web-App benötigt Authentifizierung, Server,
Datenschutz- und Abuse-Schutz.

## Reproduzierbarer erster Qualitätsvergleich

Sobald Ollama läuft und das Modell installiert ist, können wir NERAL bei
festen Kurzaufgaben messen. Die zwölf lokalen Diagnosefälle umfassen
Deutsch/Englisch, mathematische Aufgaben, Kontext, Code, Naturwissenschaften,
Anweisungsbefolgung und das Eingestehen fehlenden Wissens.

```powershell
python neral_reasoning_eval.py --model qwen3.5:9b --mode both --out local_state/neral_9b_diagnostic.json
```

Der Bericht trennt **native Modellantworten ohne Werkzeuge** von
**Modellantworten mit zugelassenem Rechner**. Beide Varianten verwenden
dieselben eingefrorenen Fälle, dieselbe Modellbasis und die gleiche
Decoding-Konfiguration. Case-SHA-256, Antworten und Werkzeugaufrufe werden
protokolliert. `--mode native` oder `--mode tools` erlaubt getrennte Läufe.

Die Fallzahl ist absichtlich klein: Dieses Ergebnis ist eine
**technische Entwicklungsdiagnose, kein unabhängiger Benchmark und keine
Prognose der allgemeinen KI-Zuverlässigkeit**. Ein Vergleich mit Giotto,
GPQA, AIME, HLE oder ChatGPT wäre damit nicht zulässig.

## Sicherheit und fachliche Grenzen

- **Explizites Opt-in**: Ohne `NERAL_ASSISTANT_ENABLED=1` erscheint kein
  aktivierter Assistant. Kein stilles Umschalten vom Arctus- oder Helios-Modell.
- **Nur localhost**: Ollama-Anfragen gehen fest an `127.0.0.1:11434`;
  keine Ollama-Proxys, Redirects, externen APIs oder Cloud-Inferenz.
- **Kein automatischer Download oder Training**: Start und Pull sind manuell.
- **Keine freie Codeausführung**: Das Modell kann nur ein eingeschränktes,
  deterministisches Rechenwerkzeug mit prüfbaren mathematischen Ausdrücken
  anfordern. Keine Shell, Dateizugriffe, Websuche oder externen Plugins.
- **Trennscharfe Kennzeichnung**: NERAL zeigt Qwen3.5 als Modellbasis und
  Rechenwerkzeuge separat an. Tool-Genauigkeit ist keine eigene Modellleistung.
- **Speicher und Kontext**: Maximal 4096 UTF-8-Bytes an Nutzer- und
  Verlaufstext sowie sechs Gesprächspaare. Keine Garantie umfassender
  faktischer Richtigkeit oder Sicherheit.
- **Non-streaming**: Dieser erste Pilot antwortet als Ganzes. Die
  bestehende Abbrechen-Schaltfläche ist in diesem Modus ausgeblendet;
  Streaming/Abbruch bedürfen eigener Implementierung.

## Nachweise

- PR #129: https://github.com/Cola968/neral-helios/pull/129
- RTX-Diagnose (nach erneutem Lauf: Ollama + 9B verfügbar, Antwort `BER`):
  https://github.com/Cola968/neral-helios/actions/runs/40577556815
- Getrennte CPU-Tests: `python -m unittest tests.test_neral_assistant -v`
- Vollständige Helios-App-Tests: `.github/workflows/helios-app-ci.yml`

**Nächster Gate:** Ollama manuell installieren, Qwen3.5-9B lokal laden,
dann denselben festen Satz aus Deutsch, Mathematik, Wissenschaft, Code,
Unsicherheit, mehrstufigem Kontext sowohl im nativen Modellmodus als auch
mit zugelassenen Werkzeugen testen. Berichte als separate Messwerte
dokumentieren. Keine Gewichte trainieren oder automatisch veröffentlichen.
