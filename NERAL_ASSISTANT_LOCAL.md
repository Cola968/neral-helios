# NERAL Assistant – lokaler 9B-Pilot (RTX 4070)

**Status:** Der Code für NERAL Assistant und die eigene App-Oberfläche ist als
Entwicklungs-Pull-Request vorhanden. **Kein Leistungsversprechen und keine
Produktfreigabe.** Der Runner hat am 27.10.2026 die RTX 4070 (12.282 MiB)
erkannt, aber keine Ollama-API auf `127.0.0.1:11434` erreicht. Daher wurde
bislang **kein nativer Qwen3.5-9B-Test** auf der RTX 4070 ausgeführt.

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
- RTX-Diagnose (erreichbarer Runner, kein Ollama):
  https://github.com/Cola968/neral-helios/actions/runs/40577556815
- Getrennte CPU-Tests: `python -m unittest tests.test_neral_assistant -v`
- Vollständige Helios-App-Tests: `.github/workflows/helios-app-ci.yml`

**Nächster Gate:** Ollama manuell installieren, Qwen3.5-9B lokal laden,
dann denselben festen Satz aus Deutsch, Mathematik, Wissenschaft, Code,
Unsicherheit, mehrstufigem Kontext sowohl im nativen Modellmodus als auch
mit zugelassenen Werkzeugen testen. Berichte als separate Messwerte
dokumentieren. Keine Gewichte trainieren oder automatisch veröffentlichen.
