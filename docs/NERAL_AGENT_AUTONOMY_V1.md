# NERAL KI-Agent: lokaler Modell-/Werkzeug-Kreislauf

Stand 06.11.2026. Das ist die zweite Agentenstufe, nicht nur eine statische
Werkzeugliste. Ein lokales Sprachmodell entscheidet in einem begrenzten
Plan-Act-Observe-Zyklus über das nächste Werkzeug. Es bekommt die Beobachtungen
zurück und kann weitere Schritte wählen oder eine Änderung vorschlagen.

## Verwendung in der Helios-App

* NERAL starten wie üblich und **Agent → KI-Agent** öffnen.
* Aufgabe formulieren, **KI-Agent starten**, dann **Automatisch
  weiterarbeiten** oder **Nächste KI-Aktion** drücken.
* Ergebnisse und den Status jeder Modellentscheidung im Agentenprotokoll sehen.
* KI-generierte Änderungen werden nur als Diff vorgeschlagen. Schreiben ist
  ausschließlich nach bewusstem Klick auf **Änderung freigeben** möglich.
  Vor einer Änderung einer bestehenden Datei muss das Modell dieselbe Datei
  zuerst gelesen haben; SHA-256 verhindert das Überschreiben zwischenzeitlicher
  Nutzeränderungen.

Die Werkzeuge können Dateien inventarisieren, lesen und suchen sowie statisch
Python-Syntax, JSON/TOML/CSV, Security-Muster, Markdown-Links und Formatierung
prüfen. Keine Shell, keine Browsersteuerung, keine selbstständigen Downloads,
keine unbestätigten Schreibzugriffe. Das ist noch **kein Funktionsparitäts-
Versprechen** mit dem ChatGPT-Agenten und keine Garantie für gute Patches.

## Konfiguration für lokale KI

Standardmäßig wird **nur** `http://127.0.0.1:11434/api/chat` verwendet,
ohne HTTP-Proxy. Der Ollama-Modellname kann mit
`NERAL_AGENT_OLLAMA_MODEL=qwen3.5:9b` festgelegt werden. Das muss ein
**lokal bereits installiertes Modell** sein. Fehlende Modelle verursachen
einen Fehler; NERAL sendet die Anfrage nicht an eine externe Qwen-API.
Die bloße Nutzung von Qwen per Ollama beweist jedoch **keine** aktive
NERAL-LoRA.

Optional kann eine bereits lokal installierte NERAL-LoRA auf einer
unterstützten CUDA-GPU geladen werden:

```text
NERAL_AGENT_BACKEND=hf_adapter
NERAL_AGENT_BASE_PATH=<lokaler, geprüfter Qwen2.5-Coder-7B-Snapshot>
NERAL_AGENT_ADAPTER_PATH=<lokal trainierter PEFT/LoRA-Adapter>
```

Die HF-Variante lädt 4-Bit-NF4-Gewichte nur lokal, aktiviert
`HF_HUB_OFFLINE` und verlangt BF16 und ausreichenden GPU-Speicher.
Sie liest zunächst nur die Adapterkonfiguration und den Dateihash; erst
bei einer Inferenz wird der Adapter tatsächlich geladen. Die Bezeichnung
`NERAL` setzt einen geladenen lokalen Adapter voraus. Ein erfolgreicher
Load ist **kein** Nachweis hoher Modellqualität.

## Live-Abnahme auf dem Windows-PC

Der Workflow `.github/workflows/neral-agent-live-pc.yml` läuft nur auf dem
verbundenen GitHub-Runner mit Label `neral-gpu`. Er installiert keine Modelle,
startet kein Training und prüft die Steuerung zunächst mit echten Python-
Tests direkt auf Windows. Danach versucht er eine **echte lokale Modellinferenz**
mit bereits installierten Gewichten/Adapter, sonst mit lokalem Ollama.

In einem isolierten temporären Workspace muss das Modell eine Python-Datei
lesen, die Syntax prüfen und die Aufgabe abschließen. Das ist ein enges
Smoke-Testszenario; es führt keinen von der KI geschriebenen Code aus. Ein
bloßer erfolgreicher CPU-Regressionslauf gilt nicht als Modell-Live-Erfolg.
Der Workflow speichert ausschließlich einen zulässigen JSON-Bericht aus
Skalaren; keine Dateien, Prompts, Gewichte oder generierten Ausgaben werden
nach GitHub hochgeladen. Er markiert den echten Modelltest als fehlgeschlagen,
falls kein solches Resultat nachweisbar ist.

## Weitere Grenzen

Auftragszustand liegt nur im Prozessspeicher. Eine Sitzung endet spätestens
nach 8 Modellentscheidungen oder Ablauf des lokalen Zeitbudgets. Das Modell
kann falsche Werkzeugwahl treffen, ungültiges JSON zurückgeben oder auf
einem kleinen lokalen Modell zu langsam/ungenau arbeiten. Die Nutzerfreigabe
gilt je Patch, nicht pauschal für beliebige Aktionen.

Für echte Parität mit einem vollständigen Agenten werden gesondert benötigt:
ein robuster browser-/desktopfähiger, isolierter Werkzeugadapter, sichere
Test-Ausführung, persistent gespeicherte Aufträge, automatische
Fehlerkorrekturschleifen und eine geprüfte Adapter-Evaluation. Diese Punkte
sind hier **nicht als fertig implementiert** deklariert.
