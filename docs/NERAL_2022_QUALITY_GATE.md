# NERAL: Zielqualität des ursprünglichen ChatGPT (30.11.2022)

**Status:** eigenständiger Entwicklungs- und Prüf-Gate; **keine** Gleichwertigkeit mit GPT-3.5 behauptet. Keine Änderung der Modellgewichte. Kein PC-/GPU-Lauf. Basis ist der vorhandene gestapelte Qwen-Assistant-Branch #204 → #203 → #175 → #173.

## Produkt- und Forschungsziel sauber trennen

**Pfad A, schnell zum nutzbaren Produkt:** NERAL/Helios mit gekennzeichnetem externem Qwen3.5-9B (lokal in Ollama), vorhandener Chat-/Projektschnittstelle, echtem Rechner als getrenntem Werkzeug, Normal/Deep-Modus. Keine zweite Qwen-Integration bauen. Modellherkunft stets \`external-model\` und \`base_model=qwen3.5:9b\`. Das ist **nicht** selbsttrainiertes Arctus350.

**Pfad B, eigenes Modell:** Eigene Arctus350-Gewichte, originaler 16K-BPE, faire native 1B-/2B-/SFT-Checkpoint-Evaluierung und kontrolliertes SFT auf geprüften Daten wie in Research Issue #179 und Drafts #186/#209. Qwen-Antworten nicht als native Arctus-Accuracy ausgeben. Keine Distillation mit Giotto ohne eigene ausdrückliche Lizenz.

## Messbarer Produkt-Gate: 34 DEV-Fälle

\`neral_2022_cases_v1.json\` enthält **26 deterministisch bewertbare** und **8 nur menschlich bewertbare** Aufgaben auf DE/EN zu Instruktionen, Rechnen, Schlussfolgern, Multi-Turn-Kontext, überprüfbarem Wissen, strukturiertem JSON, Code, Übersetzung und kalibrierter Unsicherheit.

Die automatisch bewertbaren Fälle bestehen auf **exakte, alleinstehende Ausgaben** und blockieren beispielsweise "Antwort 12, aber auch 13" bei einer Ein-Zahl-Anweisung. Kein Wort-Substring-Scoring. JSON muss gültig sein und genau den erwarteten Inhalt haben; doppelte JSON-Schlüssel werden abgewiesen.

Offene Coding-, Übersetzungs-, Kontext- und Kalibrierungsaufgaben benötigen echte menschliche Prüfung. Sie werden niemals als automatischer Erfolg gezählt. Diese sichtbare, kleine, eigene DEV-Suite ist weder ein verdecktes/öffentliches Benchmark noch eine historische ChatGPT-3.5-Reproduktion.

### Sichere, kostenlose Offline-Kontrolle

~~~bash
python -m unittest -v tests.test_neral_2022_gate
python -m neral_2022_gate --plan --cases neral_2022_cases_v1.json
~~~

Standardmäßig findet **überhaupt keine Modellinferenz** statt. Report-Fingerprints hängen vom kompletten Fallkatalog ab und können beim späteren Vergleich geprüft werden.

### Echte lokale Modellmessung – NUR nach PC-Freigabe

Für einen ausdrücklich genehmigten Windows-Test mit bereits installiertem und gestartetem lokalem Ollama und \`qwen3.5:9b\`:

~~~powershell
python -m neral_2022_gate --execute --allow-local-ollama --model qwen3.5:9b --quality standard --tools off --out local_state/neral_2022_native_standard.json
python -m neral_2022_gate --execute --allow-local-ollama --model qwen3.5:9b --quality deep --tools off --out local_state/neral_2022_native_deep.json
python -m neral_2022_gate --execute --allow-local-ollama --model qwen3.5:9b --quality standard --tools on --out local_state/neral_2022_product_tools.json
~~~

Die \`--execute\` + \`--allow-local-ollama\` Doppelzustimmung verhindert versehentliche Modellaufrufe bei Dokumentations- oder CI-Befehlen. Der Runner benutzt ausschließlich \`neral_assistant.assistant_turn\` mit Ollama über festen lokalen Loopback, keine Cloud, keine neuen Modelle, keine Shell oder Trainingsjobs. Der Bericht wird nur als **neue Datei** geschrieben und enthält keine Rohantworten/Prompts, sondern Fall-IDs, Kategorien, Ergebnisse und Antwort-Hashes. Die genannten Beispiele laufen **nicht** automatisch.

Für eine echte **App-Freigabe** ist zusätzlich Browser → Flask → lokales Ollama → Qwen auf dem Nutzersystem als durchgängiger Test erforderlich, einschließlich Persistenz/Herkunft, Fehler-/Offline-Fällen, 12-GB-VRAM-Beobachtung, Antwortlängen und Geschwindigkeit. Ein Green in simulierten Tests ersetzt das nicht. \`deep\` darf nicht automatisch als besser gelten: OOM, Abbrüche, langsame oder leere Antworten müssen sichtbar bleiben.

## Entwicklungsentscheidungen nach Evidenz

1. **Produkt:** Alle 26 maschinenbewerteten DEV-Aufgaben ehrlich berichten, Fehler nach Kategorien priorisieren; mindestens 8 offene Aufgaben menschlich beurteilen. Jede Behauptung von "GPT-3.5-Niveau" ist weiterhin gesperrt; dafür fehlt der historische, identische Referenzlauf. Bei Fehlern unterscheiden zwischen Modell, Runtime, Prompting, Kontext und Tool. 
2. **Qualitätsverbesserung:** Zuerst inkorrekte Provider-/Tool-Kontrakte und Kontext/Format-Fehler reparieren; überaschende Fälle mit RED→GREEN-Regressionen sichern. Nicht an den Arctus-Gewichten drehen, um ein Produktproblem zu lösen.
3. **Eigene Forschung:** Baseline auf unveränderten 1B-/2B-/SFT-Arctus-Checkpoints, danach genau **eine** kontrollierte Daten-/SFT-Ablation pro Run unter getrenntem Budget und explicit PC-Approval.
4. **Release-Entscheidung:** Kein Merge in \`main\`, keine automatischen GPU-/PC-Tests, keine öffentlichen Leistungsversprechen, bis Originalmodellidentität, App-E2E, CPU-Regressionen, manuelle Reviews und Lizenzprüfung vorliegen.

OpenAI-Beschreibung des Originalprodukts: https://openai.com/index/chatgpt/ (30.11.2022).
Upstream Qwen3.5-9B: https://huggingface.co/Qwen/Qwen3.5-9B (Apache-2.0).
