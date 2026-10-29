# NERAL Assistant und Arctus350 im gemeinsamen Chat

Der Draft baut auf #175 (`2d0066f`), das wiederum auf der Chatoberfläche aus #173 beruht. Die vorhandene Qwen-Implementierung und die Persona aus #129 (`72fb88d`) sind gezielt übernommen; die neuere Oberfläche und deren Kontext-, Import-, Sicherheits- und Streamingregeln bleiben die Grundlage. Die bestehenden Qwen-Provider- und Startertests aus #129 werden mit ausgeführt. Die Änderungen betreffen Inferenz und Bedienung, nicht das Training oder die Gewichte von Arctus.

## Auswahl und Nutzung

NERAL Assistant nutzt Qwen3.5 als externes Open-Weight-Modell **lokal über den separaten Ollama-Prozess**. Arctus350 und die Byte-Modelle bleiben eigene, getrennte Modellpfade. Qwen ist beim normalen Start deaktiviert. Ist der Assistent ausdrücklich aktiviert, bietet der Chat die Auswahl **NERAL Assistant · Qwen** und **Eigene Modelle**; im Menü der eigenen Modelle ist Arctus350 bei entsprechender Konfiguration als experimentell wählbar.

In einer vorhandenen Windows-PowerShell-Umgebung für den normalen Starter:

```powershell
$env:NERAL_ASSISTANT_ENABLED = '1'
$env:NERAL_ASSISTANT_MODEL = 'qwen3.5:9b'  # alternativ qwen3.5:4b
$env:NERAL_ASSISTANT_THINK = '0'
$env:HELIOS_PREPARATION_MODE = '0'
$env:HELIOS_DEVICE = 'cpu'
.\START_NERAL.bat
```

Ollama muss lokal laufen und das ausgewählte Modell installiert haben. Die App installiert oder lädt keine Modelle nach. Alternativ startet der aus #129 übernommene `Start-NERAL-Assistant.cmd` den vorhandenen 9B-Assistenten explizit. Wenn bereits ein Dienst an Port 9680 läuft, der die gewählte Konfiguration nicht erfüllt, bittet der normale Starter darum, die alte App zu beenden; er startet keinen zweiten Server. Für die zusätzliche Arctus-Auswahl siehe `docs/ARCTUS350_FLASK.md` und dessen lokale Prüfsummen-/Tokenizeranforderungen.

Das Statusfeld prüft ausschließlich Ollamas lokale Modellliste. „Modell installiert“ besagt nicht, dass es bereits im Speicher liegt oder eine Antwort erfolgreich erzeugen wird. Bei fehlendem Ollama oder Modell erscheint eine Fehlermeldung. Qwen-Antworten sind momentan nicht gestreamt und besitzen keinen Stoppknopf; ein Ollama-Aufruf ist zeitlich begrenzt. Der Browser wartet bis zu 270 Sekunden, da der bestehende Assistent höchstens zwei Aufrufe für eine Antwort mit Rechnerwerkzeug zulässt.

## Herkunft und Kontext

Qwen-Antworten tragen `mode=neral-assistant`, `source=assistant:ollama-qwen` und die genaue Modellkennung (`base_model`). Beim erlaubten Rechneraufruf zeigen `tools_used` und `tool_assisted` die Werkzeugbeteiligung an. Modellkennung und Herkunft bleiben beim Speichern und Synchronisieren erhalten. Importierte Angaben werden als unbestätigt angezeigt und nicht als Modellkontext benutzt.

Der Assistent übernimmt nur vollständige Qwen-Nachrichtenpaare desselben konfigurierten Modells. Der Server prüft diese Herkunftsangaben und entfernt sie erst vor dem Aufruf der bestehenden #129-Schnittstelle. Arctus prüft weiterhin seine eigene Checkpoint-Identität; der Byte-Modellpfad lehnt explizit fremden oder importierten Kontext ab. Qwen-Antworten mit Ollama-Abschlussgrund `length` werden als unvollständig markiert und nicht in späteren Kontext übernommen. Diese Metadaten trennen App-Kontext, sind aber kein kryptografischer Nachweis gegen Manipulation durch einen Benutzer mit Zugriff auf lokale Chatdaten. Ollama-Modelltags identifizieren zudem keine unveränderliche Datei: Wird ein Tag lokal ersetzt, erkennt die App das derzeit nicht an einer Prüfsumme.

Die bestehende Transportschicht nutzt ausschließlich `127.0.0.1:11434`, ohne Proxy oder Weiterleitung, begrenzte Antwortgrößen und eine feste Allowlist. Die NERAL-API behält lokale Sitzung, Origin-Prüfung und Limits. Eine gemeinsame Sperre verhindert einen nativen Modellwechsel während einer Qwen-Antwort. Ein Qwen-Wechsel lädt oder entlädt Arctus nicht.

## Validierung und offene Grenze

`.github/workflows/neral-qwen-unified.yml` verwendet einen GitHub-gehosteten CPU-Runner mit leeren Assets und einem isolierten Zustand. Neben den ursprünglichen #129-Tests und den #173/#175-Regressionen prüft er Herkunft und Verlauf, Ollama-Offline-Fehler, Werkzeuge, unvollständige Antworten, lokalen Zugriff und gleichzeitige Modellwechsel. Der Browser-Smoketest startet das echte `app.py` und einen **simulierten** Loopback-Ollama-Server. Es werden keine echten Modellgewichte geladen oder vom Nutzer-PC abgefragt.

Ein früherer direkter Ollama-Test auf der RTX 4070 lieferte `BER`; er ist kein vollständiger Test durch die NERAL-Oberfläche. Ein tatsächlicher Durchlauf über `START_NERAL.bat`, Browser, Ollama und Qwen steht noch aus und erfordert den eingeschalteten, ausdrücklich freigegebenen PC. Laufzeit, Modellqualität und tatsächliche GPU-Nutzung dieser Kombination sind daher nicht bestätigt.

Die Modellbasis ist Qwen3.5 von Qwen, unter Apache-2.0; siehe [Qwen3.5-9B auf Hugging Face](https://huggingface.co/Qwen/Qwen3.5-9B), [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) und [Ollama-Chat-API](https://docs.ollama.com/api/chat). NERAL ist die Produktoberfläche und Persona, kein Anspruch auf Urheberschaft an diesen Modellgewichten.
