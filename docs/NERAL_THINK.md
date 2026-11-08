# NERAL HIGH – optionaler lokaler Prüf- und Vertiefungsmodus (experimentell)

HIGH (API: /api/neral/chat/think) ist ein lokaler Werkzeug-/Orchestrierungs-Layer, kein neues Modelltraining und kein nachgewiesener Zuwachs an allgemeiner Intelligenz.

## Nutzung

Im Helios-Abschnitt „NERAL AI – lokaler Chat“ HIGH aktivieren. Standardmäßig bleibt LOW aktiv. Der unabhängige NERAL-EXT-A/B-Benchmark bleibt frei von THINK-Werkzeugen, damit er Modellgewichte fair vergleicht.

## Was genau verifiziert wird

Bei einer eigenständigen, vollständig erkannten Rechenaufgabe wie „Löse schrittweise: 2x+3=11“ oder „Berechne schrittweise: 0.1+0.2“ prüft NERAL die numerische Endantwort mit den vorhandenen sicheren AST-Mathematikfunktionen aus neral_agent_math.py. Es gibt kein eval, exec, Shell oder Ausführen generierter Python-Programme.

Bei richtigem Endwert wird verified=true und verification_scope=numeric_final_only gesetzt. Das prüft ausschließlich die numerische Endantwort, nicht alle Erklärungen und Zwischenschritte.

Bei unklarem oder falschem Endwert wird höchstens eine weitere lokale Modellantwort mit einem Korrekturhinweis angefordert. Wenn diese ebenfalls nicht verifiziert werden kann, liefert das deterministische Werkzeug den exakten Endwert. Solche Antworten werden als model_generated=false, tool_used=neral_math_exact und verification_status=tool_fallback ausgewiesen; sie sind keine zusätzlichen KI-Modellfähigkeiten.

Allgemeine Fragen, Codingvorschläge und mehrteilige Gespräche besitzen keine unabhängige semantische Verifikation. Im NERAL-1.0-HTTP-HIGH-Modus wird bei kurzen offenen Antworten ein weiterer lokaler Modellaufruf zur Fehlersuche und Überarbeitung gestartet (deep_review=True). Die Antwort bleibt verified=false und verification_status=not_available. Direkte Python-Aufrufe ohne deep_review=True bleiben abwärtskompatibel.

## Technische Grenzen

- Pro Anfrage höchstens zwei Modellaufrufe. HTTP-HIGH nutzt Qwen3 mit bis zu 4096 Generierungstokens pro Aufruf, Kontextbudget 8192 und nicht-gierigem Sampling. Je Ollama-Aufruf höchstens 90 Sekunden.
- Eine Zeitprüfung vor dem zweiten Aufruf vermeidet unnötige Wiederholungen; sie kann laufende GPU-Generierungen nicht hart unterbrechen. hard_timeout_enforced=false.
- Der Browser kann nach seinem Timeout abbrechen, aber dies stoppt die lokale Backend-Generierung nicht automatisch.
- Die Chat-/Modelllocks schützen innerhalb des lokalen Prozesses. Ein separater GPU-Prozess (z.B. ein GitHub-Actions-Runner) bleibt davon unberührt.
- Keine neue Cloud-API, keine Trainingsdaten, keine Änderung von LoRA-Gewichten. Der Modus ist nicht automatisch in NERAL Mini oder dem Android-Gateway aktiviert.
- Keine persistenten Prompt-, Antwort- oder Trainingslogs durch diesen Layer.

## Schnittstelle

POST /api/neral/chat/think verwendet dieselben validierten Felder wie /api/neral/chat/generate: prompt und optional history. Die bisherigen Host-, Session- und Origin-Schranken der Flask-App gelten weiter.

Die Antwort enthält mode, calls_used, seconds, verified, verification_scope, verification_status, extractor_success, model_generated und die vorhandene Modell-/Adapter-Herkunft.

## Tests ohne GPU

python -m unittest discover -s tests -p test_neral_think.py -v

python -m unittest discover -s tests -p test_neral_think_api.py -v

python -m unittest discover -s tests -p test_neral_think_ui.py -v

node --check static/neral-local-chat.js

Die HTTP-Tests benötigen Flask, aber keine GPU. Echte Latenz und Qualitätsgewinne müssen anschließend separat mit einem ausdrücklich freigegebenen GPU-Lauf gemessen werden.
