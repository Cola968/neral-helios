# Arctus350 – überprüfter Qwen-Teacher-Datensatz, ohne Training

Status: **Draft, reine Offline-Datenvorbereitung. Keine echten Qwen-Antworten erzeugt, keine Modellgewichte verändert.**

## Vorarbeiten wiederverwenden

Die native NeralTalk-JSONL-Pipeline, Assistant-only Loss, EOS- und gepaarte Evaluierung stammen aus Draft #186. Projektgenerierte DE/EN-Grounding-Train/VAL/TEST-Fälle stammen aus #184; semantische Mathe-Split-Prüfungen aus #182. Der zusätzliche Näheduplikat-Audit #208 bleibt ein unabhängiges Gate. Dieser PR implementiert ausschließlich einen sicheren Import **reviewter Teacher-Antworten** in das vorhandene Trainingsformat.

Teacher-Kandidat: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), upstream Modelllizenz Apache-2.0. Qwen im lokalen Ollama zu starten, wurde hier **nicht** versucht; der Importeur nimmt nur vorhandene, manuell überprüfte Daten. Ein SHA-256 des tatsächlich verwendeten **lokalen Artefakts** ist erforderlich. Das ist weder automatisch der Hugging-Face-Commit noch ein Beweis, dass wirklich Qwen generiert hat.

Giotto wird ausdrücklich **nicht** für Arctus-Distillation verwendet: Die [Giotto EULA, Ziffer 4.1(vii)](https://www.giotto.ai/eula), untersagt die Verwendung zur Entwicklung, zum Training oder zur Verbesserung konkurrierender KI-Systeme.

## Eingabe

Eine JSONL-Zeile pro geprüfter Antwort. Nur train/val, nur Modell-native Ausgabe ohne Rechner. Für den ersten Pilot ausschließlich exakte Antworten oder unabhängig nachgerechnete Addition, keine offenen ungeprüften Lehrtexte:

~~~json
{"id":"addition-de-001","split":"train","language":"de","domain":"math","task_type":"addition","prompt":"Berechne 59 + 77.","answer":"136","independent_reference":"136","human_reviewed":true,"teacher_model":"Qwen/Qwen3.5-9B","teacher_artifact_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","teacher_license":"Apache-2.0","generation_mode":"model-native","tools_used":[]}
~~~

Der im Beispiel verwendete Hash ist **nur ein Platzhalter**. Echte Herkunft und Nutzungsrechte am Prompt sind vor Nutzung manuell zu belegen. Der Validator kann eine behauptete menschliche Prüfung oder Modellausführung nicht selbst beweisen.

Der separate TEST-Input muss von den bereits im Projekt vorhandenen deterministischen Generatoren kommen. Deren Antworten werden aus der Quellenlogik neu berechnet. Teacher-Ausgaben kommen **niemals** ins TEST-Set.

~~~bash
python -m frontier350.arctus350_teacher_gate \
  --teacher-jsonl local_state/reviewed-teacher.jsonl \
  --frozen-test-jsonl local_state/generated-grounding/test.jsonl \
  --out local_state/arctus350-teacher-candidate
python -m unittest -v tests.test_arctus350_teacher_gate
~~~

Der neue Ausgabeordner wird nicht überschrieben. Er enthält train/val/test JSONL im nativen Arctus-Format und teacher_audit.json mit SHA-256-Eingangs- und Ausgangsfingerprints, Splitgrößen, transparenter Teacher-Herkunft und explizit **training_approved=false**, **training_started=false**, **native_quality_improved=false**.

## Weitere notwendige Gates

1. Echtes Qwen lokal nur nach gesonderter Freigabe ausführen, Gewichts-/Prompt-Rechte nachweisen und Antworten unabhängig auf Korrektheit prüfen.
2. Den vorhandenen historischen, semantischen und Näheduplikat-Leck-Audit sowie die Original-Tokenizer-/EOS-/Assistant-Masking-Checks aus #186/#208 mit realen Daten durchführen. Der Importeur schützt nur gegen *exakte normalisierte* Prompt-Kollisionen und einige verifizierbare Antwortfehler.
3. In der bestehenden Arctus350-Trainingspipeline einen kleinen kontrollierten SFT-/Replay-Versuch erst nach ausdrücklicher GPU-/PC-Freigabe planen. Nicht als neu erfundenen Trainer duplizieren.
4. Vorher/nachher ausschließlich echte native Arctus-Checkpoint-Antworten auf unveränderten unabhängigen Entwicklungs- und Testaufgaben vergleichen. Keine Giotto- oder Qwen-Antworten als Arctus-Score ausweisen. Kein automatisches Merge/Release.

**Ergebnis:** Datensatz-Vorprüfung statt tatsächlicher Distillation. Der Train/VAL-Pilot ist kein öffentlicher Benchmark, und ein grüner Test beweist keine Steigerung der Intelligenz.
