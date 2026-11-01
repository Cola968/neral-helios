# Arctus350 – überprüfter Qwen-Teacher-Datensatz, ohne Training

Status: **Draft, reine Offline-Datenvorbereitung. Keine echten Qwen-Antworten erzeugt, keine Modellgewichte verändert.**

## Vorarbeiten wiederverwenden

Die native NeralTalk-JSONL-Pipeline, Assistant-only Loss, EOS- und gepaarte Evaluierung stammen aus Draft #186. Projektgenerierte DE/EN-Grounding-Train/VAL/TEST-Fälle stammen aus #184; semantische Mathe-Split-Prüfungen aus #182. Der zusätzliche Näheduplikat-Audit #208 bleibt ein unabhängiges Gate. Dieser PR implementiert ausschließlich einen sicheren Import **reviewter Teacher-Antworten** in das vorhandene Trainingsformat.

Teacher-Kandidat: [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B), upstream Modelllizenz Apache-2.0. Qwen im lokalen Ollama zu starten, wurde hier **nicht** versucht; der Importeur nimmt nur vorhandene, manuell überprüfte Daten. Ein SHA-256 des tatsächlich verwendeten **lokalen Artefakts** ist erforderlich. Das ist weder automatisch der Hugging-Face-Commit noch ein Beweis, dass wirklich Qwen generiert hat.

Giotto wird ausdrücklich **nicht** für Arctus-Distillation verwendet: Die [Giotto EULA, Ziffer 4.1(vii)](https://www.giotto.ai/eula), untersagt die Verwendung zur Entwicklung, zum Training oder zur Verbesserung konkurrierender KI-Systeme.

## Eingabe

Eine JSONL-Zeile pro geprüfter Antwort. Nur train/val, nur Modell-native Ausgabe ohne Rechner. Für den ersten Pilot nur nachvollziehbar geprüfte Ausgaben: exakte Antworten (zusätzliche menschliche Inhaltsprüfung nötig), Addition, zweistufige Arithmetik sowie dreigliedrige Ordnungslogik. Noch keine offenen, ungeprüften Lehrtexte:

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


## Schnellerer Lernzyklus statt blinder SFT-Runden

Dieser Importer prüft ab sofort auch unabhängig berechenbare zusammengesetzte Aufgaben:

- DE \`Berechne (59 + 77) - 14.\` → \`122\`, als \`two_step\`.
- EN \`Mara is faster than Lena. Lena is faster than Theo. Who is slowest?\` → \`Theo\`, als \`ordered_logic\`.
- Spiegelung \`Calculate 77 + 59.\` und \`Berechne 59 + 77.\` gilt als **dieselbe** mathematische Aufgabe und ist splitübergreifend verboten.
- Bei dreigliedriger Ordnungslogik werden auch vertauschte Faktensätze mit identischem logischen Gehalt und derselben Frage als Doppelung erkannt.

Das sind streng definierte **Mikroaufgaben**, keine Evaluierung realen Alltagsreasonings. Für die genannten Beispiele braucht man keinen Teacher, weil Referenzantworten lokal vollständig ermittelt werden können: Qwen bringt hier **kein zusätzliches Wissen** ein. Für spätere echte Knowledge Distillation brauchen wir zusätzlich qualitativ überprüfte, lizenzrechtlich zulässige *offene* Lehrerantworten und faire Kontrollgruppen.

Der schnellste glaubwürdige Forschungszyklus bleibt:

1. **Echte lokale Baseline:** Unveränderte 1B-/2B-/SFT-Gewichte auf denselben bisher ungesehenen DEV-Prompts aus Issue #141/#179 messen (EOS, DE/EN, Rechnen, Instruktionsbefolgung, einfache Wissensfragen).
2. **Daten vor Training prüfen:** Vorher vorhandene menschliche SFT- und eigene Curriculumsdaten, Tokenizer-SHA, Assistant-only Targets, EOS und Split-Leaks auditieren; auf Grundlage von #186, #190–#202 und #208, nicht neue konkurrierende Auditoren erfinden.
3. **Nur einen Trainingsfaktor ändern:** Bei separater Freigabe und fixem Budget vergleichender SFT-Pilot (z. B. kompakte Batches vs. bisheriger Sampler, oder ein geprüftes neues Curriculum; nicht gleichzeitig). Baseline/VAL-Quoten und Replay konstant halten.
4. **Entscheidung mit echten Modelloutputs:** Nach dem Pilot dieselben zuvor eingefrorenen Fälle prüfen; bei fehlendem oder instabilem Gewinn **nicht** die Gewichte austauschen. Separate ungezeigte TEST-Aufgaben erst am Ende.

Leistungsziel auf kurze Sicht: eine nachweisbare native Verbesserung gegenüber den eigenen bisherigen Checkpoints; mittelfristig faire Vergleiche zu vergleichbar großen kleinen Open-Weight-Modellen. »GPT-3 erreicht« ist kein Resultat dieses Skripts. Vollständiges GPT-3-175B-Niveau mit 350M Parametern ist kein planbarer, garantierter Trainingsausgang.

**Betrieb:** Keine Ausführung auf dem Nutzer-PC, keine GPU, keine Modelle geladen, keine automatischen Downloads, kein Training und keine kostenpflichtigen Jobs. GitHub-hosted CPU CI testet nur den Import und die Referenzlogik.
