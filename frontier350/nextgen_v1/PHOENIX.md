# Phoenix – isolierter Vergleich (Entwicklungsstand)

**Status:** Vergleichscode implementiert; keine neuen Modellgewichte trainiert. Dieses Dokument ist kein Beleg für eine Qualitätssteigerung. Der bestehende 350M-SFT-Pilot (`SFT_PILOT.md`) ist eine separate, nur lokal ausführbare und ausdrücklich nicht freigegebene Trainingskomponente. Er benötigt einen vertrauenswürdigen, stabilen Checkpoint, einen dokumentiert geprüften SFT-Datensatz und eine freie RTX 4070. Keine privaten Daten oder Modellgewichte nach GitHub hochladen. Niemals auf aktive Trainingsdateien zeigen.

## Vergleich der freien Generierung

`phoenix_gate.py` ist ein CPU-gestützter **Gesundheitsvergleich, kein Intelligenzbenchmark**. Vorher und nachher müssen dieselben 8–512 unabhängigen Fragen, derselbe unveränderte Decoder, Seed, Chat-Template, Tokenizer, Tokenbudget und dieselbe Messmethode verwendet werden. Die SHA-256 des vollständigen Protokolls muss für beide Läufe identisch sein. Erzeugung und Provenienz der Eingabeberichte werden von diesem Tool **nicht** selbst verifiziert; der Exporter und die Checkpoint-Hashes müssen unabhängig überprüft werden.

Jeder Eingabebericht ist ein lokales JSON-Objekt mit genau diesen Feldern: `schema` (1), `checkpoint_sha256`, `protocol_sha256`, `cases`. Jedes Fallobjekt hat genau `id`, `prompt_sha256` (SHA-256 der Frage unter einem festgehaltenen Normalisierungsverfahren), `answer` (tatsächliche Modellantwort), `finish_reason` (`eos`, `max_tokens`, `stop` oder `error`), `raw_model` (`true`), `tool_calls` (0), `tokens_generated`, `max_tokens`. Eine künstliche Stoppsequenz ist **kein** Modell-EOS. Die vollständigen Eingabedateien bleiben ausschließlich lokal. Ihre Richtigkeit und Originalität muss der Operator vor der Auswertung sichern.

```powershell
python frontier350/nextgen_v1/phoenix_gate.py --baseline PFAD_ZUM_BASELINE_BERICHT.json --candidate PFAD_ZUM_KANDIDATEN_BERICHT.json --out PHOENIX_BERICHT.json
```

Die Ausgabedatei muss neu sein. Ausgabe nur aggregierter Raten und Identitäts-Hashes, keine Rohfragen oder Modellantworten. Exit-Code 2 signalisiert eine Verschlechterung mindestens eines Gesundheitskriteriums. Selbst ein bestandener Gesundheitsvergleich **gibt kein Modell frei**: Blindprüfung der Antwortkorrektheit, vollständiges NIS2, Sicherheit, Lizenz/Datenschutz und reale Laufzeitmessungen bleiben zwingend.

## Tatsächliche Tests und Live-Logs

Lokaler CPU-Vertragstest (keine Inferenz):

```powershell
python -m unittest discover -v -s frontier350/nextgen_v1 -p test_phoenix_gate.py
```

GitHub Actions: `.github/workflows/phoenix-gate-ci.yml`. Ein laufender GitHub-Job liefert seine echten Logs in der Actions-Oberfläche. Diese CI testet ausschließlich die Auswertungssoftware anhand künstlicher Fixtures, nicht das Modell oder die GPU. Für **echte Trainingslogs** muss zuerst ein explizit gestarteter und geprüfter lokaler Trainingslauf existieren; der Vergleichscode simuliert keine Trainingslogs. Keine automatisierte lange Trainingskampagne und keine Promotion.

**Release-Regel:** Das Ausgangsmodell bleibt unverändert. Nur tatsächliche, unabhängig geprüfte Antwortverbesserung ohne Rückschritte kann eine spätere separate Freigabe begründen.
