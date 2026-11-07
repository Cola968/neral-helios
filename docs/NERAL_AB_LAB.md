# NERAL Lab: Der Unterschied zwischen Basisgewichten und eigenem Adapter

## Ziel und Grenzen

Das neue Panel in **Helios → NERAL 7B** enthält neben dem normalen Chat den
Button **NERAL vs. Qwen vergleichen**. Nur wenn ein echter lokaler, versiegelter
NERAL-LoRA-Adapter konfiguriert ist, steht die Funktion zur Verfügung.

Der Server führt eine Nutzereingabe **zweimal hintereinander auf derselben
lokal geladenen PEFT-Modellinstanz** aus:

1. Foundation (Qwen2.5-Coder-7B), per `PeftModel.disable_adapter()`
2. Foundation mit tatsächlich aktiviertem NERAL-LoRA-Adapter

Beide Aufrufe nutzen **identische Nachrichten, dieselbe neutrale Systemanweisung,
denselben Tokenizer, `do_sample=False`, dasselbe Antwortbudget**. Der Vergleich
verwendet absichtlich **keine** exakte Math-Toolroute, Websuche, Agenten, unterschiedliche
Persönlichkeiten oder spezielle NERAL-Systemprompts. Würden wir diese Werkzeuge
einbeziehen, wäre es ein **Produktvergleich**, kein Vergleich der Modellgewichte.

Es gibt **keinen zweiten lokalen Qwen-Download und keinen Cloud-Inferenzdienst**.
Die Inferenzsperre wird über beide Generationen gehalten: Agenten können nicht
zwischen Foundation- und Adapterdurchlauf schalten. Browsertranskript und
Vergleichsantworten bleiben flüchtig im Browser; rohe Prompts/Outputs gehen nicht
an GitHub.

## Sichtbare Unterschiede richtig lesen

- **Verschiedene Texte:** Der aktive Adapter hat das Ausgabeverhalten bei
  *diesem* Prompt verändert. Daraus folgt **nicht**, dass er besser geworden ist.
- **Gleiche Texte:** Bei *diesem* Prompt besteht kein sichtbarer Unterschied.
  Das beweist nicht, dass beide Modelle überall identisch sind.
- **Nicht verfügbar:** Kein nachweisbar aktiver lokaler NERAL-Adapter mit
  PEFT-Schalter. Ein Ollama-Fallback wird **nicht** als NERAL-LoRA ausgegeben.
- **Fehler:** Kein Fallback auf ein anderes Modell, kein gefälschter Vergleich;
  HTTP 503 ohne interne Dateipfade oder GPU-Fehlermeldungen.

## Wissenschaftliche Weiterarbeit

Ein möglichst unabhängiger NERAL-EXT-Benchmark mit gesperrtem Testsplit muss
künftige Kandidaten gegen das Basisgewicht und den vorherigen NERAL-Adapter
testen. Mindestens zu dokumentieren: Aufgabenherkunft, Antwortschlüssel,
Kategorien, korrekt/falsch, verschlechterte und verbesserte Fälle,
Adapter-SHA, Basismodellrevision und Decoding-Einstellungen.

Das A/B-Labor trainiert **keine Gewichte** und bestätigt **keinen nachgewiesenen
Qualitätsvorsprung**. Erst nachdem eine valide Auswertung vorliegt, darf
ein Qualitätsvorteil öffentlich beworben werden. Die bestehende
Identity-SFT-Kandidatenpipeline und v5-Training bleiben hiervon getrennt.

## Tests

```shell
python -m pytest -q tests/test_neral_ab_lab.py tests/test_neral_local_chat_api.py
node --check static/neral-local-chat.js
```

Die Tests fälschen nur die GPU-Ausgabe, prüfen aber die echte Adapter-ON/OFF-
Aufrufreihenfolge, gleiche Prompts, Sperren, Fehlerverhalten und HTTP-Schutz.
Reale CUDA/PEFT-Inferenz auf der RTX 4070 muss zusätzlich geprüft werden.
