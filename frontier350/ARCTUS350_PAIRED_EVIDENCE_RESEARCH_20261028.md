# Arctus350 – Forschungsschritt 1: belastbare, gepaarte Modell-Evidenz

**28.10.2026 | Issue [#179](https://github.com/Cola968/neral-helios/issues/179) | Draft [#180](https://github.com/Cola968/neral-helios/pull/180)**

## Forschungsfrage

Können wir mit den vorhandenen **349.825.152 nativ trainierten Arctus350-Parametern** tatsächlich bessere **neue Antworten der Modellgewichte** erzielen, anstatt lediglich bestehende mathematische Werkzeugrouten oder Trainings-Loss-Werte zu verbessern?

**Aktuelle Antwort: noch nicht nachgewiesen.** Ein früherer, echter read-only v3-VAL-Test lieferte **0/32** korrekt; v3 VAL wurde damit zum **Entwicklungsdiagnostikum** und ist kein unberührter Finaltest. Die getrennten Arctus-Checkpoints aus Issue #141 sind weiterhin schreibgeschützt; der neue Code lädt sie nicht.

## Implementiert: Paired native evidence gate

\`frontier350/arctus350_paired_evidence.py\` kann zwei **gleich strukturierte** Beobachtungsdateien mit einzeln gemessenen Fällen vergleichen. Es führt **keine** Inferenz aus. Nur die echte lokale Inferenz kann die beiden Eingabedateien liefern. Somit vermeiden wir falsche Modell-Verbesserungsmeldungen durch synthetische Test-Fixtures.

Pflichtfelder beider Eingaben:
\`\`\`json
{
  "schema": "arctus350-native-case-observations-v1",
  "model_family": "neral-arctus350",
  "evaluation_source": "native-checkpoint-readonly",
  "benchmark_id": "v1-local-dev",
  "benchmark_sha256": "<64 hex - generated from canonical id/category list>",
  "prompt_template_sha256": "<64 hex of the actually frozen prompt specification>",
  "scoring_version": "exact-instruction-v1",
  "tokenizer_sha256": "4a25bb40ad66e2d5a674d32e95bc0076d13cdb4decb4d508226124711e2aa7dd",
  "checkpoint_sha256": "<64 hex of actual local native checkpoint>",
  "sampling": {"temperature": 0, "top_k": 0, "max_new_tokens": 64},
  "cases": [
    {"id": "de-exact-01", "category": "instruction_exact", "correct": true, "eos": true}
  ]
}
\`\`\`

Diese JSON-Darstellung ist **nur ein Schema-Beispiel**, keine vollständige Datei: echte Vergleichsdateien erfordern **8–512 Fälle**, mindestens zwei je Kategorie und exakt denselben Kanon an Fall-IDs. \`benchmark_sha256\` ist der SHA-256 über die UTF-8-Bytes aus \`json.dumps([{"id":..., "category":...} in nach ID sortierter Reihenfolge], sort_keys=True, separators=(",",":"), ensure_ascii=False)\`. **Der Fingerprint allein prüft nicht, ob Prompt-Texte eingefroren sind oder ob Modelle die Fälle im Vortraining sahen.** Ein unabhängiger Run-Manifest-/Output-Validator ist weiterhin erforderlich.

Command (only **read** two JSON files; output is an aggregate printed to stdout):
\`\`\`sh
python -m frontier350.arctus350_paired_evidence \
  --baseline /local/read_only/baseline_observations.json \
  --candidate /local/read_only/candidate_observations.json
\`\`\`

Ergebnisse: Anzahl gepaarter Fälle, absolute korrekte Antworten pro Kategorie, Gewinne, Verluste, korrekt terminierte Antworten (\`eos\`) und eine **Review-Empfehlung**. \`REVIEW_CANDIDATE\` bedeutet ausdrücklich **nicht trainieren, checkpoint befördern oder veröffentlichen**. Wenn eine Kategorie bei korrekten Antworten oder EOS zurückfällt, blockiert der Gate. Unterschiedliche Checkpoints, Tokenizer, Chat-Templates, Sampling-Einstellungen, fehlende Fälle, ungewöhnliche Score-Typen, doppelte JSON-Schlüssel, Rohantworten und unsichere Payloads werden abgelehnt.

**Beweiskraft:** Eingabeangaben sind nicht kryptografisch an die tatsächlichen GPU-Inferenzläufe gebunden. \`provenance_verified=false\` und \`release_authorization=none\` bleiben daher auch bei erfolgreichen Paarvergleichen gesetzt. Die Test-Fixtures im Repository sind synthetisch und enthalten **keine tatsächlichen Modellleistungen**.


### Brücke für echte v3-VAL-Modellbeobachtungen

Zusätzlich implementiert: \`make_v3_carry_run(observations, checkpoint_sha256=..., sampling=...)\`. Es verarbeitet das bestehende **lokal erzeugte** \`arctus350_carry_baseline_audit.py\`-Observationsformat (\`row\`, \`text\`, \`eos\`, \`tokens\`) und prüft die originalen v3-Labels mit dem unabhängigen Quellparser. Nur \`correct/eos\` je Fall werden ausgegeben, plus eine aus den **tatsächlichen Prompt- und Referenztexten** berechnete Prüfsumme; Rohtexte und Tokens bleiben privat im lokalen Inferenzprozess. Die strenge Erfolgsdefinition verlangt **exakte Antwort und EOS-Abschluss**, kein bloßes Erwähnen der richtigen Zahl.

Wichtig: Dies ist **zunächst ein getesteter Adapter für spätere reale Modellläufe**, kein Nachweis, dass ein solcher Lauf jetzt auf einer GPU durchgeführt worden wäre. Der vorhandene 32-Fälle-v3-Test wurde bereits zu Entwicklungsdaten; er ist **nicht** als unberührtes TEST-Set verwendbar. Der exakte Checkpoint-Digest muss vom lokalen System mit den tatsächlichen Gewichtedateien belegt werden.

**CI (GitHub-hosted, ohne CUDA):** [15/15 grüne echte CPU-Tests](https://github.com/Cola968/neral-helios/actions/runs/40783964233) einschließlich bestehender v3-VAL-Auswahl/Scoring-Schutztests und der neuen Bridge-Tests. RED davor: [40783900407](https://github.com/Cola968/neral-helios/actions/runs/40783900407), da die Bridge zunächst nicht implementiert war.

## Hugging Face Referenzprüfung ohne Modell-Download

| Referenz | Relevanz | Lizenz laut HF-Hub-Metadaten |
| --- | --- | --- |
| [SmolLM2-360M](https://huggingface.co/HuggingFaceTB/SmolLM2-360M) | Externes Basismodell ähnlicher Größe: 361,8 Mio. Parameter; **nicht** Arctus-Gewichte | Apache-2.0 |
| [SmolLM2-360M-Instruct](https://huggingface.co/HuggingFaceTB/SmolLM2-360M-Instruct) | Anweisungsoptimierte Variante; spätere **unabhängige** Vergleichsidee | Apache-2.0 |
| [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) | Englischsprachiger, bildungsorientierter großer Textdatensatz; potenzielle Inspirationsquelle für bessere Datenqualität, **nicht ungeprüft herunterladen** | ODC-BY |

Die Metadaten wurden auf Hugging Face abgerufen. Lizenzen müssen vor dem tatsächlichen Training **datensatz- und quellenbezogen** erneut geprüft werden (Web-Scraping, Zuschreibungen, Datenschutz, Heldout-Überlappungen, Version/Revisionspinning). Keine HF-Daten oder Gewichte wurden geladen oder geteilt.

## Nächste zulässige Schritte

1. **Offline und ohne RTX:** existierenden 1B-SAFE-, 2B-Foundation- und SFT-Nachweis vereinheitlichen; rechtlich nutzbare Daten nach Quelle/Split/Revisionshash katalogisieren; reproduzierbare gelockte Satzvorlagen und Bewertungsregeln vorbereiten.
2. **RTX4070 nur nach ausdrücklicher Freigabe und wenn der PC verfügbar ist:** Die gleichen eingefrorenen Aufgaben auf den geschützten Originalgewichten ausführen und die Datenschutz-minimalen binären Fallbeobachtungen erzeugen. Rohtexte und Gewichte bleiben lokal.
3. **Danach:** höchstens ein kleiner, zuvor parametrierter Trainings-Ablationsversuch bei nachweislicher Baseline/VRAM-Sicherheit. Candidate erst vergleichen und menschlich freigeben, wenn keine Kategorie-Regression.

**Budget/Grenzen:** Keine weiteren direkten Zahlungen, keine kostenpflichtigen HF Jobs, kein PC-Remote-Zugriff, keine öffentlich zugänglichen Eval-Leaks, kein automatischer Merge oder Deployment. Github Actions braucht nur übliche kostenlose CPU-CI-Zeit innerhalb vorhandener Kontingente. Das globale zusätzliche direkte Projektbudget bleibt **maximal 25 € insgesamt (ohne Haushaltsstrom)**; nicht automatisch freigegeben.
