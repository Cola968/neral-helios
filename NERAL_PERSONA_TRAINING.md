# NERAL-P1: eigene Persönlichkeit, Training-Daten und überprüfbare Baseline

Stand: 28.10.2026. **Noch KEIN Training gestartet; noch KEINE gemessene Qualitätssteigerung.**

## Was ist bereits umgesetzt?

- 39 eigens erzeugte deutsch- und mehrsprachige Trainings**entwürfe** in `data/neral_persona_seed_v1.jsonl`: Identität, verlässliche Unsicherheitsangaben, Coding, Logik, Sprache, Grenzen, Gesprächsstil und Kreativität.
- `neral_persona_data.py`: Python-only, reproduzierbarer JSONL-Export und SHA-256-Manifest für LoRA-SFT. **`approved_for_training=false` und `training_executed=false` bleiben bewusst gesetzt**, bis eine Person Inhalte und Modellstrategie geprüft hat.
- 16 separate, eingefrorene, original erstellte kleine Nachfragen in `eval/neral_persona_holdout_v1.jsonl`, ausdrücklich **nicht** als Training zu verwenden.
- `neral_persona_eval.py`: Score nur bei vollständigen, korrekt gelabelten Antworten; keine Vermischung von Qwen, Arctus und Werkzeugbenutzung. Keine NIS2-Messung.
- `neral_persona_ollama_runner.py`: Nur nach manueller Ausführung mit laufendem Ollama auf `127.0.0.1:11434`; überprüft Modell-Digest aus `/api/tags`, fixiert Inferenzparameter, schaltet Tools/Reasoning-Ausgabe aus und speichert Antworten zusammen mit den Metadaten in einer lokalen Datei.
- Echte Auswertung nur für `mode=local_ollama_http`; eingespritzte CI-Testtransporte werden als `simulated_test_transport` markiert und nicht gewertet.
- Kein Cloudanbieter, keine tokenbasierten Zahlungen, keine Anforderung an GPU im CI; kein Upload privater Gesprächsdaten.

**Ehrlichkeitsgrenze:** Ollama-Logs und JSON-Dateien sind keine kryptografische Hardware-Attestierung. Die wenigen exakten Testfragen messen Formate, Grundwissen, Arithmetik und einfache Logik – **nicht** eine vollständige KI-Intelligenz, Coding-Fähigkeit oder die subtile NERAL-Persönlichkeit. Insbesondere sind daraus weder ein NIS2-Wert noch eine Aussage \"NERAL ist klüger als Qwen\" zulässig.

## Jetzt ohne PC: Reproduzierbare CI-Tests

```bash
python -m unittest tests.test_neral_persona_data tests.test_neral_persona_eval tests.test_neral_persona_ollama_runner tests.test_neral_persona_recorded_run -v
```

GitHub Actions übernimmt dies ohne API-Token, Inferenz und Training. Im Repo ist die eigenständige Entwicklungsbranch `feature/neral-persona-training-data-20261027`.

## Beim nächsten Start deiner RTX 4070 (Windows / PowerShell)

Wechsle auf den angegebenen Entwicklungsbranch; Python 3.12 ist ausreichend für den Datenexport und die Test-CLI. **Die GPU-Treiber und Ollama müssen für die echte Baseline laufen.** Das ist keine Trainingseinrichtung.

```powershell
New-Item -ItemType Directory -Force local_state\neral_persona
python neral_persona_data.py --train local_state\neral_persona\train_draft.jsonl --manifest local_state\neral_persona\train_draft_manifest.json
ollama list
python neral_persona_ollama_runner.py --model qwen3.5:9b --output local_state\neral_persona\baseline_qwen9b.json
python neral_persona_eval.py --predictions local_state\neral_persona\baseline_qwen9b.json
```

Der Lauf ist **ausschließlich lokale Inferenz**; er verändert kein Modell. Es werden echte Antworten von deinem installierten `qwen3.5:9b` verwendet, nicht simulierte Musterantworten. Ein fehlendes Modell oder nicht erreichbares Ollama führt zu einem Fehler – es gibt keinen stillen Cloud-Fallback.

Hinweis: Ausgaben unter `local_state/` können private Modellantworten enthalten und sollten **nicht** automatisch auf GitHub veröffentlicht werden. Der Export überschreibt vorhandene Dateien nicht. Für einen erneuten Durchlauf neuen Dateinamen wählen.

Für einen fairen Vorher-Nachher-Vergleich der **4B-Modellgewichte** muss zunächst auch die echte Baseline mit *dem gleichen* `qwen3.5:4b` und derselben Evaluierung gemessen werden. Das vorhandene 9B ist ein separater Produktvergleich und kein fairer Vorher/Nachher-Effekt eines 4B-Trainings.

## NERAL-P1-Persönlichkeit getrennt beurteilen (ohne automatische Selbstbenotung)

In `eval/neral_persona_style_holdout_v1.jsonl` liegen **16 weitere, strikt
separate Aufgaben**, die speziell den persönlichen Charakter testen:
Identität, klare Unsicherheit, Coding-Vorgehen, Eigeninitiative,
respektvoller Ton, Sicherheit, Transparenz und Mehrsprachigkeit. Es sind
original erzeugte, noch nicht extern geprüfte Entwürfe – **nicht** aus
dem Training kopieren oder als SFT-Beispiele verwenden.

Für jede Frage zunächst eine **echte Modellantwort** aufzeichnen (die
Standard-Ollama-Routine für die 16 *exakten* Smoke-Tests führt diesen
qualitativen Test noch NICHT automatisch aus). Die Antworten als JSON-Objekt
`{"nstyle-001": "Antwort...", ...}` mit allen 16 Kennungen ablegen.
Danach:

```powershell
python neral_persona_style_review.py prepare --answers local_state\\neral_persona\\style_answers.json --model qwen3.5:9b --output local_state\\neral_persona\\style_review.json
```

Öffne die erzeugte Datei und bewerte **manuell** für jede Antwort anhand
des zugehörigen `rubric`-Textes: `rating=0` = verfehlt,
`rating=1` = teilweise, `rating=2` = überzeugt. Setze dazu
`reviewer="human"` und ergänze begründende Notizen. Ein leeres Feld
bleibt ausdrücklich **unbewertet**, nicht 0 Punkte.

```powershell
python neral_persona_style_review.py score --sheet local_state\\neral_persona\\style_review.json
```

Die Bewertungsroutine lehnt fehlende Fälle, fremde Kategorien,
sich widersprechende Rubriken sowie `reviewer="llm"` ab. Ein
vollständiger Human-Report wird als **`human_reported_unverified`**
gekennzeichnet: Selbst eingetragene menschliche Bewertungen sind
keine unabhängige Identitäts- oder Modell-Attestierung.
Für einen echten Vergleich müssten die Reviewer blind und unabhängig
arbeiten, dieselben Fragen und Parameter sehen und die Rohantworten
aufbewahrt werden. Kein NIS2.

## RTX 4070 und die spätere 4B-LoRA-Pilotphase

Nach offiziellen Unsloth-Angaben: **Qwen3.5-4B BF16-LoRA etwa 10 GB VRAM**, Qwen3.5-9B BF16-LoRA etwa 22 GB. Die RTX 4070 mit 12 GB liegt beim 4B-Modell knapp über dem Richtwert, aber reale Speicherbelegung kann höher ausfallen. Training mit Batch 1, kurzer Sequenzlänge (zum Beispiel 512–1024 Tokens), Gradient-Checkpointing und rechtzeitigem Checkpoint-Speichern planen. **Unsloth empfiehlt für Qwen3.5 derzeit kein 4-Bit-QLoRA** wegen erhöhter Quantisierungsabweichungen.

Quelle: https://unsloth.ai/docs/models/qwen3.5/fine-tune

Der kleine Seed ist **noch keine gute Trainingsbasis**. Vor einem echten LoRA-Experiment müssen wir:
1. Alle Entwürfe sprachlich/faktisch/lizenzrechtlich prüfen, Trainingsfreigabe explizit dokumentieren.
2. Weitere diverse, eigenständige Dialoge mit nachvollziehbaren Coding- und Mehr-Turn-Szenarien ergänzen.
3. Eine **separate menschlich bewertete, blinde NERAL-Persönlichkeits-Evaluation** aufbauen (nicht aus Trainingsbeispielen).
4. Exakte Modellrevision/Chat-Template, PyTorch/Transformers-v5/Unsloth und GPU-Speicher mit einem kleinen Smoke-Test verifizieren.
5. Erst nach lokaler Baseline den 4B-LoRA-Pilot **ausdrücklich** freigeben. Keine selbstständigen Downloads, Installationen oder kostenpflichtigen Jobs.

Die Benutzergrenze bleibt **25 € direkte zusätzliche Projektausgaben gesamt, ohne Haushaltsstrom**. Neue Ausgaben bleiben 0 € ohne Freigabe. Ein lokaler Pilotlauf ist daher der bevorzugte Weg.

## Künftige echte Modellvergleiche

- Raw-Baseline Qwen3.5-4B vs. **NERAL-P1-4B-LoRA** auf demselben eingefrorenen Testsatz, identischen Parametern, mehreren Durchläufen und dokumentierter Modellrevision.
- Separate qualitative menschliche Bewertungen für natürliche deutsche Sprache, ehrliche Unsicherheit, eigenständige Art und Codequalität.
- Kein Versprechen, dass Fine-Tuning die Faktenqualität verbessert; bei Verschlechterung Training überarbeiten bzw. Adapter nicht ausrollen.
- Arctus **getrennt** evaluieren: sein NIS2 bleibt bis zu vollständigen Rohmodell-Evidenzen ungemessen.

Die beiden bestehenden Persönlichkeits-Integrationen liegen in den anderen Draft-PRs: **NERAL lokal #129**, **NERAL Cloud #131**. Diese Branch bereitet Daten und Tests für ein späteres eigenes NERAL-4B-Modell vor.
