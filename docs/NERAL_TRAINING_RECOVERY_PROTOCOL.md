# NERAL: Evidenz statt blindem Training (Stand 2026-11-08)

## Ausgangsbefund

- **v8 ist kein Update der v7-Gewichte.** `scripts/neral_v8_selfverify_train.py` ruft `_gpu_train` mit dem versiegelten **v5-R16-Adapter als Elterncheckpoint** auf. v7 und v8 sind alternative Forschungskandidaten; eine numerisch höhere Versionsnummer ist kein Qualitätsnachweis.
- v8 verwendet **960 SFT-Beispiele**, darunter **560 v5/v6-Replay**, **240 ältere direkte Logikbeispiele** und **160 neue Logikbeispiele** (100 mit korrigiertem falschem Vorschlag). Die Checkpoint-Auswahl erfolgt nach **Validation Loss**. Diese Kennzahl misst nicht, ob neue, unbekannte Fragen richtig gelöst werden.
- `tokenized_completion` in `scripts/neral_v6_r16_gpu.py` maskiert Prompt-Tokens bereits mit `-100` und prüft die Qwen-Antwortgrenze. Ein angeblich fehlendes *assistant-only loss masking* ist daher **kein durch Codebefunde gestützter Fehler**.
- Der v8-Test mit 32/128 berechneten Aufgaben ist ein **reproduzierbarer Entwickler-Test**, kein unabhängiger, verblindeter Generalisierungsbeleg. Erfolg oder Versagen dort darf niemals als allgemeine Überlegenheit oder Nicht-Verschlechterung behauptet werden.

## Technische Regel, ab jetzt

`scripts/neral_v8_unseen_eval.report_scores` erzeugt `experiment_policy` durch `scripts/neral_v8_experiment_policy.evaluate_experiment`.

1. **REJECT**: Schon eine neue falsche Antwort gegenüber Qwen, v5 oder v7, Verstoß gegen die Retention, fehlender Transfergewinn, EOS-/Tokenbudget-Regression oder Pilot ohne vollständige Prüfung blockiert den Kandidaten.
2. **INDEPENDENT_REVIEW_REQUIRED**: Ein auf allen vorhandenen Entwicklerfällen erfolgreicher Kandidat darf **ausschließlich** in einen zweiten unabhängigen, vorher versiegelten Blindtest gehen. Keine automatische Gewichtsfreigabe.
3. **Kein Holdout-Leak**: Der aggregierte Bericht enthält weder Aufgaben noch Modelltexte oder Aufgaben-IDs. Eine Diagnose darf nur **Fehlerkategorien und aggregierte Zählungen** in einen neuen Trainingsplan übernehmen, niemals die verbrauchten Prüfaufgaben selbst.
4. **Unveränderte Referenz**: Qwen-Basis und alle früheren LoRA-Gewichte bleiben SHA-identifizierbar und unangetastet. Die NERAL-1.0-Local-Preview verwendet derzeit separat eine lokale Qwen3-Ollama-Basis; sie darf nicht behaupten, mit einem nicht freigegebenen v8-LoRA zu laufen.
5. **Keine GPU-Verschwendung**: Erst den Engpass klassifizieren: `EOS_OR_DECODING_REGRESSION` → Decoding testen, **nicht** trainieren; `RETENTION_REGRESSION` → vorherige Fähigkeiten auf separatem Entwicklungsdatensatz analysieren; `NO_TRANSFER_GAIN` → Datenverteilung und Trainingsziel neu prüfen. Jeweils **eine Hypothese, ein isoliertes Experiment und ein neuer Blindtest**.

## Forschung, die unsere nächsten Entscheidungen beeinflusst

Das Problem ist nicht nur unsere Implementierung. Kontrollierte Studien zeigen, dass Supervised Fine-Tuning auf Offline-Korrekturspuren allein oft keine wirksame Selbstkorrektur hervorbringt. **SCoRe** (ICLR 2025) fand einen Verteilungsunterschied zwischen Trainingskorrekturen und tatsächlichen Modellantworten und untersuchte stattdessen On-Policy-Reinforcement-Learning mit Regularisierung:
https://proceedings.iclr.cc/paper_files/paper/2025/hash/871ac99fdc5282d0301934d23945ebaa-Abstract-Conference.html

**S²R** (ACL 2025) verwendet zunächst sorgfältig kuratierte SFT-Verifikationsdaten und verstärkt das Verhalten dann durch Outcome- und Process-Reinforcement-Learning. Die publizierten Zahlen anderer Modelle sind **kein** zu erwartendes NERAL-Ergebnis:
https://aclanthology.org/2025.acl-long.1104/

**Mix-CD Rehearsal** (NAACL 2025) untersucht gezieltes Replay von zuvor korrekten, nach Fine-Tuning aber verlorenen Fähigkeiten statt reinem Zufalls-Replay:
https://aclanthology.org/2025.findings-naacl.138/

**Umsetzung auf RTX 4070 (12 GB):** Zuerst die vorhandene Qwen/v5/v7/v8-Messung vollständig auswerten. Ein künftiger, kleiner Verifier-gestützter On-Policy-Versuch ist eine **Forschungshypothese**, keine autorisierte neue GPU-Trainingskampagne; deren Speicherbedarf, Reward-Korrektheit und Bewertungsunabhängigkeit müssen vor der Ausführung getestet werden.

## Release-Regel

Kein Experiment darf von sich aus `adapter_promoted=true`, `release_validated=true` oder einen Adapterwechsel im laufenden NERAL-Dienst setzen. Auch **null beobachtete Regressionen auf 128 Aufgaben** beweisen nicht, dass auf anderen Aufgaben nie Fehler auftreten. Release 1.0 darf daher bei bewährten lokalen Grundlagen bleiben; experimentelle Gewichte werden nur nach weiterer unabhängiger Abnahme optional bereitgestellt.
