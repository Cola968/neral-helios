# NERAL Mathematik-v2: kleineres Learning Rate, überprüfte Daten, breitere Entwicklungsprüfung

Motiv: Das vorherige Rank-16-Experiment mit 98 historischen Arctus/NeralTalk-Rechenbeispielen und LR 2e-4 erzeugte auf den 16 Entwicklungsaufgaben 0/16 strikt gültige Zahlenantworten. Das neue Experiment soll insbesondere die Formatdegradation verhindern und den alten NERAL-Rank-32-Adapter kontrolliert übertreffen. Ein Gewinn kann nicht vor dem echten Test behauptet werden.

Datensätze (lokal, keine Rohdaten in GitHub):
- 98 originalgetreue Arctus/NeralTalk-Matheaufgaben, SHA256 genau an vorigen Maschinen-Audit und `math_train.review.json` gebunden.
- 86 bereits source-reviewte NERAL-Allgemein-/Codebeispiele, erneut gegen die unveränderte QLoRA-Review-Sperre geprüft.
- 560 neue arithmetische Aufgaben mit ausschließlich numerischer Antwort; acht Familien, Deutsch und Englisch, unabhängige Ganzzahl-/Fraction-Nachrechnung und eindeutige Fragehashes.

Trainingsversuch: separater Rank-16-LoRA-Adapter auf unverändert gepinntem Qwen2.5-Coder-7B-Instruct, LR von 2e-4 auf 5e-5 gesenkt, maximal exakt 120 Optimizer-Schritte, 512 Token, kein Modell-Download, geschützter one-shot-GPU-Lauf nur auf dem eigenen Windows-Runner. Der originale QLoRA-Trainer wird unverändert gepinnt; ausschließlich eine überprüfte Kopie im temporären Runner-Checkout wird in genau einem Lernratenanker angepasst. Früherer Rank-32, der gescheiterte Rank-16 und Qwen-Basisgewichte bleiben unberührt.

Vergleich: Basis-Qwen, ursprünglicher R32-Adapter und neuer R16-v2-Adapter auf 16 eingefrorenen kleinen Entwicklungsaufgaben plus 64 weiteren prozedural generierten Entwicklungsfällen mit eigenem Seed und anderer Formulierung. Striktes Zahlenformat sowie exakte Arithmetik separat ausgewiesen; keine automatische Abweichungskorrektur, keine nachträgliche Änderung der Bewertung. **Dies ist keine unabhängige, blinde, öffentliche Holdout-Evaluation.** Die Testdaten sind vom Generator bekannt; echte allgemeine Verbesserung bleibt unbewiesen.

GitHub Actions: `neral-math-v2.yml`, manuell mit `audit` + `AUDIT_ONLY` oder `train` + `TRAIN_NERAL_MATH_V2_R16_120_STEPS`. `train` nur nach bestandenem CPU-Daten-Audit, pinned Modell-Hash, CUDA-NF4/BF16-Preflight, drei VRAM-Samples alle >= 10000 MiB, <= 15% GPU-Aktivität in mindestens zwei Samples, atomarem einmaligem lokalen Claim. Keine Cloud-GPU und keine Kosten.

Bericht: ausschließlich begrenzte Zähler, Prozente und SHA256. Keine privaten Fragen/Antworten, Modellgewichte, Chatlogs oder Adapter in GitHub-Artifacts. Keine automatische Helios-Aktivierung. Jede Regression blockiert eine spätere Promotion.

Tests: `python -m unittest discover -s tests -p 'test_neral_math_v2.py' -v`.
