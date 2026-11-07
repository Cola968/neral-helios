# NERAL-EXT – Phase 1: konservativer Offline-Auswerter

Status: **Scoring-Vertrag; kein durchgeführter externer Benchmark.**

NERAL-EXT bewertet separat erzeugte, lokal gespeicherte Antworten von
Qwen und NERAL anhand desselben neuen Testsatzes. Es lädt keine
Modellgewichte, führt keinen generierten Code aus, ruft keine APIs auf
und startet kein GPU-Training.

## Eingaben

Testsatz: UTF-8-JSONL mit 1–500 Aufgaben. Jede Zeile enthält genau
id, domain, kind, prompt, answer. kind muss integer, rational oder eur sein.
Beispiel:
    {"id":"ext-001","domain":"math","kind":"integer","prompt":"Wie viel ist 9+4?","answer":"13"}

Für jedes Modell wird eine eigene lokale JSONL mit exakt denselben IDs
benötigt. Jede Zeile:
    {"id":"ext-001","text":"FINAL_ANSWER: 13","complete":true}

Abgebrochene oder abgeschnittene Antworten erhalten complete:false.
Die getrennte Modellinferenz muss noch mit identischen Prompts,
Token-Limits und Decodingparametern durchgeführt werden. Dies ist
Phase 1 und kein bereits absolvierter Modellvergleich.

Zusätzlich werden der SHA-256 der exakt verwendeten Testdatei, die
40-stellige Revision des Foundation-Snapshots und der 64-stellige
SHA-256 des NERAL-Adapters benötigt.

## CLI-Beispiel (lokale CPU)

    python -B -m scripts.neral_ext --cases C:\privat\ext.jsonl --case-sha256 HASH \
      --base-responses C:\privat\base.jsonl --neral-responses C:\privat\neral.jsonl \
      --base-revision c03e6d358207e414f1eca0bb1891e29f1db0e242 \
      --adapter-sha ADAPTER_SHA --train-jsonl C:\privat\v5-train.jsonl \
      --out C:\privat\ext-summary-neu.json

Dies ist eine Darstellung mit Zeilenfortsetzung für Lesbarkeit; auf
Windows die Argumente in einer einzigen Zeile eingeben.

Der --train-jsonl-Parameter ermöglicht eine Prüfung auf wortbasierte
10-Gramm-Überlappung, ist aber keine Garantie gegen Paraphrasen oder
Pretraining-Kontamination. Für belegte Unabhängigkeit ist ein
separater Quellen- und Daten-Audit notwendig.

## Metriken

strict_correct: Referenzantwort wörtlich getroffen (außer Rand-Leerzeichen).
semantic_correct: exakte mathematische Gleichheit bei eindeutiger
Endantwort. Akzeptiert direkte Einzelwerte oder die letzte explizit
markierte FINAL_ANSWER-/Antwort-Zeile, nicht die letzte Zahl im Text.
ungraded: keine eindeutig prüfbare Antwort, falsche Einheit, mehrdeutig
oder unvollständig. Ungraded ist kein nachgewiesener Rechenfehler.

Pro Domäne und insgesamt: strict, semantic, ungraded, paarweise
Verbesserungen/Regressionen und exakter McNemar-p-Wert.

Im öffentlichen Report werden keine Prompts oder Antworten gespeichert.
Die Kennzeichen external_independence_verified, general_improvement_proven,
release_validated und trained_adapter_promoted bleiben ausdrücklich false.

Für funktionale Coding-Aufgaben fehlt noch die abgesicherte Sandbox-Prüfung.
Die Integration in die Inferenz und echte OOD-Evaluation folgen separat.

## CPU-Test

    python -m unittest discover -s tests -p test_neral_ext.py -v

Der bestehende v5-Benchmark bleibt unverändert.
