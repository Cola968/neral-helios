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
 
## NERAL-EXT: 400 Fälle, Quellen und sichere Reihenfolge

Der Generator `scripts/neral_ext_dataset.py` erzeugt lokal einen
400-Fälle-Datensatz mit A=200 (100 GSM8K-Test, 50 numerische MATH-500-Test,
50 MBPP-Code), B=100 unabhängigen synthetischen Aufgaben (Zinsen,
Einheiten, Syllogismen), C=50 expliziten v5-Style-7-Kontrollen und
D=50 Umformulierungen von A/B. Originalfälle aus A dürfen in D erneut
vorkommen; D ist ein Robustheitstest, kein unabhängig neues Problem.

**Wichtig:** Hugging Face GSM8K / MATH-500 / MBPP sind zwar externe
Benchmarkquellen, aber nicht garantiert ungesehen während des
Foundation-Pretrainings. Python-MBPP-Aufgaben werden bis zur geprüften
Docker-Test-Sandbox im Semantikscore **nicht** bewertet.

1. Installiere `datasets` und `huggingface_hub`, dann erstelle
   ein *neues* lokales Verzeichnis für den Datensatz.
2. Für einen Erstaufbau mit den geprüften, fest im Code gepinnten
   Hugging-Face-Revisionen:

       python -m scripts.neral_ext_dataset --out-dir C:\neral-ext-private

   Der Generator speichert `neral-ext-400.jsonl` und
   `neral-ext-400-manifest.json` mit Quellen-SHAs, Testfall-SHAs und
   Gesamt-SHA. Für vollständige Wiederholbarkeit können die drei
   Manifest-Revisionen statt `--resolve-public-revisions` über
   `--gsm8k-revision`, `--math500-revision` und
   `--mbpp-revision` direkt angegeben werden.

3. Erzeuge die alten v5-Splits rein per CPU:

       python -m scripts.neral_v5_transfer_curriculum --out C:\v5-verified

   Das sind **1.920 Trainings- + 320 Validierungs- + 320 Testfälle**,
   insgesamt 2.560. Beim Kontaminationscheck dürfen keine A/B/D-Fälle
   zehn identische normalisierte Wörter hintereinander mit den v5-Daten
   teilen. C wird als Negativkontrolle separat ausgewiesen.

4. Der GitHub-hosted Workflow
   `.github/workflows/neral-ext-data-ci.yml` prüft die echten öffentlichen
   Quellen, den SHA und die 2.560 Fälle rein auf CPU und veröffentlicht
   nur einen scalar-only Report; keine Prompts, Codeantworten oder Gewichte.
   Dieser Preflight ist **kein Modellvergleich**.

5. Nach erfolgreicher Datensatzprüfung kann
   `scripts/neral_ext_run.py` manuell lokal mit dem verifizierten
   Foundation-Snapshot und dem **versiegelten v5-Adapter** für jeweils
   `base`, `base_fewshot`, `neral_v5` in getrennten Prozessen laufen.
   Der `combine`-Modus erzeugt Format-/Strict-/Semantikscores je
   A/B/C/D, McNemar und den formatbereinigten A+B-Abstand. Modellantworten
   bleiben lokal. Es gibt **keinen automatischen GPU-Workflow**.

6. **THINK/DEEP und neue LoRA-Trainingsläufe werden nicht automatisch
   aktiviert.** Die strenge Reihenfolge lautet: Preflight der Daten,
   Kontaminationsprüfung, kontrollierte A/B/C-GPU-Inferenz, erst danach
   Produktintegration und Profil-Messungen.

### Lokale CPU-Prüfungen

    python -m unittest discover -s tests -p 'test_neral_ext.py' -v
    python -m unittest discover -s tests -p 'test_neral_ext_data.py' -v
    python -m unittest discover -s tests -p 'test_neral_ext_run.py' -v

Ohne externe tatsächliche Scores bleiben
`release_validated` und `general_improvement_proven` auf `false`.

### Verifizierter GitHub-CPU-Preflight (7. November 2026)

- Workflow: https://github.com/Cola968/neral-helios/actions/runs/42392230682
- SHA-256 des **wirklich erzeugten** 400-Fälle-JSONL:
  `47258a911ce0beb96635a34137a634556f364457487fb18a160a892302124cb1`
- Quellenrevisionen: GSM8K `740312add88f781978c0658806c59bc2815b9866`,
  MATH-500 `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`,
  MBPP `4bb6404fdc6cacfda99d4ac4205087b89d32030c`.
- Eingelesene v5-Zeilen: Train 1920, Val 320, Test 320.
- Normalisierte 10-Gramm-Überlappungen: A=0, B=0, C=50 (beabsichtigt), D=0.
- Echte Modellantworten: **noch nicht generiert**. NERAL-/Qwen-Scores:
  **noch nicht vorhanden**.
- Das öffentliche CI erzeugt Fälle nur temporär, veröffentlicht ausschließlich
  Hashes und Mengen. Die gleichen Quellrevisionen reproduzieren sie lokal.
- `--resolve-public-revisions` ist ein **bewusstes Re-Pinning** auf neue
  externe Commits und darf nicht stillschweigend mit dem hier geprüften
  400-Fälle-SHA vermischt werden.

