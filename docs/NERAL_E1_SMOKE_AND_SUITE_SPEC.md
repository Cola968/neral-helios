# NERAL v6: Smoke-Test + Suite-Spezifikation (Stand 07.11.2026)

Status: **CPU-Verträge vorbereitet; kein neuer Adapter trainiert, keine Releasesuite
anerkannt und kein DPO-/ORPO-GPU-Smoke auf der RTX 4070 ausgeführt.**

## 1. Versiegelte v6-Testsuite und frei nutzbare Dev-Suite

Der bestehende Intake `scripts/neral_v6_suite_intake.py` verlangt pro Suite
600 **reale**, fachlich geprüfte Aufgaben (200 Mathematik, 200 Coding,
200 Logik/Reasoning; mindestens 5 Methodenfamilien je Gebiet).
Die bisherigen 100 synthetischen v6-Holdout-Beispiele und die 400
NERAL-EXT-Diagnoseaufgaben gelten nicht als diese Release-Suite.

Für **jede** Suite drei private Dateien:

- `v6-{dev|release}-600.jsonl`: unverändertes Schema aus PR #339
  (`id, domain, family, source_id, kind, prompt, answer, tests_sha256`).
- `v6-{dev|release}-provenance.jsonl`: **600 Herkunftsdatensätze**
  in exakt der `id`-Zuordnung, mit `source_id, domain, family, language,
  source_class, source, source_revision, source_index, license,
  reference_sha256, tests_sha256`.
- `v6-{dev|release}-review.jsonl`: mindestens **20 dokumentierte,
  positive menschliche Stichproben je Domäne** (mindestens 60/Suite),
  Felder `id, reviewer, reviewed_at_utc, checked_task,
  checked_reference, checked_overlap`. Ein Bool-Wert ersetzt
  keine wirklich ausgeführte unabhängige Prüfung.

Provenienz-Beispiel für **einen** originalen Mathematik-Fall:

```json
{"id":"release-math-000","source_id":"private-original-math-000","domain":"math","family":"rate-composition","language":"de","source_class":"original","source":"independent-review-release","source_revision":"original-human","source_index":0,"license":"author-original","reference_sha256":"<SHA256-der-exakten-UTF8-Referenzantwort>","tests_sha256":null}
```

Der angegebene SHA-Wert muss beim Erstellen **echt berechnet** werden; der
Platzhalter ist kein gültiger Testdatensatz. Bei Coding-Aufgaben referenziert
`tests_sha256` die separat verwahrte Datei mit versteckten Unit-Tests.
Die Herkunftsprüfung berechnet den Hash der Referenzantwort erneut und
kontrolliert Sprache (de/en), Revision, Ursprung/Index-Duplikate,
20 Reviewer-Stichproben je Fachbereich und Übereinstimmung mit dem Suite-JSONL.

**Öffentliche Benchmarks kritisch verwenden:** GSM8K, MATH, MBPP und
HumanEval sind keine automatisch *unbekannten* Aufgaben für Qwen, weil sie
potenziell im ursprünglichen Pretraining vorkamen. Sie eignen sich als
diagnostische Vergleichsaufgaben. Der v6-Release-Claim muss sich dagegen
auf originelle, unabhängig geprüfte Aufgaben und neue Problemkombinationen
stützen. Neu gemischte Parameter und bloße Paraphrasen beweisen das nicht.

Eine Suite darf erst **nach** dem Review und vollständigem Decontamination-
Inventar in der bereits existierenden `GATE.md` vorregistriert werden.
Der Code setzt `suite_frozen=false`, `human_review_independently_verified=false`
und `release_validated=false`; ein Mensch/protected-CI muss die Wahrheit
der Belege, die zeitliche Reihenfolge und den tatsächlichen Schreibschutz
prüfen. Die private Release-Suite **niemals in GitHub committen**.

```powershell
python -m scripts.neral_v6_suite_intake `
  --dev C:\private\v6-dev-600.jsonl `
  --release C:\private\v6-release-600.jsonl `
  --train-jsonl C:\private\v5-train.jsonl `
  --train-jsonl C:\private\v5-val.jsonl `
  --train-jsonl C:\private\v6-train.jsonl `
  --train-jsonl C:\private\v6-val.jsonl `
  --out C:\private\v6-intake-new.json

python -m scripts.neral_v6_provenance `
  --dev C:\private\v6-dev-600.jsonl `
  --dev-provenance C:\private\v6-dev-provenance.jsonl `
  --dev-review C:\private\v6-dev-review.jsonl `
  --release C:\private\v6-release-600.jsonl `
  --release-provenance C:\private\v6-release-provenance.jsonl `
  --release-review C:\private\v6-release-review.jsonl `
  --out C:\private\v6-provenance-new.json
```

**Jedes spätere DPO-/ORPO-Paartraining muss erneut auf die beiden Suiten
und alle bislang verwendeten Trainingsbestände abgeglichen werden.**
Ein unkritischer 10-Gramm-Trefferbericht ist kein semantischer Beweis.

## 2. DPO/ORPO: überprüfbare Trainingspaare (niemals aus der Release-Suite)

Der lokale Smoke erwartet `e1-pairs.jsonl`: je Zeile genau

`id, domain, family, source_id, prompt, chosen, rejected,
chosen_verified, rejected_verified, verifier_sha256, verifier_type, split`.

Anforderungen: `chosen_verified=true`, `rejected_verified=false`,
`chosen != rejected`, `split=train|validation`; 30–200 Trainingspaare
und 10–100 **andere** Validierungspaare, mindestens fünf Fehlerfamilien
je Split, **keine** einzelne Familie >20 %.
`verifier_type`: `numeric_oracle` für `math`, `sandbox_unit` für
`code`, `symbolic_oracle` für `reasoning`.
`verifier_sha256` ist der tatsächliche SHA des externen Verifizierers.
Ein eingetragener Hash allein beweist nicht, dass er tatsächlich lief.

**Keine Failure-Paare aus versiegelten Testaufgaben ins Training nehmen.**
Der Paarbauer muss auf separaten *Trainingsaufgaben* laufen, dort fehlerhafte
Originalantworten von NERAL sammeln, verifizierte Lösungen finden und
anschließend pro Aufgabe gleiches Prompt und unterschiedliche chosen/rejected
nachweisen. `k=6` mit Temperatur 0,6–0,7 ist eine mögliche Mining-
Konfiguration, **kein** automatischer Nachweis. Es gibt noch keinen
freigegebenen Bestand von 100–200 Paaren.

### Unabhängiges Review vor GPU-Ausführung

Der GPU-Modus verlangt zusätzlich `e1-review.json` mit:

```json
{"schema":1,"status":"APPROVED_FOR_LOCAL_GPU_SMOKE_ONLY","pairs_sha256":"<echter SHA256 von e1-pairs.jsonl>","pairs_reviewed":100,"reviewer":"reviewer-id","reviewed_at_utc":"2026-11-07T17:00:00Z","verifiers_independently_executed":true,"suite_exclusion_reviewed":true,"review_evidence_sha256":"<SHA256 eines privaten Reviewprotokolls>"}
```

`pairs_reviewed` muss der tatsächlichen Zahl entsprechen. Der Status
genehmigt **ausschließlich einen 30-Schritte-Experimentallauf**.
Der Code kontrolliert Hashes und Eingabeformen, **nicht** die menschliche
Wahrhaftigkeit der Angaben. Solange Review/Paare fehlen: **nicht starten**.

### CPU-Preflight (jetzt möglich, sobald private Suites und Paare vorliegen)

```powershell
python -m scripts.neral_e1_smoke `
  --mode preflight --method dpo `
  --pairs C:\private\e1-pairs.jsonl `
  --dev-suite C:\private\v6-dev-600.jsonl `
  --release-suite C:\private\v6-release-600.jsonl `
  --out C:\private\e1-dpo-preflight-new.json
```

Mit `--method orpo` denselben Datensatz unter denselben Grenzen prüfen.

### Manuell autorisierter GPU-Smoke (ERST bei freier RTX 4070)

```powershell
python -m scripts.neral_e1_smoke `
  --mode run-local-gpu --method dpo `
  --pairs C:\private\e1-pairs.jsonl `
  --dev-suite C:\private\v6-dev-600.jsonl `
  --release-suite C:\private\v6-release-600.jsonl `
  --review-manifest C:\private\e1-review.json `
  --base C:\private\hf-qwen25-coder-7b `
  --adapter C:\private\v5-r16-480-adapter `
  --out C:\private\e1-dpo-new-experiment `
  --authorize RUN_NERAL_E1_30_STEP_GPU_SMOKE
```

`--method orpo` nutzt dasselbe Input-Manifest, aber **einen anderen
frischen Ausgabeordner**. Niemals beide Prozesse parallel starten.

**Sicherheits- und Speichergrenzen:** Nur vorhandene, vollständig
SHA-verifizierte lokale Qwen-Gewichte und v5-Rank-16-Adapter. Keine
Cloud-Anfragen oder automatischen Downloads. `nvidia-smi`: freie GPU
mindestens 10.000 MiB und Auslastung ≤20 %. **30 Steps, Microbatch 1,
Gradient Accumulation 4, 512 Tokens**, gradient checkpointing, 4-Bit NF4,
paged 8-bit AdamW, kein automatischer Merge, kein Überschreiben des
ursprünglichen v5-Adapters. Bei unzureichendem Headroom oder
unstabilen/nicht endlichen Loss-Werten abbrechen.

DPO und ORPO **nicht verwechseln**: ORPO ist ohne Referenzmodell;
DPO muss für faire v5-Fortsetzung eine eingefrorene Kopie des
**tatsächlichen v5-LoRA-Adapters** als Referenz benutzen. Der Smoke
verweigert bei DPO die Optimierung, wenn die installierte TRL-Version
keinen solchen Referenzadapter bereitstellt. Es gibt keinen garantierten
1–2-h-Lauf und keinen validierten GPU-Smoketest auf Windows. Vor dem
manuellen Einsatz müssen TRL, Transformers, PEFT und bitsandbytes in der
bestehenden CUDA-Python-Umgebung zueinander passen; **keine** automatische
Installation oder Re-Konfiguration durch das Skript.

**Noch kein Leistungsbeweis:** Ein stabiler 30-Step-Smoke qualifiziert nur
den Optimizer-/Speicherpfad. Erst ein eigener Blindvergleich v6 gegen
v5 und die eingefrorene Qwen+Few-Shot-Baseline kann neue Fähigkeiten
nachweisen. Nicht niedrigen Loss als Release-Gate verwenden.

## 3. Reihenfolge und Eigentümer

| Schritt | Owner | Done |
| --- | --- | --- |
| 400er NERAL-EXT | GPU (läuft separat) | Aggregierter numerischer A/B-Gap, keine unbewerteten Python-Fälle |
| Private 600/600 + Quellen + Stichproben | CPU + unabhängiger Reviewer | Zwei vollständige private Suites, geprüfte Oracles, Intake-/Provenienzberichte, vorregistrierter Hash und tatsächlicher Schutz |
| E1 Failure-Mining auf TRAININGSaufgaben | GPU **später** | 100–200 verschiedene, extern verifizierte Paare, Testausschluss |
| E1 30 Steps DPO und ORPO | GPU **manuell** | Beide getrennt von v5, endliche Verluste, dokumentierter VRAM, 30/30 Schritte; kein Release |
| v6 Dev-Gate, dann geschützter Release-Test | GPU + Reviewer | Verifizierte fachliche Überlegenheit; p/CI/Nichtverschlechterung laut `GATE.md` |

Dieser PR installiert keine Trainingsabhängigkeiten, erzeugt keine
vertraulichen Testaufgaben und genehmigt keine Modellversion.
