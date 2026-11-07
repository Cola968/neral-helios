# NERAL Release-Gate (verbindlicher Vorschlag für v6 und v7)

**Status: definiertes Prüfverfahren, KEINE Trainingsfreigabe und KEIN bestandenes
Release.** Gilt für neue, getrennte Modellgewichte des 7B-Adapters. Ein schönes
Antwortformat, guter Trainings-Loss oder eine neue Versionsnummer sind keine
Fähigkeitsmessung. Gilt ab dem ersten *vorregistrierten* Release-Versuch; nicht
rückwirkend auf v5 und nicht auf den laufenden NERAL-EXT-400-Diagnosetest.

## Vorregistrierung A und C: Vor jedem Release-Versuch einfrieren

In einem Review-Commit **vor dem Öffnen der noch ungesehenen Testsuite**
folgende Daten und Parameter fixieren:

- Version und eindeutige `attempt_id` (z. B. `v6-release-01`).
- Quelle, Lizenz-/Nutzungsprüfung, Fallzahl, Fachbereich-/Methodenfamilien,
  Datensatz-SHA-256, unabhängige Kontaminationskontrolle und deren SHA.
- Unveränderte Foundation-Revision, Qwen-Few-Shot-Text **einschließlich Reihenfolge
  der Beispiele**, SHA des Baseline-Prompts; NERAL-Adapter-SHA.
- Identischer Systemprompt für beide Gewichtskandidaten, identischer
  Decoder, `temperature=0`, Seed und Tokenlimit (Plan: höchstens 512 neue
  Tokens; abgeschnittene Antworten zählen NICHT als korrekt).
- SHA-256 der ausgeführten Grader- und Statistik-Dateien und aller
  Verifikationsregeln, fester primärer Endpunkt und Schwellen von unten.
- Fixierte Ausführungsumgebung, GPU-/Runtime-Version und Teststart.

Dazu `registration.json` mit genau den im Statistikskript verlangten Feldern
erstellen. `registration_commit_sha` ist die Referenz zu einem **bereits
bestehenden** Review-Commit mit den eingefrorenen Einstellungen; er darf
nicht der SHA desselben Commits sein, der sich selbst referenziert.
Das Skript prüft Dataset- und Statistikskript-SHA **lokal**, aber es kann
**weder GitHub-Chronologie noch einen unabhängigen Review selbst bezeugen**.
Das ist ein separater, unverzichtbarer menschlicher/protected-CI-Gate.

**Kein Ergebnis darf die Release-Schwellen nachträglich ändern.** Ein Update
von Grader, Prompt, Suite oder Definition ist ein neues Pre-Registration-Paket.

## B: Frische einmalige Releasesuiten

Eine v6-Suite enthält **600** vorher ungesehene Aufgaben: je 200 Mathematik,
Coding/Debugging und Logik/Reasoning. Pro Fachbereich mindestens fünf
nachweisbar unterschiedliche Aufgabenmethoden/Familien (nicht nur fünf Seeds
derselben Schablone). Jede Suite nur **ein einziges Mal** zur
Versionsentscheidung öffnen. Ein Fehlversuch verbraucht sie. Entwicklungssuiten
dürfen wiederholt werden, aber nicht in die Releasewertung eingehen.

v7 benötigt zwei getrennte, jeweils 600 Aufgaben große, vorregistrierte
Releasesuiten. Beide müssen einzeln bestehen. Falls der Adapter diese Ziele
zweimal verfehlt, folgen dokumentierte strategische Entscheidungen statt
nachträglich gesenkter Schwellen.

Die Prüfung muss unabhängige Originalaufgaben und für Python isolierte
*hidden* Unit-Tests enthalten. Bloße Textparaphrasen sowie die bisherigen
100 synthetischen v6-Holdouts sind **kein** hinreichender Release-Nachweis.
Ein 10-Gramm-Audit ist notwendig, aber nicht hinreichend, um semantische
Kontamination auszuschließen.

## D und E: Endpunkte, Korrektheitsmetriken und Tests

**Primärer Endpunkt:** Gepaarter Unterschied in *verifizierten korrekten*
Antworten, NERAL minus die vorregistrierte stärkste Qwen+Few-Shot-Baseline,
bei unveränderten Werkzeug- und Tokenbedingungen (reiner Gewichtsvergleich
ohne Tools). Für funktionales Python nur versteckte Unit-Tests in sicherer
Sandbox; nicht bewertbare Fälle sind **niemals** heimlich richtig.

- **Exakter zweiseitiger McNemar-Test** der diskordanten Paare (NERAL-richtig/
  Qwen-falsch versus Qwen-richtig/NERAL-falsch); p ≤ 0,01.
  Beide richtig und beide falsch zählen als Ties und tragen zum Nenner der
  Trefferrate bei, **nicht** zur bedingten McNemar-Verteilung.
- **Gepaartes Newcombe-95-%-Konfidenzintervall (Methode 10)** anhand der
  Marginalanteile und der korrigierten Phi-Korrelation. Dessen Untergrenze
  muss >0 sein. Kein CI für unabhängige Zwei-Stichproben-Proportionen!
- Stratifizierter **Familien-Cluster-Bootstrap** (fester Seed,
  2.000 Durchläufe) als ergänzende Sensitivitätsanalyse: pro Fachbereich
  komplette Methodenfamilien resamplen, nicht Antworten einzeln. Bei
  wenigen Methodenfamilien können auch diese Intervalle instabil sein.
- Für jeden Fachbereich zusätzlich gepaarter Unterschied, 95-%-CI,
  korrekt/strict/format_valid und Verbessert/Regressiert ausweisen.
  Domain-Checks sind gemeinsam verbindliche Nebenbedingungen,
  nicht drei ungeschützte neue Signifikanzbehauptungen.
- *Strict* bewertet das exakte Antwortformat, *format_valid* seine
  syntaktische Erfüllung; beide werden **zusätzlich** berichtet, nicht
  anstelle fachlicher Korrektheit.

Die Regeln für Non-Inferiority bei n=200 pro Domäne sind ambitioniert.
Die Punkt- und CI-Grenzen unten sind bewusst **vor dem Test festgelegte
Projektentscheidungen**, nicht aus dem jeweiligen Datensatz abgeleitet.

| Kriterium | v6 | v7 |
| --- | --- | --- |
| Anzahl | 600 (200 je Fachbereich) | **2 frische** Suiten à 600 |
| Primärer Vorsprung | ≥ **8,0 PP** | ≥ **15,0 PP** je Suite |
| Exakter McNemar | p ≤ 0,01 | p ≤ 0,01 je Suite |
| Gepaartes 95-%-CI | Untergrenze > 0 | Untergrenze > 0 je Suite |
| Fachbereiche mit positiver Punktschätzung | mindestens 2/3 | 3/3 |
| Höchster einzelner Punktverlust | höchstens −3 PP | höchstens −2 PP |
| Fachbereich-95-%-CI-Untergrenze | > −10 PP | > −7 PP |
| Rückgang auf vorherigen Kernfähigkeiten | höchstens 2 PP | höchstens 2 PP |
| Unerlaubte Sicherheitshandlung | 0 | 0 |

Die ersten sechs statistischen Bedingungen sind in
`scripts/neral_release_gate.py` implementiert. **Nicht implementierte
externe Gates** (z. B. Kontamination, früherer Adapter, Audit, zweite v7-Suite)
bleiben externe Sperren. Das Tool meldet daher immer
`release_validated=false`, auch wenn
`statistical_gate_passed=true` ist.

**Zahlenbeispiel:** Bei 400 Paaren und +8 PP sind 32 Fälle netto gewonnen.
Exakt zweiseitig p ≤ 0,01 erfordert hier höchstens 144 diskordante Paare;
die Normalapproximation mit Schwelle 154 wäre zu optimistisch.

## F: Entwicklungszwischenziele (keine Release-Gates)

- v6-dev1: >+3 PP auf einer Entwicklungsuite.
- v6-dev2: >+5 PP auf einer separaten Entwicklungssuite.
- Öffnung der versiegelten Releasesuite nur wenn Entwicklungsmessung
  plausibel über dem +8-PP-Gate liegt, sonst weiterentwickeln.
- **Nie** eine verbrauchte Releasesuite nach Fehlversuch als Training
  oder Hyperparameter-Feedback verwenden.

Diese Zwischenziele sind Messhilfen und dürfen nicht als „v6 bestanden“
beworben werden.

## G: v7-Systemtest strikt getrennt vom Gewichtstest

Erst nach einem erfolgreichen Gewichts-Gate:
- THINK ≥ +5 PP und DEEP ≥ +10 PP gegen NERAL FAST bei fixiertem
  Kontext/Seed/Tokenbudget und gesperrten Testfällen.
- Vergleich zusätzlich gegen **Qwen mit identischen Werkzeugen und
  identischem maximalem Retry-, Zeit- und Verifikationsbudget**.
- Mindestens 70 % des festgelegten Coding-Agenten-Benchmarks mit
  erfolgreichen versteckten Tests; mindestens 50 % der reparierbaren
  Erstfehler in zulässigem Retry-Budget erfolgreich korrigiert.
- Security-Audit mit reproduzierbaren Fail-Closed-Tests:
  **null** Schreiboperationen außerhalb der Sandbox, **null** Fälle,
  in denen ein fehlgeschlagener Verifier als Erfolg gemeldet wird.
- FAST-Median und p95 bei festgelegter RTX-4070-Last, Batchgröße,
  Kontextlänge und Modellversion dokumentieren;
  vereinbarter v6-Geschwindigkeitsvergleich: höchstens +30 %.
- Nur wenn der Toolsystemvergleich den gleichen technischen Aufwand und
  das gleiche Budget erhält, kann eine Differenz dem Modell zugerechnet
  werden. Toolvorteile sind separat auszuweisen.

## Ausführung: lokale CPU-Statistik

Der Auswerter bekommt zwei **private** lokale JSONL-Dateien:

`cases.jsonl`: je Zeile `id`, `domain`, `family`, optional
weitere Felder wie `prompt`, `answer`. Die vollständige Datei muss
über `case_sha256` in der Vorregistrierung eingefroren sein.

`scores.jsonl`: je Fall genau `id`, `domain`, `family`,
`base_correct`, `neral_correct`, `base_strict`, `neral_strict`,
`base_format_valid`, `neral_format_valid`; alle Messwerte echte
Boolesche Werte aus einem **unabhängig prüfbaren Grader**.
Dieses Skript kontrolliert nur die Struktur dieser Werte, nicht die
Original-Korrektheit eines Codes oder die Glaubwürdigkeit des Graders.

```powershell
python -m unittest discover -s tests -p test_neral_release_gate.py -v
python -m scripts.neral_release_gate --cases C:\private\cases.jsonl --scores C:\private\scores.jsonl --registration C:\private\registration.json --out C:\private\aggregate-new.json
```

Vorher `registration.json` aus dem Review-Commit vorbereiten:
`schema=1`, `version=v6|v7`, `attempt_id`,
`registered_at_utc`, `case_sha256`, `scorer_sha256`,
`baseline_prompt_sha256`, `decoding_sha256`,
`foundation_revision`, `adapter_sha256`,
`grader_contract_sha256`, `registration_commit_sha`,
`pre_run_independence_review_sha256`.
Alle Hashes sind hexadezimale SHA-256-Strings, Revisions-/Commit-SHAs
40-stellig. `scorer_sha256` ist der SHA-256 der **tatsächlich auf dem
prüfenden Rechner ausgeführten** `scripts/neral_release_gate.py`.
Das Skript verweigert eine Score-Auswertung bei Hash-Abweichungen,
fehlenden/zusätzlichen IDs, falsch zugeordneten Domänen, unpassendem
Format oder einem bereits bestehenden Report.

**Datenschutz:** Es werden nur aggregierte Ergebnisse exportiert.
Die rohen Aufgaben und Modellantworten niemals öffentlich als
GitHub-Artefakte hochladen. Die Vorregistrierung darf ausschließlich
nicht-sensitive Hashes enthalten.

**Nicht automatisch überprüfbar:** Ob der Registrierungstext wirklich
vor dem Ausführen committet wurde, ob die Suite wirklich frisch ist,
ob die Score-Bools aus einem objektiven Grader stammen und ob v7
eine zweite unabhängige Suite bestanden hat. Dafür braucht es einen
separaten geschützten Audit bzw. Review. Kein Skript kann diese Evidenz
durch den bloßen Abdruck von SHA-Zeichenketten ersetzen.

Referenz zur Methode:
- Newcombe 1998, *Improved confidence intervals for the difference between
  binomial proportions based on paired data*:
  https://www.eiti.uottawa.ca/~nat/Courses/csi5388/Newcombe.1998.pdf
- Paired Wilson/korrektes Phi, StatsDirect:
  https://statisticalhelp.org/help/proportions/paired.htm
- Exakter zweiseitiger McNemar: bedingter Binomialtest, nicht
  asymptotische Chi-Quadrat-Näherung.
