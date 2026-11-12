# NERAL Qwen3.5 9B QLoRA: unabhängige Quellen- und Zahlenprüfung

**Prüfstand:** 2026-11-12; Quell-Commit `d2f76c13883c654c2ac1ea53bb5a396a14650abd`
auf `codex/neral-qwen35-gpu-validation-20261112`.
**Quellen:** `docs/neral_qwen35_qlora_local_result_20261112.md` sowie
13 Belegdateien in `evidence/reports/qwen35-qlora-20261112/`.

## Unabhängig aus 24 Zeilenpaaren ausgezählt

| Metrik | Identische Q4_K_M-Kontrolle | NERAL-Adapter |
|---|---:|---:|
| Boolesche Aufgaben korrekt | 1/12 | 7/12 |
| Ordnung korrekt | 12/12 | 12/12 |
| Insgesamt korrekt | **13/24** | **19/24** |
| Formal akzeptierte Erstextraktionen | 18/24 | 19/24 |
| Reparaturversuche | 5 | 5 |
| Erfolgreiche Reparaturen | 0 | 0 |

Gepaarte 2×2-Tafel: **13** beide korrekt; **6** nur Adapter korrekt;
**0** nur Kontrolle korrekt; **5** beide falsch.
Gewonnene Aufgaben: `holdout-bool-02`, `-03`, `-06`, `-07`,
`-10`, `-11`. Verbleibende Adapterfehler:
`holdout-bool-04`, `-05`, `-08`, `-09`, `-12`.
Das sind **+6/24 = +25 Prozentpunkte** im identischen lokalen
Extraktions-/Solver-Messverfahren; keine beobachtete gepaarte Regression.

Als rein deskriptiver Test ergibt ein exakter zweiseitiger McNemar-Test
bei 6 Gewinnen/0 Verlusten `p=0.03125`. Die 24 Aufgaben stammen jedoch
aus einem bereits bekannten, synthetischen und thematisch korrelierten
Entwicklungs-Holdout. Der p-Wert ist deshalb **keine** Absicherung der
allgemeinen Produktqualität oder unabhängiger repräsentativer Stichproben.

## Belege und Grenzen

- Training laut `full.json`: 720 Zeilen, 90 Optimierungsschritte,
  `training_seconds=1998.3275706`, NF4+PEFT Rank 8, BF16 und
  `trainable_parameters=21639168`. `artifacts.json` nennt
  `device_peak_used_mib=10331`.
- `data-audit.json`: unveränderter Trainings-SHA256
  `2a6c2219f4c52365be8a3a60bd080877c5d3fecef37371024d8936e31b18ec79`;
  die Generator-CPU-Abnahme reproduziert diesen Hash.
- `checkpoint.json` enthält vier benannte Referenz-Checkpoint-Dateien
  mit Größen und SHA256; `environment-lock.txt` listet installierte
  Komponenten; `export-stages.json` dokumentiert erfolgreiche
  lokale Phasen.
- `gguf-tensor-diff.json` meldet 442 Export-Tensoren, davon 248 geändert
  und 194 unverändert. `artifacts.json` enthält getrennte
  Base-/Adapter-Export-Hashes und exakt sechs gewonnenen IDs.
- `control-holdout-console.txt` und `trained-holdout-console.txt`
  bestätigen im Text dieselben Zähler und vollständige 24/24-Fortschritte.
- Die **rechenbare Konsistenz** dieser Belege und der Original-GitHub-Commit
  wurden unabhängig geprüft. Die lokalen GGUF-Dateien und Gewichtsshards
  stehen im GitHub-Branch bewusst **nicht** zur Verfügung; ihre Byte-Hashes,
  das tatsächliche GPU-Training und die Modellinferenz konnten dadurch
  **nicht nochmals unabhängig ausgeführt oder kryptographisch
  gegengeprüft** werden.
- Ergebnis ist **Solver-gestützte strukturierte Extraktion**, nicht
  werkzeugfreie Mathematik-/Logikleistung. Der PR-#537-Guard war bei diesem
  24er-Durchlauf **nicht** beteiligt.
- Die Hashtrennung schützt vor exakten Dopplungen; sie beweist nicht, dass
  keine semantisch verwandten Problemfamilien im Training vorkommen.

## Nächster Test (separater Evaluationszweig, ohne Training)

`verify/neral-qwen35-generalization-100-20261112` enthält einen
getrennten deterministischen 100er-Transferdatensatz (25 Mathematik,
25 Logik, 25 Python-Codeausgaben, 25 gemischt) mit eingefrorenem SHA256
`ecff6f8e88c2c0f6c1f6a7d2bc77f1c6f0f94dfa047f035f4d5157d458c02560`.
Der neue Messlauf prüft die unveränderten Digests
`ed6a74343db547257a9a78150d647f33fbf25db93d87b8e3d80273b116fa7bb9`
(Kontrolle) und
`9c45488f99cc446821f2008f74a4320a1de58a2f5382a68db889eadc8660dda8`
(Adapter) mit unverändertem Guard-Blob
`b87c594ee54a20b2209b6bf176bc0aacfe256ce8` aus PR #537.

**Freigabe-Einschätzung:** Der 24er-Nachweis ist rechnerisch nachvollziehbar
und belegt einen positiven engen Entwicklungseffekt; für allgemeine
Überlegenheit bzw. einen Produktmerge ist er nicht ausreichend.
