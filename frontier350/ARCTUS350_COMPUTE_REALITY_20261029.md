# Arctus350: Rechenrealität statt unfinanzierter 4T-Tokens-Ziele (29.10.2026)

**Forschungs-Issue:** [#179](https://github.com/Cola968/neral-helios/issues/179). **Abhängigkeit:** Research-Integration [#186](https://github.com/Cola968/neral-helios/pull/186). Dieses Werkzeug ist ein **CPU-only Rechen-/Planungshilfsmittel**, **kein neuronaler Qualitätsfortschritt**.

## Historische echte Basis vs. rechnerische Projektion

Der lokale, native Arctus350-2B-Foundation-Checkpoint ist in [Issue #141](https://github.com/Cola968/neral-helios/issues/141) mit **349.825.152 Parametern** und **2.073.161.216 verarbeiteten Tokens** dokumentiert. **Keine Gewichte wurden für dieses Dokument geladen oder geändert.**

Das neue, rein deterministische `frontier350/arctus350_compute_budget.py` rechnet mit einer **groben, bekannten Trainings-FLOP-Heuristik: `6 × Parameter × Tokens`** für einen dichten autoregressiven Transformer. Die Zahl schließt zusätzliche Kosten (Attention/Context, Padding, optimizer, recompute, I/O, Hardwareeffizienz) nicht zuverlässig ein.

| Vorhaben | Tokens | Tokens/Parameter | Multiplikator gegen dokumentierte 2B-Basis | `6ND` theoretische Operationszahl |
| --- | ---: | ---: | ---: | ---: |
| Dokumentierter Arctus350-2B-Stand | 2.073.161.216 | 5,9263 | 1,0000× | ca. 4,351 × 10^18 |
| Beispiel 7B Tokens (keine Freigabe) | 7.000.000.000 | ca. 20,0 | ca. 3,38× | ca. 1,469 × 10^19 |
| Öffentliche SmolLM2-360M-Trainingsmenge (kein Ziel) | 4.000.000.000.000 | ca. 11.435 | **1.929,4206×** | ca. 8,396 × 10^21 |

**Wichtige Einschränkung:** Tokens/Parameter und theoretische FLOPs ergeben weder intelligente Antworten noch tatsächliche GPU-Stunden oder 12-GB-VRAM-Machbarkeit. Ein größerer geschätzter Rechenaufwand ist keine Qualitätsgarantie, und `6ND` allein ist keine tatsächliche Performance-Messung.

## Neue Forschungs-Hypothese nach Prüfung des SmolLM2-Papers

[SmolLM2: When Smol Goes Big (2025)](https://arxiv.org/html/2502.02737), Abschnitt **6 SmolLM2 135M and 360M**, berichtet etwas für Arctus350 Entscheidendes:

- SmolLM2-360M (~361,8M) bekam etwa **4T** Tokens und wurde mit leistungsfähiger Hardware trainiert. Das ist nur ein **Vergleichsanker**, nicht die Empfehlung für unsere RTX 4070.
- Während die **1,7B**-Version im Bericht auf einen mehrstufigen Datenmix setzt, hätten die **135M/360M**-Varianten laut den Forschern mit einem **einphasigen, durchgehend hochwertigen Mix** besser funktioniert. Sie führten dazu gezielte Ablationen durch.
- Die 350M-Größenklasse darf deshalb nicht unbesehen die Trainingsstrategie eines viel größeren Modells übernehmen.

**Falsifizierbare Hypothese:** Auf **unseren** unveränderten Arctus350-Gewichten könnte ein kleiner, systematisch kuratierter und stabil balancierter DE/EN-SFT-Datenmix gegenüber zunehmend komplexen, sich wiederholenden Mathematik-Kurrikula mehr Generalisierung pro GPU-Step erreichen. Das ist eine Hypothese, **keine gemessene Verbesserung**.

## Nächster sinnvoller Test, ohne den PC anzuschalten

1. Bereits vorhandene Daten- und Bewertungsgates von #180–#192 prüfen, **keine weitere konkurrierende Sampling-Implementierung** einführen.
2. Quellenlizenzen, Trainingslabels, mathematische Aufgaben-/Zahlenleaks, Assistant-Ziel-Tokenrate, EOS-Format und Language/Domain-Balance **nur auf train/val** auditieren. Frozen TEST und NIS2 unangetastet.
3. **Eine einzige** Abweichung vorab registrieren: gleicher 2B-Native-Checkpoint, Datenstand, Seed, Optimizer-Updates, effektive Ziel-Tokens und VAL-Suite; Vergleich 'lang antwortende Beispiele dominieren' vs. 'pro Beispiel normalisieren' (#191). Falls zwei Verfahren verglichen werden, keine zweite Variable (Dataset-Mix) gleichzeitig ändern.
4. Bei Rückkehr des PCs nur nach separater Erlaubnis kurz testen, VAL-Metriken + EOS + Trigram-Repetition + Kosten/Watt-Zeit erfassen. Erst dann mehr GPU-Training erwägen. Eine mechanische 7B-/4T-Fortsetzung **nicht** starten.

## Reproduzierbarer kostenloser Rechner

```bash
python -m frontier350.arctus350_compute_budget --planned-tokens 2073161216
python -m frontier350.arctus350_compute_budget --planned-tokens 4000000000000
python -m unittest tests.test_arctus350_compute_budget -v
```

Der Code erzeugt auf stdout JSON mit klarer Statusangabe `THEORETICAL_ESTIMATE_ONLY`. Er greift **nicht** auf Dateien, GPU, Checkpoints, HF-Jobs oder GitHub zu. Er verweigert ungültige und übergroße Zahlen sowie fremde Parametergrößen. Keine Cloudkosten, kein `main`-Merge, kein Release, keine Modellgewichte. Alle echten Verbesserungen bleiben bis zum vollständigen nativen Abnahmetest **NOT_MEASURED**.
