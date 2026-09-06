# Benchmark-Audit – 2026-09-06

## Scope

Geprüft wurden Quellcode, eingefrorene Rohgenerationen, RTX-4070-Throughput-Reports, Testreports und die vorhandene Benchmarkmethodik. Der 50M-Inferenzcheckpoint selbst liegt absichtlich nicht im Git-Repository; deshalb wurden in dieser Audit-Runde **keine neuen 50M-GPU-Generierungen behauptet**. Neue Checkpoint-Benchmarks sind in `benchmark_v3.py` vorbereitet und müssen auf einem Rechner mit installiertem Checkpoint ausgeführt werden.

## Verifizierter Modellaufbau

- 20M: 20.587.008 Parameter
- 40M: 40.051.200 Parameter
- 50M: 50.012.976 Parameter
- 50M: d_model 624, 8 Blöcke, 12 Heads, Head-Dimension 52, SwiGLU 2496
- eigener Byte-Tokenizer, Vokabular 260
- keine Fremdgewichte im dokumentierten 50M-Pilot

Die 50M-Attention verwendet parameterloses Null-Padding der Head-Dimension 52 auf 56 nur innerhalb von SDPA. Die Skala bleibt 1/sqrt(52), anschließend werden die vier Null-Dimensionen abgeschnitten. Der vorhandene CPU-Äquivalenztest vergleicht die Logits mit der ungepaddeten Referenz.

## Vorhandene Testabdeckung

`evidence/reports/pytest_50.xml` dokumentiert:

- **142 Tests**
- **0 Fehler**
- **0 Fehlschläge**
- **0 übersprungene Tests**
- Gesamtlaufzeit etwa 6,562 s

Enthalten sind unter anderem:
- 20M/40M/50M Parameter- und Lineage-Tests
- 50M Padding-Äquivalenz
- Tokenizer-Roundtrip einschließlich Unicode
- Security Allow/Block-Fälle
- Secret-Redaction
- API-Validierung, Origin-/Hostschutz, Rate-Limit, Cancellation
- CASCADE/RIPPLE/LIVING-Projektlogik
- SOLVE-Arithmetik und lineare Gleichungen
- statische Codeprüfung ohne Ausführung
- Split-/Leakage-Regressionen

Der SOLVE-Gridtest prüft intern 21 * 10 * 4 = **840 exakte Arithmetikfälle** für +, -, *, / zusätzlich zu den parametrisierten Einzeltests. Das ist Werkzeugleistung, kein Rohmodellbenchmark.

## 50M Trainingsdurchsatz: gemessene Verbesserung

Vergleich des jeweils ausgewählten 256x32-Falls:

| Messung | vor Attention-Optimierung | nach Optimierung | Delta |
|---|---:|---:|---:|
| Tokens/s | 53.709,24 | 76.613,97 | **+42,65 %** |
| Peak allocated | 4.510.209.024 B | 3.584.221.696 B | **-20,53 %** |
| Peak reserved | 4.831.838.208 B | 3.938.451.456 B | **-18,49 %** |

Bei 1024x8 ist der Unterschied größer:

- 24.570,85 → 70.556,56 Tokens/s
- **2,87x / +187,16 % Durchsatz**
- allocated ungefähr **-53,74 %**
- reserved ungefähr **-50,51 %**

Der 1024x16-Fall stieg im kurzen Benchmark von 3.357,70 auf 67.589,19 Tokens/s. Das ist ein Faktor von rund 20,13. Dieser Extremwert darf nicht als allgemeiner Speedup beworben werden: Der alte Lauf reservierte über 15 GB und befand sich wahrscheinlich in einem ungünstigen Speicher-/Kernelpfad.

## 50M Pretraining-Pilot

Dokumentierter Pilot:

- 999.424 verarbeitete Tokens
- 122 Updates
- 18,38 s gesamte Trainingsschleife inklusive Evaluation/Schreiben
- 13,24 s reine Update-Berechnung
- 75.507 Tokens/s Update-Durchsatz
- 532.480 deutsche / 466.944 englische Tokens
- 0 Padding-Tokens
- Validation 5,6234 → 2,3636
- 50M-Modell wurde frisch initialisiert; kein Instruction-Tuning nach dem Pilot

Wichtig: Das vorbereitete Training enthält 21.915.506 Tokens. Der Pilot nutzte nur rund 4,56 % davon. Ein frisch initialisiertes 50M-Modell nach knapp 1 Mio. Byte-Tokens ist **massiv untertrainiert**. Die schlechte freie Generation beweist daher nicht, dass die 50M-Architektur grundsätzlich ungeeignet ist.

## Freie Generation: 20M gegen 50M

100 eingefrorene Prompts, greedy, maximal 64 Byte-Tokens:

| Metrik | 20M Preview 02 | 50M Pilot A | Einordnung |
|---|---:|---:|---|
| Valides UTF-8 | 100 % | 100 % | Gleichstand |
| EOS-Abschluss | 29 % | 0 % | 50M klar schlechter |
| mittlere wiederholte Trigramme | 2,95 % | 76,62 % | 50M massiv schlechter |
| rohe Modellrechnung | 0/10 | 0/10 | beide unbrauchbar |
| Decode Byte-Tokens/s | 139,23 | 91,24 | 50M ca. 34,47 % langsamer |
| mittlere Antwortzeit | 0,372 s | 0,710 s | 50M ca. 90,74 % höher |

Die Trigramm-Wiederholungsrate des 50M-Piloten ist ungefähr **25,9x** so hoch wie die 20M-Baseline. Typische Rohantworten bestehen aus Schleifen wie `der der der ...` bzw. `the the ...`.

## Methodischer Befund: bisheriger 50M-Chat-Test ist absichtlich hart, aber nicht isoliert

Der 50M-Pilot ist allgemeines Wikipedia-Pretraining. `free_generation50.py` formatiert jedoch sämtliche freie Prompts als:

`User: ...\nAssistant:`

Damit testet er gleichzeitig allgemeine Sprachmodellierung und ein Chatformat, das der 50M-Pilot noch nicht per Instruction-Tuning gelernt hat. Der Test bleibt als Regression wertvoll, darf aber nicht allein entscheiden, ob das **Base-LM** Sprache gelernt hat.

Deshalb ergänzt Benchmark v3:

1. rohe Base-LM-Fortsetzung ohne Chatmarker,
2. konditionale Fortsetzungs-Likelihood,
3. echtes Byte-BPB ohne BOS/EOS-Ziele,
4. Kontextlängenvergleich,
5. getrennten Chat-/Instruction-Test.

## Code-Audit: größter aktueller Inferenzhebel

### Kein KV-Cache

Sowohl `model.generate()` als auch die Browserfunktion `generate_tokens()` führen für **jedes neue Byte** erneut einen vollständigen Forward-Pass über den gesamten bisherigen Kontext aus.

Das ist korrekt, aber für autoregressive Inferenz sehr ineffizient. Bereits berechnete Keys/Values werden nicht wiederverwendet. Mit zunehmender Antwort- und Kontextlänge wird dadurch viel Arbeit wiederholt.

**Priorität nach Sprachqualität:** KV-Cache-Prototyp bauen und gegen die bisherigen Logits/Rohgenerationen auf numerische Äquivalenz testen.

### RoPE wird jedes Mal neu berechnet

`apply_rope()` erzeugt Positionen, inverse Frequenzen, Cosinus und Sinus bei jedem Forward erneut. Ein Cache für RoPE-Werte bis zur konfigurierten Kontextlänge ist ein weiterer plausibler Performancehebel. Auch hier erst messen und numerisch vergleichen.

### RMSNorm rechnet intern FP32

Das ist eine nachvollziehbare Stabilitätsentscheidung. Eine optimierte/fused RMSNorm kann später gebenchmarkt werden, sollte aber nicht ohne Äquivalenz- und Stabilitätstest ersetzt werden.

### Dokumentübergreifende Attention im Pretraining

Die packed Streams enthalten BOS/EOS und 0 Padding, aber Attention kann über Dokumentgrenzen reichen. Das ist nicht automatisch falsch, sollte jedoch als eigenes Experiment gegen boundary-aware Packing verglichen werden. Datenpacking nicht gleichzeitig mit anderen großen Änderungen vermischen.

## Neue Benchmark-v3-Kategorien

`benchmark_v3.py` ergänzt:

- deutsche Grammatik
- englische Grammatik
- Fakten
- Commonsense
- Code-Semantik
- einfache rohe Modellmathematik
- Base-LM continuation DE/EN
- Chat instruction DE/EN
- echtes Byte-BPB DE/EN
- BPB bei 128/256/512/1024 Kontext
- Forward-Tokens/s
- Peak allocated/reserved VRAM
- Repetition
- EOS
- UTF-8
- Output-Diversität

## Etablierte Benchmarks, die als nächste externe Stufe sinnvoll sind

Nur mit Originaldatensätzen und dokumentiertem Scoring reporten:

- WikiText-2 / BPB bzw. Sprachmodell-Loss
- LAMBADA
- ARC-Easy / ARC-Challenge
- HellaSwag
- MMLU
- GSM8K, streng getrennt von SOLVE
- HumanEval erst bei brauchbarer Codegenerierung und ausschließlich in isolierter Sandbox

## Prioritäten nach diesem Audit

1. **Nicht** 100M starten, solange 50M noch kaum trainiert ist.
2. 50M zunächst deutlich länger auf hochwertigem allgemeinem Text pretrainieren.
3. Währenddessen Base-LM-BPB und Base-Fortsetzungen messen.
4. Erst wenn Base-LM verständliche Sprache zeigt, Instruction-Tuning hinzufügen.
5. Freie Chatgeneration danach erneut messen.
6. KV-Cache/RoPE-Cache als separate Inferenzoptimierung benchmarken.
7. 20M bleibt Standard, bis 50M die Qualitätsgates tatsächlich schlägt.

## Was diese Audit-Runde nicht behauptet

- keine neue RTX-4070-Messung ohne lokale GPU
- keine neue 50M-Rohgeneration ohne lokalen Checkpoint
- kein offizieller MMLU/HellaSwag/GSM8K/HumanEval-Score
- keine Aussage, dass SOLVE-Leistung Neral-Modellleistung sei
