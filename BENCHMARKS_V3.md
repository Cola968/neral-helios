# Neral Benchmark Suite v3

Ziel: Modellqualität, Sprachmodellierung, Chatverhalten, Robustheit und Laufzeiteffizienz getrennt messen. Ein niedriger Loss oder ein gutes Werkzeugergebnis darf nicht als allgemeine Neral-Fähigkeit ausgegeben werden.

## Warum v3?

Der bisherige 50M-Pilot ist **allgemeines Wikipedia-Pretraining ohne Instruction-Tuning**. Der bestehende `free_generation50.py` testet ihn dagegen mit `User: ... / Assistant:`. Das ist weiterhin als Regressionstest nützlich, vermischt aber zwei Fragen:

1. Hat das Modell allgemeine Sprachstruktur gelernt?
2. Hat das Modell das Chat-/Instruction-Format gelernt?

`benchmark_v3.py` trennt diese Achsen.

## Enthaltene Kategorien

### 1. Base-LM continuation
Freie Fortsetzung normaler deutscher und englischer Textanfänge ohne Chat-Markierungen. Das ist für reine Pretraining-Checkpoints die wichtigste freie Generationsprüfung.

Metriken:
- valides UTF-8
- EOS-Rate
- wiederholte Wort-Trigramme
- eindeutige Ausgaben
- Byte-Tokens/s
- rohe Ausgaben unverändert

### 2. Chat / Instruction
Bewusst getrennte `User: ... / Assistant:`-Prompts. Schlechte Ergebnisse nach reinem Pretraining sind kein Architekturbeweis; sie zeigen fehlendes Instruction-Tuning bzw. fehlende Chatgeneralisierung.

### 3. Conditional continuation preference
Zwei mögliche Fortsetzungen werden mit konditionalem Byte-NLL bewertet. Kategorien:
- deutsche Grammatik
- englische Grammatik
- elementare Fakten
- Commonsense
- Code-Semantik
- einfache Mathematik

Die Kandidaten werden nach **mittlerem NLL pro Byte** verglichen, damit unterschiedliche Kandidatenlängen nicht automatisch längere Antworten bestrafen. Dies ist ein eigener Neral-Diagnosetest und **kein offizieller MMLU/HellaSwag-Score**.

### 4. Echtes Byte-BPB
`pretrain50_data/val_de.bin` und `val_en.bin` werden in festen Fenstern bei 128/256/512/1024 Kontextpositionen bewertet. Nur rohe Byte-Ziele (IDs >= 4) zählen zum Loss; BOS/EOS/PAD/SEP werden ausgeschlossen.

Das beseitigt eine Schwäche älterer Reports, deren `bits_per_token` BOS/EOS einschloss und deshalb kein reines Bits-per-Byte war.

### 5. Kontext-Skalierung
BPB über mehrere Kontextlängen zeigt, ob mehr Kontext messbar hilft. Ein Modell, dessen BPB von 128 auf 1024 überhaupt nicht profitiert, nutzt längeren Kontext möglicherweise noch schlecht.

### 6. Forward-Performance
Auf CUDA werden mehrere Sequenz-/Batchkombinationen nach Warmup mehrfach gemessen:
- mediane Forward-Zeit
- Tokens/s
- Peak allocated VRAM
- Peak reserved VRAM
- OOM

Trainingsdurchsatz und autoregressiver Decode-Durchsatz bleiben getrennte Metriken.

## Bereits vorhandene wichtige Benchmarks

Weiterhin behalten:
- `benchmark.py`: Curriculum-Domain-Loss und feste Rohgenerationen
- `free_generation50.py`: 100 eingefrorene freie Chat-/Robustheits-Prompts
- `pretrain50.py --tune`: RTX-4070 Throughput-Matrix
- `tests/test_regressions.py`: Security, API, SOLVE, Tokenizer und Datenregressionen

## Etablierte externe Benchmarks – nächste Stufe

Diese Tests sollen nur unter ihrem echten Namen reportet werden, wenn der **Originaldatensatz und das korrekte Scoring** verwendet werden. Keine selbst erfundenen Ersatzfragen als „MMLU“ oder „HellaSwag“ bezeichnen.

Empfohlene Reihenfolge für ein kleines 20M/50M-Byte-Modell:

1. **WikiText-2 / allgemeiner LM-Loss** – Sprachmodellierung; wegen unterschiedlicher Tokenizer primär BPB statt Token-Perplexity vergleichen.
2. **LAMBADA** – Langkontext-/Wortabschluss-Diagnostik.
3. **ARC-Easy** und später **ARC-Challenge** – Multiple-Choice Naturwissenschaft/Reasoning.
4. **HellaSwag** – Commonsense-Continuation über konditionale Likelihood.
5. **MMLU** – breites 4-Wahl-Wissen; bei einem 50M-Frühmodell ist ein Ergebnis nahe Zufall erwartbar und trotzdem informativ.
6. **GSM8K** – rohe Modell-Mathematik getrennt von Helios SOLVE. Kein SOLVE-Routing im Modellscore.
7. **HumanEval** – erst wenn Neral zusammenhängenden Code generiert. Ausführung ausschließlich in einer isolierten Sandbox; niemals ungeprüften Modellcode direkt auf dem Host starten.

Für alle externen Benchmarks müssen Dataset-Version, Hash/Revision, Lizenz, Promptformat und Scoring dokumentiert werden.

## Lokaler Aufruf

20M oder 50M:

```powershell
python benchmark_v3.py .\export\neral50m\neral50m.pt --out .\reports\benchmark_v3_50m.json
```

Nur Likelihood/BPB/Forward ohne langsame freie Generierung:

```powershell
python benchmark_v3.py .\export\neral50m\neral50m.pt --out .\reports\benchmark_v3_50m_fast.json --skip-generation
```

Für einen fairen 20M-vs-50M-Vergleich denselben Befehl mit beiden Checkpoints ausführen und **keine Cases nach Sichtung der Ergebnisse ändern**.

## Qualitätsgates für einen zukünftigen 50M-Standard

50M soll 20M nur ablösen, wenn mindestens:
- keine massive Repetitionsregression vorliegt,
- EOS/Antwortabschluss nicht klar schlechter ist,
- Base-LM-BPB besser ist,
- Base-LM continuation klar verständlicher wird,
- Chat nach Instruction-Tuning besser ist,
- Security- und Tooltests bestehen,
- keine Daten-/Test-Leakage festgestellt wird.

Größere Parameterzahl allein ist kein Promotion-Kriterium.

## Bekannter Performance-Hebel

Der aktuelle Browser-Decode berechnet für **jedes neue Byte erneut den gesamten bisherigen Kontext** und besitzt keinen KV-Cache. Das ist korrekt, aber ineffizient. Vor einer größeren Inferenzoptimierung soll ein eigener KV-Cache-Prototyp gegen exakt dieselben Logits und Rohgenerationen geprüft werden.

Auch RoPE `cos/sin` wird aktuell pro Forward neu aufgebaut. Ein positionsabhängiger RoPE-Cache ist ein weiterer messbarer Kandidat. Beide Optimierungen erst nach numerischer Äquivalenzprüfung übernehmen.
