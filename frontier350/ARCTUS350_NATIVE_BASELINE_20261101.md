# Arctus350: erste native Greedy-Inferenz, 1. November 2026

**Beleg:** [RTX-4070-Run 41307865234](https://github.com/Cola968/neral-helios/actions/runs/41307865234),
erfolgreich. Kein Training, keine Gewichtsänderungen und kein Upload privater Texte.

## Exakter Versuchsaufbau

- Unveränderter Original-Checkpoint bei 2.073.161.216 Trainings-Tokens,
  gegen bekannten SHA-256 **f7a73e283421…cda883e7** geprüft.
- Original-Tokenizer gegen erwarteten SHA-256 geprüft.
- RTX 4070, PyTorch BF16, Greedy-Decoding ohne Hilfsrechner, ohne Qwen,
  ohne Retrieval; höchstens **24 neue Tokens** je Aufgabe.
- Acht eigens für diesen Test formulierte, statische kurze Aufgaben
  (`arithmetic` 3, `instruction` 2, `english` 2, `german` 1).
  Der Aufgabensatz-Hash ist
  `06d2923d64f11e5624956350827950d265332d23d3da5213b4ce427f681a8802`.
  Damit handelt es sich um einen **Smoke-Test**, nicht um ein
  vollständig verblindetes Benchmark-Set.
- Derselbe Checkpoint und dieselben Aufgaben bei identischem
  greedy Decoder einmal als bloße Präfixe (`bare`), einmal im
  genauen Chat-Format `User: …\nAssistant:` (`chat`) ausgeführt.
- Modelltexte wurden ausschließlich im Arbeitsspeicher ausgewertet,
  **nicht** ausgegeben, als Artefakt gespeichert oder hochgeladen.
  Das Tokenizer-Python-Paket wurde nur in einem temporären
  Runner-Verzeichnis bereitgestellt; die bestehende PC-Umgebung
  blieb unverändert.

| Kennzahl | Nur Text | Chat-SFT-Rollenformat |
|---|---:|---:|
| Exakt richtige Ausgaben | **0/8** | **0/8** |
| EOS innerhalb von maximal 24 Tokens | **12,5 %** | **25 %** |
| Ausgegebene Tokens insgesamt | 177 | 169 |
| Gemessene Generierungszeit | 7,640 s | 7,812 s |
| Geschwindigkeit | 23,168 Token/s | 21,633 Token/s |

## Interpretation ohne Schönfärberei

Das korrekte Chat-Rollenformat allein beseitigt den
Instruktions- und Antwortabschlussfehler **nicht**.
Die kleine Probe kann weder semantisch richtige, aber zu
lange Antworten ausschließen noch Aussagen über die Qualität
bei längeren Aufgaben oder anderem Sampling belegen.
Sie beweist insbesondere nicht, dass Training grundsätzlich
wirkungslos wäre oder das Modell keine Grundkenntnisse hat.

Nächster sinnvoller Forschungsversuch: **nur lesende
Fehlerkategorien** (Zahl im Antworttext vorhanden?,
Spezialtokens? EOS oder Längenlimit?, Antwortformat),
anschließend kontrollierte unabhängige Evaluation nach
einem eigenen SFT-Experiment. Dessen Durchführung und
GPU-Verbrauch erfordern gesonderte Freigabe.
Das Modell darf ohne unabhängige Vergleichswerte nicht als
intelligenter oder verbessert dargestellt werden.
