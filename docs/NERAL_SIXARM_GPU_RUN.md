# NERAL: echter Prompt-Paritätsvergleich – sechs Arme, nur drei neue GPU-Läufe

**Anlass:** Historische diagnostische Tests verglichen Qwen+3 Few-Shots
gegen NERAL v5/v6.1 mit 0 Few-Shots. Das ist kein isolierter
Gewichtsvorteil. Wir dürfen deswegen aus 19/72 gegen 16/72 keinen
sauberen Schluss über die Effektivität des Adapters ziehen.

## Experiment (lokale RTX4070, kein Training)

Derselbe **bereits verbrauchte**, am 07.11.2026 korrigierte
72er-Entwicklungssatz aus dem erfolgreichen Job
[#42413414400](https://github.com/Cola968/neral-helios/actions/runs/42413414400)
wird mit den exakt gleichen System-/User-Prompts und greedy Decoding
unter zwei Promptbedingungen bewertet:

| Modell | Nullshot | Drei identische Few-Shots |
|---|---|---|
| Qwen2.5-Coder-7B-Instruct | **neu GPU** `qwen_zero` | **Reuse** `qwen_three` |
| Qwen + NERAL-v5-LoRA | **Reuse** `v5_zero` | **neu GPU** `v5_three` |
| Qwen + NERAL-v6.1-LoRA | **Reuse** `v61_zero` | **neu GPU** `v61_three` |

Die alten Rohantworten **bleiben auf deinem PC** und werden nur
wiederverwendet, wenn die 72 Aufgaben exakt mit der
SHA-pinned Generatorversion übereinstimmen, die ursprünglichen
Antworten nach erneuter unabhängiger Bewertung
**19/72, 16/72, 16/72** liefern und die Modellherkunft stimmt.
Kein stilles Übernehmen anderer Runs, keine neuen behaupteten
Leistungswerte aus gerundeten Summen. Die einzelnen Rohdateien
erhalten SHA-256-Einträge im lokalen privaten Manifest.

Die fehlenden drei Antwortenreihen werden nacheinander in separaten
Python-Prozessen generiert. Die 4-Bit-Foundation bleibt identisch;
die LoRA-SHA sind auf den v5-Adapter
`2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5`
und den v6.1-Kandidaten
`33e939f6063d4215807f593816a057e88d8938b30615c6000d6a0bf8c7965279`
fixiert.

Bedingungsgleiche Paare:
- `v5_zero` vs. `qwen_zero`
- `v61_zero` vs. `qwen_zero`
- `v5_three` vs. `qwen_three`
- `v61_three` vs. `qwen_three`
- `v61_zero` vs. `v5_zero`
- `v61_three` vs. `v5_three`

Gemischte Null-/Dreishot-Konfigurationen zählen **nicht** als
Gewichtsüberlegenheit. Summen allein reichen nicht: der
gepaarte McNemar-Explorationswert und genaue Nettoänderungen
werden ausgegeben.

### Reproduzierbar starten

Manuell auf `main`:
`.github/workflows/neral-sixarm-live-gpu.yml`

Im Eingabefeld exakt
`EVALUATE_NERAL_6ARM_PROMPT_PARITY_READONLY`.

Nur die drei neuen Varianten laufen auf der lokalen GPU.
GitHub erhält am Ende ausschließlich den skalaren
`sixarm-summary.json`-Bericht; niemals Prompts,
Rohantworten oder LoRA-Tensoren. Keine Cloud-API-Kosten,
kein Adapter-Finetuning, kein automatischer Merge oder Release.

**Einschränkung:** Der 72er-Datensatz wurde schon benutzt, ist
klein, synthetisch und weder menschlich unabhängig geprüft noch
eine reale Coding-Aufgaben-Suite. Seine sechs Referenzantworten
für Python-Listenaliasing waren zuvor falsch und wurden vor
dem erfolgreichen Referenzlauf #42413414400 repariert.
Der neue Vergleich ist daher nur diagnostisch, nicht
für die Modellfreigabe geeignet.

**Ziel:** Endlich beantworten, ob das LoRA-Gewicht bei identischen
Prompts einen Vorteil bietet und ob die drei Few-Shots
spezifisch die Modelle beeinflussen – ohne die Gewichte erneut zu trainieren.
