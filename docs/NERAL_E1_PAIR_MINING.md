# NERAL E1: Pair-Mining aus TRAININGSaufgaben (CPU, kein GPU-Job)

Stand 07.11.2026. Diese Pipeline macht aus **lokal erfassten**
NERAL-v5-Antworten auf die vorhandenen v6-Kompositionsaufgaben
**nur dann** DPO-/ORPO-Kandidaten, wenn die mathematische Endantwort
eindeutig lesbar und nach internem exaktem Oracle **falsch** ist.

Die Pipeline erzeugt **keine Modellantworten**, verändert keine Gewichte,
führt keinen Trainer aus und liest **keine Holdout-Aufgaben**.
Ein Modelllauf mit Temperatur ~0,6 kann später ausschließlich auf
Training/Validation generieren; er braucht einen separaten explizit
autorisierten Inferenzjob. Der abgeschlossene NERAL-EXT-400-Benchmark
ist **keine** Trainingsquelle.

## Eingabeformat: lokal gespeicherte NERAL-Antworten

Jede Zeile in `v6-train-responses.jsonl` enthält genau:

`id, family, split, index, prompt_sha256, adapter_sha256,
decoding_sha256, generation_source, model_generated, response`.

- `family` ist eine der fünf Familien aus
  `scripts/neral_v6_compositional.py`.
- `split` ist ausschließlich `train` oder `validation`.
- `index` ist der Generator-Index in diesem Split; keine freie Aufgabe.
- `id` lautet exakt `v6-{family}-{split}-{index:03}`.
- `prompt_sha256` ist SHA-256 des UTF-8-Nutzerprompts, der mit
  `make_case(family,split,index)` erzeugt wurde.
- `adapter_sha256` ist der SHA des tatsächlich geladenen v5-LoRA.
  Für den bestehenden v5-480-Adapter gilt der bekannte SHA
  `2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5`.
- `decoding_sha256` muss aus dem vorher fixierten
  vollständigen Generierungsvertrag berechnet werden.
- `generation_source` = `neral_v5_local`;
  `model_generated` = `true` **nur bei tatsächlicher lokaler Ausgabe**.
  Die CPU kann diese Herkunftsbehauptung nicht kryptografisch beweisen.

Beispiel zur **Struktur**, keine tatsächliche Modellantwort und
**nicht** für Training verwenden:

```json
{"id":"v6-inventory_flow-train-000","family":"inventory_flow","split":"train","index":0,"prompt_sha256":"<64 hex echte Prompt-SHA>","adapter_sha256":"<64 hex geprüfte Adapter-SHA>","decoding_sha256":"<64 hex realer Konfig-SHA>","generation_source":"neral_v5_local","model_generated":true,"response":"FINAL_ANSWER: 42 Stück"}
```

## Bewertung – keine Formatfalle

Für fünf Aufgabentypen werden exakte Werte miteinander verglichen:

- Geld: Cent als ganze Zahl, `12,50 EUR` und `12.50 €` gleich
- Treibstoff: gekürzte/rationale kg, `1/2 kg` und `0,5 kg` gleich
- Zeitzone: Uhrzeit **und** Tagesoffset
- Lager: vollständige Ganzzahl mit Einheit Stück
- Mischung: exakter rationaler Prozentwert

Es wird nur eine eindeutige finale Antwort am Ende akzeptiert. Ein
fehlender oder widersprüchlicher Marker, ein mehrdeutiger Text oder
nur eine falsche Antwortform sind **kein** verifiziertes falsches
Rechenergebnis und landen daher **nicht** im Preference-Paar.

- Eindeutig **gleichwertig** = korrekt; kein Paar.
- Eindeutig **falsch** = `chosen` aus rechnerischem Generator,
  `rejected` echte NERAL-Antwort; **Trainingskandidat**.
- **Nicht eindeutig bewertbar** = aussortieren, kein falsches Label.

Die interne Fraction-Arithmetik ist ein Oracle innerhalb derselben
Generatorfamilie, **kein unabhängiger menschlicher oder zweiter
wissenschaftlicher Grader**. Darum steht im Manifest
`human_verified=false` und `training_allowed=false`.
Vor einem GPU-Smoke müssen konkrete Verifizierläufe, menschliche
Prüfbelege und ein echter Modell-Generierungsnachweis vorliegen.

## Lokale Nutzung

```powershell
python -m unittest discover -s tests -p test_neral_e1_pair_miner.py -v

python -m scripts.neral_e1_pair_miner `
  --responses C:\private\v6-train-responses.jsonl `
  --dev-suite C:\private\v6-dev-600.jsonl `
  --release-suite C:\private\v6-release-600.jsonl `
  --adapter-sha256 2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5 `
  --decoding-sha256 <echter-64-stelliger-konfig-SHA> `
  --out C:\private\e1-mined-pairs-new
```

Das Skript erzeugt **nur, wenn die Kriterien erfüllt sind**, ein neues
privates Verzeichnis mit `pairs.jsonl` und `manifest.json`.
Die Paare entsprechen dem bereits integrierten
`scripts/neral_e1_smoke.py`-Schema. Pro Split müssen mindestens fünf
Familien repräsentiert sein, und **keine Familie darf mehr als 20 %**
ausmachen. Der Miner wählt dazu je Split aus jeder Familie gleich viele
**tatsächlich beobachtete** falsche Antworten aus, bei Überhang
reproduzierbar nach einer SHA-basierten Reihenfolge. `verified_wrong`
erfasst alle bewertbaren Fehlantworten, `eligible_pairs` die
balanciert exportierten Kandidaten und `balanced_discarded` die
abgewählten Überschüsse. Fehlen einer Familie genügend echte Fehler,
bleibt die Ausgabe unter der Mindestzahl und der gesamte Smoke-Export
wird abgelehnt – es werden **keine** künstlichen Fehlantworten erfunden.

Die private Dev-/Release-Suite muss zuvor wirklich erstellt,
unabhängig kontrolliert und gegen ALLE Trainingsquellen abgeglichen
sein. Dieser Paarbauer prüft nochmals Eingabekennungen und exakte
bzw. 10-Gramm-Überschneidungen mit beiden Suiten. **Weder rohe
Release-Aufgaben noch ihre Lösungen gehören in öffentliche GitHub-
Artefakte.** Das Output-Verzeichnis muss neu und außerhalb des
Repositorys liegen.

## Grenzen und nächste Schritte

1. Fehlende 600er privaten, unabhängig geprüften Dev-/Release-Suites
   erstellen und vor dem ersten Freigabeversuch vorregistrieren.
2. Erst dann auf neuen v6-TRAIN- und Validation-Aufgaben echte lokale
   NERAL-v5-Antworten generieren und **privat** speichern. Der Miner
   darf nie auf der verbrauchten EXT-400-Suite laufen.
3. Mit diesem Skript **eindeutig rechnerisch falsche** Antworten
   markieren. Zusätzlich verifizierte echte Coding-Fehlpaare aus
   einer kontrollierten Sandbox sammeln – diese Pipeline alleine
   ist noch mathematisch dominiert.
4. Nach zusätzlichem unabhängigen Review und echten Receipts erst
   einen getrennten 30-Step-ORPO/DPO-Smoketest erwägen.
5. Eigenständige v6-Modellqualität bleibt bis zum Dev-/Release-Blindtest
   **ungeprüft**; ein kleineres Loss allein belegt nichts.
