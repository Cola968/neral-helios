# NERAL dev72 – Audit vor der Oracle-Sperre (07.11.2026)

Stand beim Schreiben: `main` = `ae71fbd91e90b5f49a78b525466a065df50f65fc`.
Dieser Text ist die Bestandsaufnahme **vor** der Codeänderung. Er ändert keine
Scores, keine Adapter und keine Trainingsdaten.

CPU-Prüfung, kein GPU-Lauf, kein Training, kein Modelldownload, keine
kostenpflichtige API. Rohantworten des Laufs 42409458549 liegen nicht im
Repository; öffentlich ist nur das aggregierte JSON im Actions-Log.

## Was abgedeckt wurde

- Trainingsherkunft v6 (`scripts/neral_v6_r16_gpu.py`, `scripts/neral_v6_compositional.py`, `scripts/neral_reasoning_train_candidate.py`) und v6.1-Replay (`scripts/neral_v61_replay_train.py`): 120 Schritte, v6 Lernrate `1e-5` ohne 70/30-Replay, v6.1 Lernrate `2e-6` mit 560 v5- und 240 v6-Trainingszeilen. Completion-Maske verlangt Präfixgleichheit und bricht bei Trunkierung ab. Kein stilles Abschneiden.
- Beide 72er-Generatoren und der gemeinsame Zahlenparser `parse_answer` in `scripts/neral_v6_fresh_dev_eval.py`.
- Alle 48 Code-Tracing-Aufgaben beider Suiten: eingebettetes Programm in einem isolierten CPython-Prozess (`python -I`, Zeitlimit) ausgeführt. Gold und Druckausgabe stimmten in jedem Fall, einschließlich Listenaliasing auf aktuellem `main` (Index 0 druckt 17, nicht 14).
- Nicht-Code-Familien stichprobenartig gegen den Prompttext (GCD, Potenz, Münzen, Ziffernsumme, XOR, Erreichbarkeit, Belegungen, kürzester Weg, Binomial, Rest, Teilbarkeit, Gleichung, Bool, Ordnung, Horn, Mengen). Die dabei zuerst gesehenen „Abweichungen“ waren Fehler des Prüfparsers (Koeffizienten statt Zielzahl), nicht der Goldwerte.
- Scorer: absichtlich falsche ganze Zahlen werden nicht als richtig gezählt; Prosa, die die Goldzahl nur erwähnt, bleibt unbewertet (`ungraded`), nicht richtig. Der Parser lehnt letzte-Zahl-Raten und doppelte FINAL-Marker ab.
- AST-Tracer `neral_agent_python_trace.py`: die vier v6-Codefamilien (Slice, verschachtelte `continue`, Dict, String-Stride) stimmen mit CPython überein. Alle 24 v6.1-Codeaufgaben (Rekursion, Set-Comprehension, Alias, `enumerate`) liefert der Tracer als nicht unterstützt, inklusive `append` und Listenzuweisung. Das ist der bestehende Sicherheitsvertrag in `tests/test_neral_agent_python_trace.py`, kein stilles Falschergebnis.
- Workflows unter `.github/workflows`: GPU-Jobs auf dem Self-Hosted-Label `neral-gpu` starten bei Push nur auf benannten Zweigen (`gpu-run-*`, `main` plus Freigabetext) oder per `workflow_dispatch`. Der einzige Pull-Request, der ohne zusätzliche Freigabe auf `neral-gpu` läuft, ist `neral-agent-live-pc.yml` und nur bei dessen eigenen Pfaden. Die dev72-GPU-Workflows sind manuell und auf `main` plus Autorisierung beschränkt.
- GitHub: Lauf [42409458549](https://github.com/Cola968/neral-helios/actions/runs/42409458549) ist `success`. Das öffentliche Aggregat im Log lautet Qwen+3 Few-Shots 29/72, v5 24/72, v6 21/72; unbewertet komplett 0/7/19; unvollständig 0; v6 gegen v5: 0 Verbesserungen, 3 Verschlechterungen, exakter McNemar p = 0.25; v6 gegen Qwen: 2/10, p = 0.03857421875. Domänen: Qwen 9/15/5, v5 7/14/3, v6 4/14/3 (Mathematik / Aussagen / Code-Trace).
- Lauf [42413078181](https://github.com/Cola968/neral-helios/actions/runs/42413078181) ist `cancelled` auf Commit `d7c33cc`. Qwen hatte 72 Generationen bereits beendet; Abbruch während v5 (`The operation was canceled`, 20:53:43Z). Kein veröffentlichtes Drei-Wege-Aggregat für v6.1. Der Log nennt die Goldwerte nicht als Abbruchgrund; der Zeitstempel liegt unmittelbar vor dem Gold-Fix `95e59c88` (20:54Z).
- Lauf [42411765250](https://github.com/Cola968/neral-helios/actions/runs/42411765250) ist das v6.1-Training, grün, **kein** Vergleichsergebnis.
- PR [#363](https://github.com/Cola968/neral-helios/pull/363) ist gemergt: Receipt-SHA und UTF-8, kein neues Modellergebnis. PR [#360](https://github.com/Cola968/neral-helios/pull/360) ist gemergt: 250 AST-Fälle, Werkzeugnachweis, keine Gewichtsverbesserung.

## Unbekannt oder nicht unabhängig

- Die 72 Rohantworten je Modell. Ohne sie ist die Aufteilung der 19 unbewerteten v6-Ausgaben in Format, falsche Zahl und unklaren Text nicht erneut gezählt. `scripts/neral_v6_dev72_regression_audit.py` kann das nur auf dem privaten Windows-Ordner.
- Ob die 3 strengen v6-gegen-v5-Rückschritte falsche Zahlen oder nur nicht lesbare richtige Rechnungen sind.
- Ein matched Zero-Shot-Qwen und ein Few-Shot-v5/v6. Heute sieht nur Qwen die drei Beispiele (`prepare_fewshot_messages`, Zeilen 229–237). Zwei der Beispiele sind keine ganzen Zahlen (`1/2`, `1,50 EUR`).
- Pretraining-Kontamination. Exakt- und 10-Gramm-Prüfung gegen Train/Val/altes dev72 ist keine Unabhängigkeit.
- Familien sind je 6 Oberflächen einer Schablone, deutsch/englisch gepaart. n = 72 ist keine 72 unabhängigen Aufgaben. McNemar auf Aufgabenebene ignoriert diese Abhängigkeit.
- v6.1-Verlustkurve und Adapter-SHA `33e939f6…` aus dem abgebrochenen Lauf sind keine Fähigkeit.
- Ob eine andere Lernrate, Replay-Quote oder Antwortmaske den Rückschritt verursacht hätte. Keine gepaarte Ablation im Repository.
- Externe, unberührte Endprüfung existiert als Intake-Vertrag, nicht als ausgewertetes Ergebnis dieser Runde.

## Alias-Gold, getrennt von 21/72

Auf `d7c33cc` stand in `scripts/neral_v61_new_dev_eval.py` für `python_mutable_aliasing`:

`answer = (p + 3*i) + (q + i) + 3`

Nach `a.append(3)` enthält `sum(a)` die 3, und `len(b)` addiert eine weitere 3, weil `b` dieselbe Liste ist. Index 0 druckt 17. Die alte Formel liefert 14. Commit `95e59c88` bzw. `ae71fbd9` setzt `+ 6`. CPython bestätigt 17, 23, 29, 35, 41, 47.

Diese Aufgaben gehören **nicht** zur Suite von Lauf 42409458549 (Fingerprint `4b9a302a…`, Familien Slice/Continue/Dict/String). Der Fehler erklärt die historischen 21/72 nicht. Die alte Suite-Codefamilien stimmen heute mit CPython überein.

Die verbleibende Lücke auf `ae71fbd9`: `tests/test_neral_v61_new_dev_eval.py` prüft Alias mit `p+q+4*i+6`, algebraisch dieselbe Formel wie der Generator. `tests/test_neral_v6_fresh_dev_eval.py` `test_oracles_hand_checked_per_family` prüft keine Zahlen. Ein gemeinsamer Formelirrtum kann beide Stellen gleichzeitig bestehen. Der AST-Tracer kann die v6.1-Programme nicht gegenchecken.

## Drei wahrscheinlichste Ursachen des v6-Rückschritts

Reihenfolge nach Erklärungskraft für 29 / 24 / 21, nicht nach Behebbarkeit auf CPU.

### 1. Messung und Prompt-Protokoll, nicht nachgewiesenes Lösen

Beleg: Actions-Log 42409458549, eine Zeile, Scores 29/24/21 bei 0/7/19 unbewerteten kompletten Antworten. v6 gegen v5 nur 3 strenge Rückschritte und 0 Verbesserungen (`scripts/neral_v6_fresh_dev_eval.py` Paarbildung; p = 0.25). Die Schlagzeile 29 gegen 21 vermischt zusätzlich Few-Shot nur bei Qwen (`FEWSHOT` Zeilen 29–31, `prepare_fewshot_messages` 229–237) mit einem Parser, der mehrzeilige Rechnungen ohne genau ein terminales FINAL verwirft (`parse_answer` 200–226).

Konfidenz hoch, dass die 8 Punkte gegen Qwen **nicht** als Fähigkeitsdifferenz gelesen werden dürfen. Konfidenz mittel, welcher Anteil der 19 unbewerteten v6-Texte inhaltlich richtig war.

Alternative: Qwen wäre auch ohne Beispiele besser, und die 19 Texte sind falsch, nicht nur unlesbar.

Falsifikation: dieselben 72 Aufgaben, gekreuzt (Qwen ohne Beispiele, v5/v6 mit denselben drei Beispielen), gleicher Decoder, unveränderte Gewichte. Nur die Partition zählt, die alten 21/72 werden nicht umgeschrieben. Das ist ein GPU-Lauf und wird hier nicht gestartet. Vorher müssen die Goldwerte execution-fest sein.

### 2. Strenger Rückschritt v5 zu v6 auf derselben Prompt-Bedingung, Ursache offen

Beleg: gleiche Nachrichten für v5 und v6 (Test `test_reference_prompts_include_locked_three_shots`). Mathematik 7 zu 4, Code 3 zu 3, Aussagen 14 zu 14. Drei Fälle, in denen v5 streng richtig und v6 nicht ist. Adapter v6 `8bdc878a…` aus Training 42407667120, 120 Schritte, Lernrate `1e-5` (`scripts/neral_v6_r16_gpu.py` um Zeile 104 und 184), Daten nur Komposition plus Gegenbeispiele, kein v5-Replay.

Konfidenz hoch für den Zahlenabfall im gesperrten Parser. Konfidenz **niedrig**, dass es Vergessen der Gewichte ist: die drei Fälle können unlesbare richtige Antworten sein. Der CPU-Audit, der Format und falsche Zahl trennt, ist nicht mit öffentlichen Zählern gelaufen.

Alternative: Rauschen. p = 0.25 auf 3 gegen 0, und die 12 Familien sind korreliert, also ist der p-Wert eher zu optimistisch als zu streng.

Falsifikation: privaten Ordner `v6-new-dev72-42409458549-1` mit `neral-v6-dev72-regression-cpu-manual.yml` auswerten und nur die drei Zähler `true_observed_wrong_regressions`, `format_recoverable_regressions`, `unresolved_ungraded_regressions` veröffentlichen. Keine Rohtexte.

### 3. Schmales QLoRA-Rezept als plausible, nicht gemessene Ursache

Beleg im Code, nicht in einem Ablationsergebnis: v6 trainiert 120 Schritte bei `1e-5` auf fünf Kompositionsfamilien plus 128 Gegenbeispiele, Fortsetzung von v5-LoRA. v6.1 senkt auf `2e-6` und mischt 70/30, hat aber keinen abgeschlossenen Drei-Wege-Vergleich. Literatur (Anwendbarkeit, keine übernommenen Prozentwerte):

- Approximate Regularized Replay, arXiv:2512.22337: Replay/KL kann Vergessen nach schmalem LoRA-SFT auf Qwen2.5-**Instruct**-7B mindern. Nicht Coder, nicht 4-bit, und ein festes Token-Budget verdrängt neue Beispiele. Beweist nicht, dass 70/30 hier optimal ist.
- arXiv:2602.04998: Lernrate zuerst prüfen; die dortigen Peaks sind Llama-2-7B, nicht dieses QLoRA.
- LoRA-Tradeoffs, arXiv:2405.09673: längeres Training kann Erhalt verschlechtern; Rank, Alpha, Lernrate und Zielmodule wirken zusammen. Kein Freibrief für eine bestimmte Konfiguration.
- QLoRA arXiv:2305.14314 und Instruction-Modelling arXiv:2405.14394: Antwort-only-Masken sind nicht universell. Hier ist die Maske bereits Präfix-geprüft; ein CPU-Fixture zeigt keinen stillen Trunkierungsfehler. Ob sie für diese Daten die richtige Wahl ist, ist offen.
- rsLoRA arXiv:2312.03732 ändert die Schrittweite über `alpha/sqrt(r)`. Keine Garantie für weniger Vergessen.
- EvalPlus arXiv:2305.01210: funktionale Ausführung deckt falsche Referenzen auf und kann Ranglisten ändern. Genau das ist die Lücke der handgeschriebenen Goldformeln.

Konfidenz mittel für „das Rezept kann Vergessen erzeugen“, niedrig für „es hat die 3 Punkte verursacht“.

Alternative: die alten dev72-Aufgaben liegen nahe an den Trainingsoberflächen (Prozent, Treibstoff, Zeit), der Rückschritt ist Spezialisierung, kein allgemeines Vergessen. Dafür spricht der Mathematik-Anteil 7 zu 4 bei flachem Code und Aussagen. Dagegen spricht, dass die neue Suite andere Familien nutzt und noch nicht ausgewertet ist.

Falsifikation: neuer Kandidat nur auf familienweise zurückgehaltenen Aufgaben, getrennt nach Transfer und Erhalt, gleiche Vorlage und gleicher Decoder für alle drei Modelle, gepaarte Unsicherheit auf Familienebene. Die verbrauchten 72 und jede Endprüfung, die zur Auswahl benutzt wurde, bleiben draußen.

## Gewählte Änderung (in diesem Commit noch nicht im Code)

Eine Ausführungssperre für Repository-Goldwerte: jedes Code-Programm der beiden Dev-Suiten muss in einem isolierten CPython-Prozess die Goldzahl drucken, und jede Nicht-Code-Aufgabe muss aus dem **Prompttext** mit einem zweiten Verfahren dieselbe ganze Zahl ergeben. Die zurückgezogene Alias-Formel `+ 3` muss dabei scheitern, die aktuelle `+ 6` bestehen. Absichtlich falsche Modellantworten bleiben falsch, Prosa mit der richtigen Zahl bleibt unbewertet.

Das ist Messzuverlässigkeit. Es ist kein Intelligenzgewinn, kein neuer 7B-Wert und keine Umschreibung von 21/72. Der AST-Tracer für Modellantworten bleibt unverändert und führt weiter keinen Modellcode aus.
