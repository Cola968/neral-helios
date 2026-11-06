# NERAL v3 – zweizeilige Algebra ohne Antwortformat-Widersprüche

**Status:** Neuer Trainingskandidat, noch nicht trainiert und nicht aktiviert.

Im bisherigen 384er-Datensatz (SHA256
`4b0e54712ea889cadd98eb668d315f3eb8647a7b9ac2209a1ed9942f7bbd3941`)
sind die Gleichungen nicht im einheitlichen zweizeiligen Sollformat:
70/70 alte Gleichungen sowie 75/100 neue Gleichungen haben reine Zahlen;
die übrigen 25/100 enthalten einen *einzeiligen* semikolongetrennten
Zwischenschritt. Diese 170 Gleichungen werden **nicht in-place** geändert.
Die getrennten Train-/Validation-Splits aus Run #42283867443 bleiben erhalten.

Der Konverter `scripts/neral_v3_two_line_equations.py` lässt das
Source-Review *jeder* Zeile bestehen, verifiziert die ursprünglichen
Prompt-/Antwort-Bytes gegen die deterministischen Quellgeneratoren und
berechnet jede neue Gleichung mit einem abgesicherten lokalen AST-Solver,
nachdem er zusätzlich die algebraische Umformung als Bruchrechnung geprüft hat.
Er verändert **142 Trainings- und 28 Validierungsgleichungen** sowie die
jeweiligen **Prompts** so, dass die ausdrückliche Ausgabeanforderung
zum neuen zweizeiligen Target passt:

```text
6x = 318
x = 53
```

Die übrigen **178 Training- und 36 Validierungsbeispiele** bleiben inhaltlich
unverändert. Jeder neue Quelldatensatz und jedes Review-Manifests erhält
eine frische SHA256-Prüfsumme, die ursprüngliche Quellzeile bleibt dokumentiert.
Eine mathematisch unpassende Umformung, ein veränderter Originalprompt,
überlappende Train/Val-Quellen oder ungültige Quellen-SHA stoppen den Export.

Der manuelle Windows-Workflow `neral-v3-two-line.yml` wiederholt die
Quellprüfung mit dem **originalen unveränderten** QLoRA-Trainer und führt
danach den echten Offline-Qwen-Tokenizer durch: vollständige Sequenz
maximal 512 Tokens; nur Assistant-Completion darf Gradienten erhalten;
alle Masken müssen den originalen Completion-only-Vertrag erfüllen.
Keine Gewichte/Prompts/Trainingsantworten verlassen den PC; GitHub
bekommt ausschließlich skalare Berichte.

**Wichtig für zukünftige Evaluierung:** Durch das Training mit allen
170 zweizeiligen Gleichungen besteht das Risiko, dass Zahlen-only-Aufgaben
in den alten 80 Development-Tests schlechter werden. Deshalb wird vor
einer Rank-16- oder Rank-32-Aktivierung sowohl das strenge Zahlenformat
als auch die zweizeilige Gleichungskorrektheit getestet, zusätzlich zu
Core-/Code-Regressionen. Die 64 Validation-Tasks sind nicht unabhängig
vom Trainingsgenerator und die alten 36 Aufgaben sind bereits bekannt.
