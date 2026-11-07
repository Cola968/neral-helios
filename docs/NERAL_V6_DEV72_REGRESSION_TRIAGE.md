# NERAL v6 – Ursachenanalyse für den negativen 72er-Entwicklungstest

**Originaler Benchmark #42409458549 (07.11.2026):** Qwen+Few-Shot
29/72, NERAL v5 24/72, NERAL v6 21/72. Alle Ausgaben vollständig,
davon 0/7/19 mit dem ursprünglichen Zahlenparser nicht auswertbar.
v6 hat gegenüber v5 **0 Verbesserungen, 3 Verschlechterungen** in
der streng gemessenen Korrektheit. Das Training #42407667120 veränderte
die Gewichte, aber der Dev-Test zeigte **keinen Fähigkeitsgewinn**.

## Sofort: keine weitere LoRA-Runde mit den bisherigen Trainingsdaten

Auf dieser verbrauchten 72er-Suite keine Daten für Training, DPO
oder vermeintliche Release-Bewertungen ableiten. Der v5-Adapter
bleibt das sichere Forschungs-Rollback.

## CPU-only Root-Cause Audit

`scripts/neral_v6_dev72_regression_audit.py` liest die **gespeicherten**
72 Rohantworten je Qwen/v5/v6 aus
`NERAL-QLoRA-Research\\v6-new-dev72-42409458549-1`.

Die Original-Prompt-/Adapter-SHAs und sämtliche Original-Scores bleiben
unverändert. Je Modell wird aus den vollständigen, nicht parsebaren
Antworten abgeleitet:

- Mehrere widersprüchliche Final-Marker
- Final-Marker vor nachfolgenden Textzeilen
- Nicht eindeutig erkennbares Endergebnis
- Nicht gültiger terminaler Final-Marker

Zusätzlich zeigt der **explorative** Parser, ob eine zusätzliche
Konvention wie `FINAL_ANSWER: 42.` oder ein eindeutiges abschließendes
`\\boxed{42}` einer rein syntaktisch abgewiesenen Ausgabe eine
exakte Zahl zuordnen kann. Er akzeptiert **keine beliebige letzte Zahl**
aus einer Rechnung oder Diskussion. Diese Funde werden
**nicht** in die ursprünglichen 21/72 umgeschrieben.

Aufgaben, in denen v5 korrekt und v6 nicht korrekt war, werden
getrennt als **explorativ behebbarer Formatverlust**,
**eindeutig falsche Zahl** oder **nicht entscheidbarer Text**
ausgewiesen. Nur die zweite Kategorie ist ein eindeutiger
mathematischer Falschausgabebefund; die dritte braucht einen privaten
menschlichen Review.

Die private, unveränderte Reviewstichprobe enthält 20 Fälle je Modell,
einschließlich ursprünglicher Aufgaben, Referenzen und vollständiger
Antworten. **Sie wird ausschließlich auf deinem PC abgelegt.**
GitHub sieht nur Summenzähler.

### Manuell und ohne GPU-Inferenz

Starte `.github/workflows/neral-v6-dev72-regression-cpu-manual.yml` in
`main` mit `authorize=AUDIT_NERAL_V6_DEV72_REGRESSION_READONLY`.

Es entstehen nur zwei neue Dateien im privaten Forschungsordner:
`scalar.json` und `20-private-samples-per-model.jsonl`.
Nur `scalar.json` wird veröffentlicht.

Der Workflow liest keine Modellgewichte, verändert keine Adapter und
kostet keine GPU-Rechenzeit.

### Vor neuem Training nötig

1. Die 19 nicht extrahierbaren v6-Fälle mit den Originalantworten prüfen.
2. Falls ein reines Formatproblem vorliegt: in Helios einen unabhängig
   kalibrierten FINAL-Extraktor mit Verifikationsroutine verwenden, nicht
   einen trainierten Adapter als Formatpflaster.
3. Falls die Modelle eindeutige Mathematikfehler machen: neue,
   **unabhängig geprüfte** Trainingsfamilien und Gegenbeispiele erheben,
   nicht nur weitere Beispiele der ursprünglichen fünf Familien.
4. Neue Modellversion auf völlig frischen Testaufgaben unter identischer
   Inferenzkonfiguration prüfen. Die alte 72er-Suite ist jetzt ein
   **Entwicklungsdiagnostikum** und kein Release-Test.
