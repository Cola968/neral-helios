# NERAL v6 – neuer dreifacher Entwicklungstest (72 Aufgaben)

**Zweck:** Nach dem echten v6-120-Schritte-Training #42407667120 endlich
*vergleichbare* Modellantworten gegen die SHA-gepinnte Qwen2.5-Coder-7B-Basis
mit drei eingefrorenen Few-Shot-Beispielen und gegen den alten v5-Adapter
messen. Keine v6-Gewichte verändern, keine alte EXT-400-Antwort als Training
verwenden.

Der Test enthält **72 bislang nicht im v6-Trainingsdatensatz verwendete
synthetische Aufgaben**: je 24 Mathematik, Aussagenlogik und Python-
**Codeverständnis** mit je vier Lösungsfamilien (6 Fälle pro Familie).
Coding-*Generierung* und Debugging werden hier **nicht** getestet.
Die Aufgaben beruhen auf eigenständigen Programmen und lassen sich mit
unabhängig nachvollziehbaren Integer-Ergebnissen bewerten. Es ist eine
**Entwicklungssuite**, weder extern unabhängig noch der versiegelte
600er-Release-Test. Nicht als solche bewerben.

Der Generator prüft Prompt-/Antwort-Eindeutigkeit und normalisierte
10-Gramm-Überschneidungen gegen alle 728 v6-SFT-Trainingsaufgaben und
100 v6-Validierungsaufgaben; damit ist Pretraining-Unbekanntheit
nicht nachgewiesen.

## Vergleich

- `qwen_fewshot`: unveränderte Qwen-Basis mit eingefrorenen drei Few-Shots.
- `neral_v5`: LoRA SHA
  `2a89433bb69243b8b65b47b6492f7378b73c5e0f5a698cb549c38d7605e100d5`
- `neral_v6`: tatsächlich trainierter 120-Step-LoRA SHA
  `8bdc878ae193f8d13dd5f71e1a9cef1aeb6800f92848c9095d4e8fa5860203fd`

Für alle gilt exakt dasselbe **System- und Aufgabenprompt** (Few-Shot-
Kontrollbedingung separat protokolliert), greedy `do_sample=False`,
`max_new_tokens=160`, lokale 4-Bit-NF4-Foundation und keine Werkzeuge.
Modelle werden **nacheinander in getrennten Prozessen** auf deiner RTX 4070
geladen, damit v5/v6 nicht gleichzeitig in 12 GB VRAM liegen.

Der konservative Grader akzeptiert ausschließlich eindeutig erkennbare
Ganzzahlen, `FINAL_ANSWER`, `FINAL`, `Answer`, `Antwort`,
`Ergebnis` am Ende, `\\boxed` oder eine explizite abschließende
Antwortformulierung. Ein willkürlicher letzter Zahlenwert wird
**nicht** als korrekte Lösung gewertet. Abgeschnittene Antworten und
nicht extrahierbare Texte werden separat ausgewiesen, nicht still
als verifizierte Rechenfehler bezeichnet.

Die Auswertung zeigt richtige Antworten je Bereich und Modell,
v5-v6-vs-Qwen Netto-Differenzen und den exploratorischen exakten
McNemar-Test. Der Test hat mit n=72 zu geringe Aussagekraft für
große allgemeine Claims und ist **nicht** vorregistriert als v6-Gate.

## Ausführen

Workflow `.github/workflows/neral-v6-dev72-gpu.yml` auf `main`
manuell starten mit dem exakten Autorisierungstext
`EVALUATE_NERAL_V6_NEW_DEV72_READONLY`.

Der Workflow erzeugt vorher einen neuen privaten lokalen Ordner
`NERAL-QLoRA-Research\\v6-new-dev72-<RUN_ID>-1`, prüft die
SHA-256-Hashes beider Adapter und der Basis, das freie VRAM,
führt die drei Inferenzprozesse nacheinander aus und lädt
**nur die skalare Zusammenfassung** auf GitHub hoch.
Die originalen Modellantworten und Aufgaben bleiben auf
dem eigenen PC und dürfen nicht öffentlich als Artefakt
hochgeladen werden.

Keine Trainingsjobs, keine Cloud-APIs, keine Abos,
kein Adapter-Merge und keine automatische v6-Promotion.
Wenn dieser Entwicklungstest positiv ist, muss danach ein frischer,
unberührter, unabhängig kontrollierter Test folgen.
