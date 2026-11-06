# NERAL v3: echter Qwen-Tokenizer-Gate vor längerem QLoRA-Training

Für einen späteren fairen Rank-16-/Rank-32-Versuch prüft der Workflow
`.github/workflows/neral-v3-token-audit.yml` die lokal bereits
eingefrorenen 320 Trainings- und 64 Validierungsbeispiele.

Er verwendet den **unveränderten** QLoRA-Trainer
`a082738f9145722dec443141e5d7801a3ead93af` und den lokal
SHA-256-geprüften Qwen2.5-Coder-7B-Instruct-Tokenizer. Nicht nur das
JSONL-Format wird geprüft, sondern die tatsächlichen Chat-Tokenfolgen
und die Trainingsmasken:

- kein Beispiel über 512 Tokens; Überlängen werden strikt abgewiesen,
  niemals abgeschnitten;
- alle System- und User-Tokens haben Verlustmaske 0;
- ausschließlich die zusammenhängende Assistant-Fortsetzung trägt
  Verlustmaske 1;
- keine fehlenden oder leeren Assistant-Fortsetzungen;
- jede lokale Train-/Validation-Datei und das Quellenreview müssen die
  festgehefteten SHA-256-Werte aus Run #42283867443 besitzen;
- Berichte enthalten nur Länge, Anzahl, p95, Summen und SHA-256.
  Keine Prompts, Modellantworten oder Gewichte verlassen den PC.

Das ist ein **technisches Preflight**, kein Modelltraining, kein
Qualitätstest auf Unbekanntem und kein Release-Gate. Die 64
Validierungsbeispiele haben strukturähnliche Trainingsvorlagen.
Der Workflow autorisiert selbst keine Gewichtsänderung oder Adapteraktivierung.
