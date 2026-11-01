# NERAL – 100 überprüfbare Entwicklungsideen und neuer Releaseplan

Stand: 2026-11-01. **100 Vorschläge, NICHT 100 implementierte Funktionen.** Das Wort »Überraschung« ist eine Produktidee mit prüfbarem Ergebnis, keine Erfolgsmeldung. Kein GPT-3-/3.5-Paritätsbeleg durch diese Liste.

## Forschung, Video- und Folienbasis – was wirklich ausgewertet wurde

- **Stanford CS336 Lecture 1 (Video-Transkript, 2025):** Debugging mit kleinen Modellen, Skaleneffekte und Ressourcenrechnung; Schlussfolgerung: erst mit kleinen Tests Fehler beweisen. https://www.youtube.com/watch?v=SQ3fZ1sAqXI
- **Stanford CS336 Lecture 12 (Video-Transkript, 2025):** Benchmarks, Verteilungen und TRAIN/TEST-Kontamination; Schlussfolgerung: blindes Holdout-Set und mehrere unabhängige Gütemaße. https://www.youtube.com/watch?v=x-R5l2HsXqM
- **Stanford CS336 Lecture 14 (Video-Transkript, 2025):** Qualitätsfilterung, Near-Duplikate und Datenpipeline; Schlussfolgerung: hochwertige Daten vor mehr Tokens. https://www.youtube.com/watch?v=9Cd0THLS1t0
- **Stanford CS336 2026 (Vorlesungswebseite und Folienverzeichnis):** 18+ Veranstaltungen einschließlich SFT/RLHF, RLVR, Evaluierung, Inferenz; die PDF-Folien selbst wurden NICHT vollständig angeschaut. https://cs336.stanford.edu/
- **DataComp-LM (Original-Paper, PDF S. 1–3 einschließlich Ergebnisgrafik):** Datenkuratierung kann Effizienz stark beeinflussen; die publizierten Resultate für 7B-Modelle gelten nicht als empirische Prognose für Arctus350. https://arxiv.org/pdf/2406.11794
- **Hugging Face Modell-Metadaten:** Qwen3.5-9B ist eine externe 9.65B-Parameter-Basis unter Apache-2.0, nicht das eigene Arctus350. https://huggingface.co/Qwen/Qwen3.5-9B
- **Hugging Face Smol Course:** SFT, Daten, Präferenzabgleich und Evaluierung als Techniken; diese Module lassen sich nicht ohne Architekturprüfung direkt auf Arctus350 übertragen. https://huggingface.co/smol-course
- **FineWeb-Edu:** Bildungsqualität und nachvollziehbare Filter als Datensatz-Ansatz, ohne automatische Lizenz-/Datenfreigabe. https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- **Anthropic Building Effective Agents:** Begrenzte Workflows mit Evaluator-Optimierer statt blind unbeschränkter Agentenschleifen. https://www.anthropic.com/research/building-effective-agents
- **OpenAI Evals Best Practices:** Task-spezifische Evaluierung mit menschlicher Kontrolle; keine Gleichsetzung von Fixture-Tests und realer Modellfähigkeit. https://developers.openai.com/api/docs/guides/evaluation-best-practices
- **SWE-bench Verified:** Coding-Agenten brauchen echte Bugfix-Evidenz und eigene Testläufe. Nicht als Ergebnis für NERAL verwenden. https://www.swebench.com/

## Reale Ausgangslage / Releaseblocker

- 40+ offene, häufig gestapelte GitHub-Draft-PRs statt eines stabil integrierten Release-Branches (gezählter GitHub-Abruf 01.11.2026).
- #203/#204: externe Qwen-Anbindung und Thinking; #210: 34 DEV-Fälle, davon 26 automatisch und 8 manuell, KEINE validierte Modellquote; #212: abgeschnittene Ollama-Antworten; #176 Orbit; #207–#209 native Arctus-Forschung.
- Arctus350: ca. 349.8M Parameter; auf kleiner gespeicherter 8-Fragen-Probe zuvor 0/8 exakt; ein GPT-3.5-Niveau ist derzeit nicht belegt.
- Noch kein umfassend verifizierter Windows RTX4070-Browser→Qwen-End-to-End-Test für den zusammengestellten Release-Branch. Produktqualität und eigene Modellintelligenz bleiben strikt getrennt.
- Keine automatischen Merges, Modell-/Gewichtsänderungen, kostenpflichtige Jobs, API-Schlüssel, Cloud-GPU oder Downloads ohne separate Freigabe.

## 100 priorisierbare Produkt- und Forschungsüberraschungen

**Priorisierung:** P0 = sicherheits-/releasekritisch und sofort zu evaluieren; P1 = nach erstem stabilen Beta-Gate; P2 = Forschung/optionaler Ausbau. Umsetzung nur nach passendem Nutzen und Testergebnis, niemals alles gleichzeitig.

### Modellqualität & Messbarkeit

1. **[P0] Dreifach-Duell:** Dieselben Blindaufgaben separat mit Qwen ohne Tools, NERAL+Tools und Arctus350 auswerten. – Vorschlag/nicht als fertig ausgewiesen
2. **[P0] Eingefrorene Geheimtests:** Unveröffentlichte Holdout-Fälle mit Hashes, nie in Teacher- oder Trainingsdaten einspeisen. – Vorschlag/nicht als fertig ausgewiesen
3. **[P0] Deutsch-Englisch-Symmetrie:** Semantisch äquivalente Aufgaben in beiden Sprachen mit messbaren Differenzen bewerten. – Vorschlag/nicht als fertig ausgewiesen
4. **[P0] Fehler-Fingerabdruck:** Jede Fehlantwort als Rechnen, Fakten, Instruktion, Kontext, Format oder Unsicherheit klassifizieren. – Vorschlag/nicht als fertig ausgewiesen
5. **[P0] Kalibrierte Unsicherheit:** Wissen, Schätzung und Nichtwissen anhand echter Testfragen getrennt messen. – Vorschlag/nicht als fertig ausgewiesen
6. **[P0] Widerspruchs-Detektor:** Behauptungen innerhalb derselben Antwort auf logische Gegensätze prüfen. – Vorschlag/nicht als fertig ausgewiesen
7. **[P0] Gegnerische Minimalpaare:** Ein-Wort-Abwandlungen gegen unzulässige Mustervervollständigung testen. – Vorschlag/nicht als fertig ausgewiesen
8. **[P0] Kontextstress-Suite:** Fakten nach 2, 6 und 12 Gesprächszügen wieder abrufen lassen. – Vorschlag/nicht als fertig ausgewiesen
9. **[P0] Qualitätsverlauf pro Commit:** Genau dieselben Tests an einem konkreten Git-Commit und Modell-Digest verankern. – Vorschlag/nicht als fertig ausgewiesen
10. **[P0] Freigabe durch echte Fälle:** Release nur bei vollständigen Regressionen plus manueller Begutachtung zugelassen. – Vorschlag/nicht als fertig ausgewiesen

### Qwen-Reasoning & Werkzeuge

11. **[P0] Zwei-Werkzeuge-Komposition:** Codeanalyse und Arithmetik in einer Antwort mit maximal zwei verschiedenen Aufrufen – in PR #213. – Implementierung in Draft-PR #213, noch kein echter Qwen-E2E
12. **[P0] Intelligentes Denktiefenprofil:** Standard oder deep anhand Aufgabe vorschlagen, aber niemals heimlich umschalten. – Vorschlag/nicht als fertig ausgewiesen
13. **[P0] Präziser Tool-Entscheider:** Bei arithmetischer Exaktheit Rechner vor freier Vermutung bevorzugen. – Vorschlag/nicht als fertig ausgewiesen
14. **[P0] Gedankenlimit mit Hinweis:** Ausgabe- und Zeitbudget vor der Anfrage anzeigen, abgeschnittene Antworten sichtbar lassen. – Vorschlag/nicht als fertig ausgewiesen
15. **[P0] Kontext-Vorschau:** Vor dem Absenden zeigen, was in das begrenzte Qwen-Kontextfenster passt. – Vorschlag/nicht als fertig ausgewiesen
16. **[P0] Tool-Ergebnis-Prüfung:** Rechnerresultat und Modellbehauptung vor der Anzeige auf Zahlwidersprüche vergleichen. – Vorschlag/nicht als fertig ausgewiesen
17. **[P0] Unsicherheits-Rückfrage:** Bei fehlenden Aufgabenparametern gezielt eine Rückfrage statt erdachter Werte. – Vorschlag/nicht als fertig ausgewiesen
18. **[P0] Aufgabenzerlegung sichtbar:** Komplexe Fragen in nachvollziehbare überprüfbare Zwischenergebnisse zerlegen. – Vorschlag/nicht als fertig ausgewiesen
19. **[P0] Antwort-Korrekturpfad:** Nutzer kann eine Fehlantwort markieren und gezielte Regression erzeugen. – Vorschlag/nicht als fertig ausgewiesen
20. **[P0] Schnell-vs-tief-Vergleich:** Zwei konfigurierbare Inferenzmodi auf denselben Fragen hinsichtlich Kosten und Fehler vergleichen. – Vorschlag/nicht als fertig ausgewiesen

### Webseiten, Videos, PDFs & Wissen

21. **[P1] Quellen-Posteingang:** URLs und lokale Dokumente als getrennte, sichtbar nicht vertrauenswürdige Quellen ablegen. – Vorschlag/nicht als fertig ausgewiesen
22. **[P1] Webseiten-Leser:** Nur explizit freigegebene öffentliche HTTPS-Quellen abrufen, SSRF- und Redirect-Schutz. – Vorschlag/nicht als fertig ausgewiesen
23. **[P1] Video-Transkript-Leser:** Zeitmarkierte Transkripte erfassen und zitierfähige Abschnitte statt bloßer Titel nutzen. – Vorschlag/nicht als fertig ausgewiesen
24. **[P1] PDF-Textspur:** Textseiten samt Seite und Quellhash einlesen, Bilder gesondert kennzeichnen. – Vorschlag/nicht als fertig ausgewiesen
25. **[P1] Präsentations-Analyse:** Folienreihenfolge, Sprechertext und Grafikbeschriftungen als getrennte Evidenz. – Vorschlag/nicht als fertig ausgewiesen
26. **[P1] Zitate mit Fundstelle:** Jede Quellenbehauptung an URL, Dokumentversion, Zeitmarke oder Seitenzahl binden. – Vorschlag/nicht als fertig ausgewiesen
27. **[P1] Widersprüchliche Quellen:** Zwei Seiten systematisch auf unterschiedliche Fakten und Veröffentlichungsdaten prüfen. – Vorschlag/nicht als fertig ausgewiesen
28. **[P1] Aktualitätsanzeige:** Publikationsdatum, Abrufdatum und Gültigkeitszeitraum auseinanderhalten. – Vorschlag/nicht als fertig ausgewiesen
29. **[P1] Quellen-Offlinemodus:** Aus lokal freigegebenen Notizen antworten, ohne neue Webanfrage auszulösen. – Vorschlag/nicht als fertig ausgewiesen
30. **[P1] Prompt-Injection-Quarantäne:** Befehle aus Webseiten oder Dokumenten als Daten behandeln, nie als Anweisungen. – Vorschlag/nicht als fertig ausgewiesen

### Programmierung & überprüfbare Tools

31. **[P1] AST-Codeprüfer ausbauen:** Statische Python-Findings mit Begründungen und konkreten Stellen darstellen. – Vorschlag/nicht als fertig ausgewiesen
32. **[P1] Isolierte Test-Sandbox:** Code nur nach Nutzerfreigabe in einer ressourcenlimitierten Sandbox ausführen. – Vorschlag/nicht als fertig ausgewiesen
33. **[P1] Patch-vorher/nachher:** Jede vorgeschlagene Änderung als lesbaren Diff mit Testnachweis zeigen. – Vorschlag/nicht als fertig ausgewiesen
34. **[P1] Fehlgeschlagene Tests erklären:** Tracebacks reproduzierbar verkürzen und wahrscheinliche Ursachen abgrenzen. – Vorschlag/nicht als fertig ausgewiesen
35. **[P1] Abhängigkeits-Transparenz:** Vulnerabilitätshinweise nicht automatisch mit Versionsupdates gleichsetzen. – Vorschlag/nicht als fertig ausgewiesen
36. **[P1] Python-Formatcheck:** Syntax, Imports und Stil getrennt von funktionaler Korrektheit prüfen. – Vorschlag/nicht als fertig ausgewiesen
37. **[P1] Regex-Verifikator:** Reguläre Ausdrücke auf Nutzerbeispielen mit Negativfällen prüfen. – Vorschlag/nicht als fertig ausgewiesen
38. **[P1] SQL-Sicherheitsratgeber:** Statische Hinweise auf Parametrisierung ohne Datenbankzugriff anbieten. – Vorschlag/nicht als fertig ausgewiesen
39. **[P1] JSON-Schema-Modus:** Ausgaben strikt gegen ein vorgegebenes Schema validieren. – Vorschlag/nicht als fertig ausgewiesen
40. **[P1] Mikro-Benchmark:** Neue Codepfade hinsichtlich Zeit und Speicher auf kleinen künstlichen Inputs messen. – Vorschlag/nicht als fertig ausgewiesen

### Orbit & Cloud-Agenten

41. **[P1] Persistente Auftragswarteschlange:** Durable Tasks mit Lease, Retry, Idempotency und Status statt bloßem Cron-Heartbeat. – Vorschlag/nicht als fertig ausgewiesen
42. **[P1] Agenten-Rollenvertrag:** Planer, Entwickler, Tester und Reviewer bekommen getrennte Berechtigungen. – Vorschlag/nicht als fertig ausgewiesen
43. **[P1] Konfliktfreie Branch-Vergabe:** Jeder Task besitzt einen dedizierten Git-Branch samt geprüfter Head-SHA. – Vorschlag/nicht als fertig ausgewiesen
44. **[P1] Budgetloser Wartemodus:** Ohne KI-Kontingent pausieren statt teure APIs stillschweigend zu verwenden. – Vorschlag/nicht als fertig ausgewiesen
45. **[P1] Freigabe-Schranke für Aktionen:** Externe Schreibaktionen ausschließlich nach expliziter Nutzerbestätigung. – Vorschlag/nicht als fertig ausgewiesen
46. **[P1] GitHub-Actions-Kontingent-Schutz:** Maximale CPU-Minuten und erfolglose Wiederholungen transparent begrenzen. – Vorschlag/nicht als fertig ausgewiesen
47. **[P1] Automatischer Regressionstrigger:** Änderungen lösen nur fachlich passende Tests aus, kein permanenter CI-Sturm. – Vorschlag/nicht als fertig ausgewiesen
48. **[P1] Wiederanlauf nach Crash:** Abgebrochene Aufgaben konsistent als review- oder retry-bedürftig markieren. – Vorschlag/nicht als fertig ausgewiesen
49. **[P1] Agenten-Livejournal:** Pro Durchlauf Commit, Test, Arbeitsschritt, Fehler und Kostenbasis dokumentieren. – Vorschlag/nicht als fertig ausgewiesen
50. **[P1] Mehragenten-Review:** Unabhängiger Reviewer darf Code ablehnen, aber nie selbst auf main mergen. – Vorschlag/nicht als fertig ausgewiesen

### Datensatz-Qualität & Rechte

51. **[P2] Lizenz-Manifeste:** Jede Quelle erhält Version, Lizenz, Herkunft und erlaubte Trainingsverwendung. – Vorschlag/nicht als fertig ausgewiesen
52. **[P2] Semantische Split-Sperre:** Gleiche Aufgaben in TRAIN und TEST trotz Paraphrase erkennen und ablehnen. – Vorschlag/nicht als fertig ausgewiesen
53. **[P2] Gleichungskanonisierung:** Rechenaufgaben unabhängig von operand permutation und Sprache vergleichen. – Vorschlag/nicht als fertig ausgewiesen
54. **[P2] Qualitätsfilter-Ablation:** Datenfilter mit gleichem Budget gegen ungefilterte Stichprobe antreten lassen. – Vorschlag/nicht als fertig ausgewiesen
55. **[P2] DE/EN-Mischung messen:** Sprachanteile nach BPE-Tokens, nicht Dateigröße, vergleichen. – Vorschlag/nicht als fertig ausgewiesen
56. **[P2] Persönliche Daten abschirmen:** Geheimnisse, Telefonnummern und private Fragmente vor Datensatzfreigabe entfernen. – Vorschlag/nicht als fertig ausgewiesen
57. **[P2] Korpus-Digest:** Jeder Trainingslauf bekommt unveränderliche SHA-basierte Input-Manifeste. – Vorschlag/nicht als fertig ausgewiesen
58. **[P2] Teacher-Herkunft verifizieren:** Synthetische Antworten nur mit sichtbarem Modellstand und unabhängiger Referenz verwenden. – Vorschlag/nicht als fertig ausgewiesen
59. **[P2] Fehlerhafte Labels ausschließen:** Mathematik- und Formatlabels automatisch prüfen statt blind zu übernehmen. – Vorschlag/nicht als fertig ausgewiesen
60. **[P2] Datenvielfalt kontrollieren:** Überrepräsentierte Muster und repetitives SFT mit Diversitätsmetriken erkennen. – Vorschlag/nicht als fertig ausgewiesen

### Arctus350 – echtes eigenes Modell

61. **[P2] Checkpoint-Identitätsprüfung:** Architektur, Tokenizer, Optimizer und RNG vor jedem Resume exakt verifizieren. – Vorschlag/nicht als fertig ausgewiesen
62. **[P2] EOS-Genauigkeitsdiagnose:** Korrekte Stopp-Token-Rate unabhängig vom Antwortinhalt messen. – Vorschlag/nicht als fertig ausgewiesen
63. **[P2] Native OOD-Mathematik:** Neue Rechenaufgaben ohne Zahlenüberschneidung als harte Baseline verwenden. – Vorschlag/nicht als fertig ausgewiesen
64. **[P2] Assistant-only-SFT-Masken:** Nur erwünschte Antworttokens statt Nutzerfrage als Lernziel zählen. – Vorschlag/nicht als fertig ausgewiesen
65. **[P2] Gepaarte Trainingsvergleiche:** A/B mit identischem Start-Checkpoint, Daten, Schritten und eval-seeds durchführen. – Vorschlag/nicht als fertig ausgewiesen
66. **[P2] Vergessenheitsalarm:** Gewinne bei Mathe nur zulassen, wenn Chat/DE/EN/Format nicht unverhältnismäßig regressieren. – Vorschlag/nicht als fertig ausgewiesen
67. **[P2] Low-cost-Teacher-Curriculum:** Nur lizenzierte, unabhängig überprüfbare Lehrerbeispiele kuratiert verwenden. – Vorschlag/nicht als fertig ausgewiesen
68. **[P2] Small-to-large-Datenpilot:** Kleine Modelle zur Auswahl schwieriger SFT-Beispiele testen, ohne Lehrantworten zu erfinden. – Vorschlag/nicht als fertig ausgewiesen
69. **[P2] Parameter-vs-Token-Transparenz:** 349,8M Parameter und 2,07B gesehene Tokens als verschiedene Größen ausweisen. – Vorschlag/nicht als fertig ausgewiesen
70. **[P2] Skalierungs-Entscheidungsgate:** Erst nach belegter Generalisierung über ein größeres Arctus-Modell entscheiden. – Vorschlag/nicht als fertig ausgewiesen

### Inferenz & Trainingsgeschwindigkeit

71. **[P2] GPU-VRAM-Profil:** Qwen 9B und Arctus getrennt mit realen Peak-Memory-Messungen bewerten. – Vorschlag/nicht als fertig ausgewiesen
72. **[P2] Time-to-first-token:** Latenz bis zum ersten sichtbaren Wort getrennt vom Gesamt-Durchsatz messen. – Vorschlag/nicht als fertig ausgewiesen
73. **[P2] Context-Per-Byte-Metrik:** Kontextkosten anhand realer Tokenizerzahlen statt pauschaler UTF-8-Länge schätzen. – Vorschlag/nicht als fertig ausgewiesen
74. **[P2] Kompakte SFT-Batches:** Padding und nutzlose Loss-Tokens je Batch als Anteil überwachen. – Vorschlag/nicht als fertig ausgewiesen
75. **[P2] Flash-Attention-Check:** Kernelwechsel nur bei numerischer Parität und messbarer GPU-Beschleunigung. – Vorschlag/nicht als fertig ausgewiesen
76. **[P2] KV-Cache-Parität:** Cached und uncached Generierung bei festem Seed tokenweise vergleichen. – Vorschlag/nicht als fertig ausgewiesen
77. **[P2] Intelligente Modellwarmhaltung:** Qwen geladen lassen, aber VRAM und Idle-Kosten kontrollierbar begrenzen. – Vorschlag/nicht als fertig ausgewiesen
78. **[P2] Präziser OOM-Fallback:** Bei Speichermangel klare Optionen statt heimlich auf CPU/anderes Modell wechseln. – Vorschlag/nicht als fertig ausgewiesen
79. **[P2] Datenlader-Profiler:** Dekomprimierung, Tokenisierung und GPU-Idle-Zeiten getrennt erfassen. – Vorschlag/nicht als fertig ausgewiesen
80. **[P2] Kosten-pro-validem-Gewinn:** Trainingsbudget nur für reproduzierbar erfolgreiche Verbesserungen einsetzen. – Vorschlag/nicht als fertig ausgewiesen

### Helios-App & Nutzererlebnis

81. **[P1] Echte Quellenkarten:** Belege direkt neben der betroffenen Aussage, mit Originaltext-Verweis. – Vorschlag/nicht als fertig ausgewiesen
82. **[P1] Sichtbare Modellherkunft:** NERAL-Qwen, NERAL-Arctus und Rechnerantwort eindeutig kennzeichnen. – Vorschlag/nicht als fertig ausgewiesen
83. **[P1] Abgeschnittene Antwort markieren:** Unvollständige Qwen-Texte nicht als gültigen Chatkontext weiterverwenden. – Vorschlag/nicht als fertig ausgewiesen
84. **[P1] Gesprächsarchiv durchsuchen:** Lokale, dokumentierte Volltextsuche mit Lösch- und Exportfunktion. – Vorschlag/nicht als fertig ausgewiesen
85. **[P1] Projektbereiche:** Unabhängige Arbeitsbereiche für Programmierung, Forschung und Alltag. – Vorschlag/nicht als fertig ausgewiesen
86. **[P1] Antwortvergleich im Chat:** Zwei Qwen-Profile oder Toolvarianten nebeneinander prüfen. – Vorschlag/nicht als fertig ausgewiesen
87. **[P1] Verlaufs-Quellschutz:** Importierte und fremde Chats nie ungeprüft als Modellhistorie einspeisen. – Vorschlag/nicht als fertig ausgewiesen
88. **[P1] Tastatur- und Screenreader-Test:** Mobile Bedienung, Fokus, aria-Labels und Dialoge vollständig abnehmen. – Vorschlag/nicht als fertig ausgewiesen
89. **[P1] Offline-Fehlerdiagnose:** Ollama aus, Modell fehlt, GPU OOM und Cloud-Ausfall eindeutig unterscheiden. – Vorschlag/nicht als fertig ausgewiesen
90. **[P1] Installierbare PWA:** Offline-Shell, sichere Updates und korrektes Cache-/Versionsverhalten testen. – Vorschlag/nicht als fertig ausgewiesen

### Sicherheit, Verantwortung & Release

91. **[P0] Privatsphäre-Dashboard:** Für jede Funktion anzeigen, ob Daten lokal, in GitHub oder extern verarbeitet werden. – Vorschlag/nicht als fertig ausgewiesen
92. **[P0] Explizite Rechte je Werkzeug:** Dateisystem, Internet, Shell und private Quellen standardmäßig gesperrt lassen. – Vorschlag/nicht als fertig ausgewiesen
93. **[P0] Prompt-Injection-Red-Team:** Angriffe über Tool-Outputs, Webseiten und importierte Quellen systematisch testen. – Vorschlag/nicht als fertig ausgewiesen
94. **[P0] Supply-Chain-Überwachung:** Dependabot nur reviewed, keine automatischen unsichtbaren Versionssprünge. – Vorschlag/nicht als fertig ausgewiesen
95. **[P0] Wiederherstellungs-Drill:** Chat-DB und Konfiguration aus Backup tatsächlich wiederherstellen und prüfen. – Vorschlag/nicht als fertig ausgewiesen
96. **[P0] Release-Bundle-Signatur:** Checksums, Versions-ID und reproduzierbare Build-Metadaten veröffentlichen. – Vorschlag/nicht als fertig ausgewiesen
97. **[P0] Rollout mit Rückweg:** Beta-Feature-Flags und Rollback bei ernsten Fehlern bereitstellen. – Vorschlag/nicht als fertig ausgewiesen
98. **[P0] Transparente Modellkarte:** Trainingsstand, Grenzen, Lizenzen, Messwerte und externe Modellbasis erläutern. – Vorschlag/nicht als fertig ausgewiesen
99. **[P0] Getesteter Nullkostenmodus:** Keine zahlungsauslösenden API-, GPU- oder Speicheraktionen im Standardbetrieb. – Vorschlag/nicht als fertig ausgewiesen
100. **[P0] Freigabe-Checkliste:** Kein öffentliches Release ohne echten Geräte-E2E, Sicherheit, Datenrechte und bestandene Qualitätsgate. – Vorschlag/nicht als fertig ausgewiesen

## Ein Datum – aber mit überprüfbaren Gates

**Zieltermin für NERAL Closed Beta (Qwen-basiert): Sonntag, 31. Januar 2027.** Dies ist eine ambitionierte Zielmarke, keine Zusicherung; sie verschiebt sich, falls Sicherheits-/E2E-Gates scheitern. Ein eigenes Arctus350 auf GPT-3/3.5-Niveau hat **kein seriös terminierbares Datum**.

### Phasen

1. **01.–15.11.2026 – Integrationsstopp für neue parallele Basisbranches:** Bestehenden #173→#175→#203→#204→#210→#212→#213 Stack auf saubere gemeinschaftliche Reviews/Tests vorbereiten, keine Blind-Merges; Abhängigkeiten und CI-Gates dokumentieren.
2. **16.–30.11.2026 – reales Modell:** Ein explizit freigegebener Windows-/RTX4070-Test von Qwen3.5-9B durch die echte Helios-Oberfläche: Standard/Deep/Tools, Herkunft, Offline-Fall, OOM, Zeit/VRAM. Falls nicht freigegeben, verschiebt sich die Beta.
3. **01.–20.12.2026 – Erfahrung & Werkzeuge:** Top-P0-Fehler beheben; lokale Projekte, Kontext, Dateien/Quellen nur mit Nutzerfreigaben. Native Arctus-Studie separat, keine Gewichtsbehauptungen ohne GPU-A/B.
4. **21.12.2026–15.01.2027 – Härtung:** Unabhängige mehrsprachige 100+-Fall-Suite, menschliche Review, Datenschutz, PWA-/Windows-Abnahme, Exporte/Wiederherstellung und null kritische Blocker.
5. **16.–31.01.2027 – Closed-Beta-Gate:** Nur reviewed, getestete Features; abschließender Integrationstest auf dem signierten Release-Commit, keine offenen P0-Bugs.
6. **28.03.2027 – möglicher Public-Beta-Termin:** Nur bei belegter Stabilität, Datenschutzhinweisen, Nutzungsbedingungen und unabhängig überprüfbarer Modell-/Werkzeugherkunft. Nicht als Automatismus.

### Harte Closed-Beta-Akzeptanzkriterien

- Relevante vollständige CPU-, Browser- und Windows-Tests für den **tatsächlichen** integrierten Release-Commit bestanden; mindestens ein realer lokaler Qwen-End-to-End-Lauf mit RTX4070 nach Nutzerfreigabe dokumentiert.
- Eine neue unabhängige 100+ Fälle-Suite mit DE/EN, Toolmodus, Coding, Unsicherheit und Offlinemodus. Zielbereich: >=80% der spezifizierten Kernaufgaben, 100% der sicherheits- und identitätskritischen Pflichtfälle, alle manuellen Prüffälle bewertet. Die Zielquoten sind **Release-Anforderungen, keine gegenwärtigen Scores**.
- Keine falsch als native Arctus-Antwort gekennzeichnete Qwen-/Rechnerleistung; kein automatischer Transfer externer Daten ins Training.
- Keine durch Tests bereits bekannten offenen P0-Fehler; vollständige Chat-Export-/Löschfunktion und kontrollierter Offline-/OOM-Fall.
- Auf dem Nutzer-PC keine ungefragten Downloads, kostenpflichtigen Jobs oder Gewichtsänderungen; GitHub Actions Quota und Cloudflare Kosten separat geprüft.

## Was sofort wirklich umgesetzt wird

PR #213 erweitert aufbauend auf #212 den bereits vorhandenen lokalen Qwen-Assistenten um einen begrenzten Werkzeugpfad: **höchstens zwei unterschiedliche** vorhandene Funktionen `calculator` und `code_check` nacheinander in einem Turn; die dritte / wiederholte / fremde Aktion wird abgewiesen. Der bestehende 2022-DEV-Gate prüft die getrennten Werkzeugursprünge. Kein Shell-Zugang, kein Internet, keine zusätzliche Modellbasis, keine Trainingskosten, keine stillschweigenden Codeausführungen. Erst nach grüner CI und echtem Nutzer-PC-E2E gilt es als funktional bestätigt.

---
**Gesamt:** 100 Ideen, 1 tatsächlicher Implementierungsschritt in diesem Draft, viele bereits vorhandene Vorarbeiten müssen integriert werden. Externe Quellen und vorgeschlagene Features sind keine automatisch erlaubten Trainingsdaten.


## Fortschritt am 01.11.2026: begrenzter Codeprüfer-Kontext

Beim Test des neuen Zweifach-Werkzeugpfads wurde ein reproduzierbarer Produktengpass festgestellt: 360 riskante `eval`-Vorkommen in einem zulässigen ca. 3,6-KB-Python-Snippet erzeugten ca. **63,9 KB** zurückgesendete statische Prüfbefunde. Das ist für den begrenzten Qwen-Kontext übermäßig groß.

Der bestehende `code_check` bleibt unverändert; ausschließlich die Ausgabe an das Qwen-Werkzeug `_code_check` ist nun auf **4096 UTF-8-Bytes** beschränkt. Bei gekürzter Ausgabe werden alle ursprünglichen `finding_count`- und `severity_counts`-Werte beibehalten, `omitted_findings` sowie `truncated=true` explizit mitgeliefert und schwere Befunde vor leichteren angezeigt. Kein Code wird ausgeführt, es gibt keine Aussage über vollständige Code-Sicherheit. Kleine normale Resultate werden unverändert übertragen.

**TDD:** Der RED-Test belegte unzulässige Ausgaben von **63.880** beziehungsweise **17.093** Bytes (GitHub-Run 41325727101). GREEN wurde mit einer kleinen isolierten Änderung implementiert und muss mit allen vorhandenen CPU-/Browser-Regressionen verifiziert bleiben. Diese Änderung verbessert die Zuverlässigkeit der Werkzeugnutzung; sie trainiert **keine** Modellgewichte.

**Integrationsgrenze:** Die Closed-Beta-Termine setzen weiterhin den echten Windows-/GPU-/Browser-End-to-End-Test auf einem abgestimmten Release-Commit und vollständige unabhängige Nutzerabnahme voraus. Auch 100 geplante Features ersetzen diesen Schritt nicht.
