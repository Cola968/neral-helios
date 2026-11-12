# NERAL 1.1 — konservative Antwortfreigabe und unabhängiges 160er-Gate

Stand: 2026-11-12. Experimenteller CPU-geprüfter Kandidat; **kein GPU-Qualitätsnachweis und kein Merge**.

## Konkreter Fehler und Schutz

Im gepinnten PR-#533-TIR12-Entwicklungsvergleich: NERAL HIGH+TIR **9/12**, direktes Qwen3.5:9B **7/12**, vier paarweise Gewinne, zwei Verluste. Bei vier frischen synthetischen Logik-Transferaufgaben war NERAL nur **1/4** korrekt, u. a. wegen einer formal zulässigen, aber semantisch falschen Modellübersetzung. Das beweist keinen allgemeinen Überlegenheitsgewinn.

Neuer **optionaler** HTTP-Parameter `conservative_tir: true` bei `POST /api/neral/chat/think`. Nur zusammen mit `experimental_tir: true` und dem bereits nötigen lokalen Umgebungs-Gate `NERAL_EXPERIMENTAL_TIR=1` zulässig. Ohne Aktivierung bleibt der bestehende TIR-Vergleich unverändert.

Regel: Die unabhängige Verifikation der **unveränderten Nutzer-Python-Quelle** durch AST-Äquivalenz + offline Sandbox oder die explizite, vollständig vom Nutzer gelieferte formale Schemaaufgabe darf eine TIR-Antwort gegenüber Qwen priorisieren. Ein vom Modell vorgeschlagenes Logik-Schema, ein erfolgreiches Werkzeug, eine selbst behauptete Sicherheit oder eine generierte Abschlusserklärung allein **genügen nicht**. Bei allen übrigen nicht verifizierten Kandidaten wählt der Guard eine frische Antwort des **unveränderten direkten Qwen-Modells**, mit Original-Dialog, demselben HIGH-Modellprofil und ohne Werkzeug-Prompt. Wenn TIR bereits auf raw Qwen zurückgefallen ist, keine doppelte Inferenz. Eine explizite deterministische Sicherheitsablehnung (z. B. Sandbox nicht sicher ausgeführt, ungültige formal explizite Bedingungen) bleibt eine Ablehnung; kein Modellraten danach. Fehlformatierte Raw-Antworten werden abgelehnt.

Mehr Rechenzeit und Latenz sind möglich, weil bei unverified Kandidaten ein zusätzlicher Qwen-Lauf nötig ist. In der Praxis kann der Guard auch TIR-Gewinne verwerfen. Ohne semantischen, unabhängigen Prüfer gibt es keine magische sichere Auswahl; deshalb **kein Versprechen einer immer besseren Antwort**. Ein deterministisch geprüfter Fehlerfall bedeutet nur, dass die begrenzte Prüfung erfolgreich war, nicht dass sämtliche Nutzerintentionen verstanden wurden.

## Abnahme auf 160 neuen, wirklich separaten Aufgaben

Nur getrennt angelegte, bisher weder trainierte noch für Heuristiken/Regressionsfixes konsultierte Aufgaben. **Keine** 12er-Entwicklungsfälle, die vier Transferfälle, bestehende 24er QLoRA-Holdouts, deren Lösungen oder paraphrasierte Varianten in dieses Gate kopieren. Keine neuen Prompt-Schlüsselwortregeln für dessen Aufgaben. Bei jeder Einsicht zum Fixen wird die betroffene Aufgabe aus dem abschließenden Blindtest entfernt und durch eine neue, unberührte ersetzt.

Vorbereiten: 40 Mathematik-/Wissenschaftsaufgaben mit unabhängiger exakter Referenz; 40 kombinatorische Logikaufgaben mit **unabhängiger** Wahrheitsbelegung/Exhaustiv-Oracle; 40 Python-/Code-Trace-Aufgaben mit sicherer Referenzausführung; 40 neue gemischte Aufgaben aus Planung, komplexer Analyse, Wissenschaft und Wissen mit unabhängig überprüfbaren Bewertungskriterien. Menschlich prüfen, dass Aufgaben eindeutig, verschiedenartig und fair bewertet sind. Für subjektive Fragen eine vorab fixierte Rubrik statt nachträglicher Modellpräferenzen.

Jede Aufgabe gepaart auf **demselben eingefrorenen Qwen3.5-Digest, derselben RTX 4070, identischem ursprünglichem Prompt/Verlauf, gleichem 2048-Token-Raw-Profil und dokumentiertem per-case Seed** prüfen, mit alternierender Reihenfolge: (A) direktes Qwen, (B) bestehendes experimentelles TIR, (C) konservatives TIR. Die zusätzlichen TIR-/Tool-Inferenzen zählen separat; bessere Qualität ist kein Beleg für bessere Effizienz. Rohtexte/Goldwerte bleiben privat; GitHub erhält höchstens aggregierte Metriken und Hash-Commitments.

**Strict Promotion Gate** (politisches Produktziel, kein mathematisches Garantieversprechen): auf allen 160 Aufgaben **0 gepaarte Verluste** für C gegen A, mindestens **8 gepaarte Gewinne** für C gegen A, keine Sicherheits- oder Sandbox-Bypasses und keine erhöhte HTTP-Fehlerquote. Dazu exakter gepaarter McNemar-Test, ein geeignetes Konfidenzintervall für die Differenz, Familien-Scores und p50/p95-Latenz sowie durchschnittliche GPU-/Tokenkosten berichten. Ein einziger beobachteter Verlust verhindert die Promotion; Entwicklungsdiagnose erfolgt nur auf separaten DEV-Aufgaben.

Auch 0/160 beobachtete Verluste beweisen nicht 0 % Fehlerrisiko in der Zukunft; grob liegt bei null beobachteten Ereignissen die einseitige 95-%-Obergrenze der tatsächlichen Verlustrate unter unabhängigen ähnlichen Beobachtungen noch bei etwa **1,9 %** (3/n-Näherung). Zudem kann das gleiche Modell aufgrund von Sampling und Mehrfachaufrufen variieren.

## Aktueller Freigabestatus

- CPU Guard und HTTP-Verträge: positive RED/GREEN-Proofs, ohne echte Modellgewichtänderung.
- RTX 4070 mit `conservative_tir=true`: **nicht ausgeführt**.
- Neuer unabhängiger 160er-Blindtest: **nicht aufgebaut und nicht ausgeführt**.
- PR-Stack: #505 (Draft) → #533 (noch nicht in main) → Guard-PR (Draft). QLoRA-Zweig #536 liegt parallel auf #533 und ist separat zu bewerten.
- Vor Merge: vollständiger gezielter CPU-/HTTP-/Sicherheits-Check, separater GPU-Pilot, das neue Blind-Gate, Review und Branchschutz. Kein automatisches Training, kein Release-Claim.

