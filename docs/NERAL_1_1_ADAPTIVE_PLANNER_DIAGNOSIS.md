# NERAL 1.1 – Offline-Diagnose und Adaptive-Planner-Setup

Stand: 15.11.2026. Isolierter Entwicklungsbranch von 737413f03b347676108d507ff3fe7693975dd269 (PR #542). Kein Merge auf main, keine Änderung des ursprünglichen PR-#542-Commits, keine Änderung der bestehenden Blindtest-Korpora und kein neuer GPU-Lauf.

## Originale Offline-Evidenz

- GitHub Actions: Lauf 43722550890, Rohdaten-Artefakt 14166222735.
- Geprüfte unveränderte Rohdatensätze: 200 gepaarte Antworten.
- Rohdaten SHA-256: e1cb153c59d0712b63ad77df86074ac2b763ee85e6ee8e2ccea099bae72141f9.
- Datensatz-Siegel SHA-256: c43287a4ca6eb18fe4f45a45706f7af8155d27033638bb13c397a7417aacb157.
- Blind100: Qwen 68/100, NERAL 77/100, +9, keine Regression im Test; strenges Release-Gate nicht bestanden.
- NERAL nicht gelöst: Logik 17/25, Mathematik 6/25.
- Alle 25 Logikfälle wurden dem nominellen Pfad logic_z3 zugewiesen, aber in allen 25 Fällen laut Originalmetadaten 0 Schema-Extraktionen (extraction_attempts leer, calls_used=1); 0 quellverifizierte Z3-Ergebnisse. Die interne Coverage-Prüfung wurde deshalb nicht erreicht. Ein coverage_reason ist im Archiv für 0/25 vorhanden. Eine Einteilung nach konkreten coverage_reason-Werten wäre erfunden.
- 17 fehlgeschlagene Logikfälle: alle mit explicit_safe_refusal und base_fallback_token_limit_unverified. Die jeweils zugehörige Qwen-Baseline meldete token_limit.
- 6 fehlgeschlagene Mathefälle: 6 token_limit-Antworten sowohl bei NERAL als auch bei der ursprünglichen Baseline.
- In der Messimplementierung wird done_reason=length auf token_limit abgebildet. Die Daten belegen aber nicht, dass ein think-Tag den Abbruch verursacht hat.

Fehlerfamilien:

| Aufgabenklasse | Nicht gelöst |
| --- | ---: |
| four_color_chain | 5 |
| nine_digit_access | 5 |
| bounded_triples_mod3 | 5 |
| order_with_final_exclusion | 2 |
| modular_power (Mathematik) | 3 |
| bounded_distributions (Mathematik) | 3 |

## Ursache und nächste Strategie

PR #542 erlaubt die Quellverifikation nur, wenn strict_source_certificate eine komplette Formulierung in einer eng definierten Graph-, Binär- oder PIN-Grammatik unabhängig parsen kann. Die neuen Logikfamilien erreichten die Modell-Schema-Extraktion überhaupt nicht. Das ist in erster Linie fehlende Volltext-Zertifizierungsabdeckung, nicht automatisch ein Mangel der v2-AST-Syntax.

Die v2-AST kennt Summe, Modulo, Vergleiche, all_different, genau-k Kardinalitäten, Negation und Boolesche Verknüpfungen. Farbketten, Ordnungs- und einfache Modulo-Bedingungen können grundsätzlich als solche Prädikate darstellbar sein, auch wenn die natürliche Formulierung noch nicht zertifiziert werden kann. Dagegen fehlt ein Potenzoperator für modular_power; außerdem ist der endliche Solver auf 100.000 Belegungen begrenzt, was größere 9-stellige Suchräume unpraktisch macht. Eine Python-Brücke mit gezielter Enumeration oder Backtracking ist sinnvoll, wenn und nur wenn ein unabhängiger Parser sämtliche Bedingungen und die Vollständigkeit der Suche prüft.

## Isolierter NERAL-1.1-Controller

Datei: neral_adaptive_planner.py.

- Deterministischer, günstiger Vorfilter; keine Fall-IDs und kein Blick auf Gold-Antworten.
- Ein Vorschlag enthält Strategie und Schema beziehungsweise Python-Code gemeinsam. Kein separater Planungsmodellaufruf; der aufrufende Adapter muss denselben residenten 9B-Prozess für Korrekturen verwenden.
- Strategien: Z3_SCHEMA, PYTHON_SANDBOX, DIRECT.
- Z3 nur mit unabhängigem vollständigem Quellzertifikat, Coverage-Prüfung und Z3-Äquivalenzbeweis. Python nur in der vorhandenen Docker-Sandbox; kein Host-Exec.
- Die Python-Ausführung allein hat nur den Status execution_only. Eine Verifikation gegen alle Textbedingungen und den gesamten Suchraum muss unabhängig vom Modell erfolgen.
- Maximal 2 Reparaturrunden nach dem ersten Versuch. Reparaturfeedback enthält nur fest definierte Fehlercodes, keine Rückgaben, Rohaufgaben, Tracebacks oder Gold-Werte.
- Ohne unabhängigen Beweis wird ausschließlich auf den ursprünglichen, unveränderten Qwen-Aufruf mit den Originalnachrichten zurückgefallen. Fehlende oder unvollständige Basisantworten enden in einer klaren Ablehnung.

**Begrenzungen:** Dies ist ein testbarer Controller-Prototyp, noch nicht mit dem Chat-/THINK-Endpunkt verbunden. Der produktive unabhängige Python-Volltext-Verifier fehlt. Ein CPU-Test mit Mock-Verifier ist kein Beleg, dass natürliche Logiktexte bereits korrekt übersetzt werden können. Die Behauptung "keine zusätzliche Latenz" wäre ebenfalls falsch: Der Planungsaufruf entfällt, aber Codeausführung, Reparaturrunden und Fallback können Zeit kosten.

## Ausführung ohne GPU oder neue Blindtestdaten

Das vorhandene Original-ZIP kann ausschließlich lesend und ohne Ausgabe von Aufgaben- oder Antwortinhalten geprüft werden:

    python -B -m scripts.neral_1_1_offline_aggregate /pfad/zum/original-artefakt.zip

CPU-CI: neral-1-1-adaptive-planner-cpu.yml umfasst die neuen synthetischen Controller- und Integritätstests sowie die 16 bestehenden PR-#537/#542-CPU-Suiten. Erst nach erfolgreicher CI, zusätzlichem unabhängigem Quell-Verifier und gesondertem neuen Versiegelungsverfahren kommt ein frischer GPU-Blindtest infrage. Ziel 85/100 ist ein Entwicklungsziel, kein bereits bewiesenes Resultat.


## Schritt 2 – Unabhängiger Python-Logik-Verifier (synthetische CPU-Tests)

Implementiert im isolierten Branch:

- neral_python_logic_verifier.py: parse_bounded_logic_source, independently_count, independent_python_certificate.
- tests/test_neral_python_logic_verifier.py: 19 neue synthetische CPU-Verträge, darunter Gegenbeispiele, vergessene Zusatzbedingungen, ungültige Referenzen, Ressourcenlimits, exakte Ausgabe und Integration mit dem experimentellen Controller.
- Unbekannte Klauseln oder Format-Anweisungen werden nicht stillschweigend entfernt: Der vollständige Eingabetext muss innerhalb der unterstützten Grammatik geparst werden.
- Die maximale vollständige Suche umfasst 120.000 Belegungen. Der unabhängige Oracle-Code verwendet keine Modell-Goldantworten, keine Blindtestfälle, keine Ausführung generierten Codes, keine GPU und keine Netzwerkanfragen.
- Die Ausgabe der Docker-Sandbox wird nur dann als unabhängig geprüft markiert, wenn sie mit dem vom Oracle selbst errechneten Ergebnis exakt übereinstimmt. Die Python-Ausführung allein bleibt weiterhin execution_only.
- Der neue Verifier ist derzeit auf bestimmte englische, exakt strukturierte Sätze beschränkt. Beispielgrammatiken: Farbketten mit explizit unterschiedlichen Nachbarn und festen/verbotenen Positionen; Permutationen mit vor/nach/erstem/letztem Platz und Nachbarschaftsverbot; Tupel mit Modulo-Summe, paarweise verschiedenen Werten und fester Anzahl gerader Einträge.
- Keine freie deutsch-/englischsprachige Allzweck-Aufgabeninterpretation. Unter anderem beliebige Zahlentheorie, komplexe Farbgraphen, allgemeine Rätseltexte und andere Zusatzbedingungen bleiben nicht verifiziert.
- Der Controller ist nicht in die produktive THINK-API eingebunden. Die neue Verifier-Signatur wurde auch nicht in den unveränderten PR-#537-Guard eingebaut; das müsste vor einer Live-Aktivierung als eigene nachweisbare Sicherheitsgrenze untersucht werden.
- Ein zusätzlicher Modellaufruf nur für Planung ist für diese CPU-Vorbereitung nicht erforderlich. Aussagen über reale Modelllatenz sind mangels GPU-Messung nicht möglich.

Beispiel einer vollständig erkannten synthetischen Eingabe:

    Count permutations of numbers 1 through 4. Number 1 is before number 2.

Deren unabhängig ermitteltes Ergebnis ist 12. Wird an den Text etwa eine unbekannte weitere Bedingung angehängt, ist die Zertifizierung gesperrt, selbst wenn die Python-Sandbox 12 liefert.

Die Tests basieren ausschließlich auf neuen synthetischen Eingaben; eingefrorene Blindtests wurden weder erneut gestartet noch als Trainings-/Entwicklungsdaten benutzt. Die Zielmarke von 85/100 wird durch diesen CPU-Schritt weder belegt noch prognostiziert.
