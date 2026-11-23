# NERAL V4.2 → V4.3: Freeze-Gates und vorbereiteter Ausbau

Status: **Freeze-Kandidat, nicht freigegeben**. Dieses Dokument ist eine
Abnahme- und Entwicklungsplanung. Ein HTTP-200-Ergebnis allein erlaubt
weder ein Qualitätszertifikat noch einen Release-Freeze.

## V4.2-Freeze: unveränderliche Kriterien

- Genau ein freigegebener GitHub-Actions-3+3-Lauf auf sechs identischen,
  geprüften Checkout-SHAs; SHA nach dem Squash-Merge explizit setzen.
- CPU-Preflight und Standalone-ZIPs auf Ubuntu und Windows grün.
- Quartz und Obsidian jeweils drei öffentliche Diagnosefälle:
  Alltag: Zweitsatz-Erklärung von Kondensation (begrenzter Heuristiktest,
  zusätzliche manuelle Fachprüfung bleibt sinnvoll); Mathematik: exaktes
  `ENDERGEBNIS: 345`; Web: `1969` im Antwortteil plus wörtlich
  angezeigtes, serverseitig gegen eine **gelesene HTTPS-Seite** verifiziertes
  Zitat mit `year_source_pipeline_code=candidate_available`.
- NERAL-Prozess-VRAM **gemessen**, über sämtliche Samples **unter 9.500 MiB**.
  Systemweiter WDDM-/Desktop-VRAM separat berichten. `N/A` ist keine
  bestandene Speichermessung; fremde GPU-/Python-Prozesse vor dem Lauf nur
  identifizieren, niemals automatisch beenden.
- Automatischer Rollback: verifizierter ursprünglicher Besitzer von Port 9690,
  HTTP 200 und Rollback-Log. Ausfälle führen zum Status **nicht eingefroren**.
- Protokoll: geprüfter Squash-Commit, Lauf-ID, sechs CPU-/ZIP-Statuswerte,
  sechs fachliche Prüf-Codes, VRAM-Prozess- und Gesamtspitzen und Rollback.
  Keine Aufgaben-/Antworttexte, Trainingsdaten oder 421 Blind200-Zeilen in
  Artefakten.

## 50-Aufgaben-Quarz-vs.-Obsidian-Vergleich (nur vorbereitet)

**Nicht gestartet.** Neue, bislang ungesehene, private 50er-Suite,
vorab lokal versiegelt, keine Wiederverwendung der 421 gespeicherten
Blind200-Ergebnisse oder der öffentlichen 3+3-Diagnoseprompts. Beide Modi
bekommen denselben Aufgabensatz in randomisierter, gegengebalancierter
Reihenfolge. Zu prüfen: Korrektheit, Quellenqualität, Reparaturraten,
p50/p95-Latenz, Prozess-/System-VRAM, Ausfälle und 409. Keine Änderung an
Modellgewichten durch den Vergleich.

Vorgeschlagene Vergleichsverteilung (proportional zur alten **Kategorie-**
verteilung, nicht auf Basis alter Antworten):

| Kategorie | Aufgaben |
| --- | ---: |
| Alltag/Erklärungen | 9 |
| Multi-Turn | 10 |
| Aktuelles Webwissen | 9 |
| Mathematik und Logik | 7 |
| Programmierung | 6 |
| Schreiben | 5 |
| Unsicherheit und Grenzen | 4 |
| **Gesamt** | **50** |

Geschriebene/qualitative Antworten bleiben blind und unabhängig
bewertet; reiner HTTP-Erfolg oder Modell-Selbsteinschätzung sind keine Note.
Abnahme erfordert getrennt auswertbare Scores und Fehlerklassen.
Ein neuer Benchmark ist **nicht** durch den 3+3-Release-Lauf autorisiert.

## Nächster isolierter Entwicklungszweig: Obsidian Math & Codex

V4.2 bleibt unangetastet. Erst auf einem getrennten V4.3-Branch:

**Obsidian Math** – Modulschnittstellen `math_router`,
`z3_constraints`, `sympy_symbolic`, `lean4_proof_check`:
Modellgenerierte Vorschläge werden von deterministischen Tools getrennt.
Z3 löst entscheidbare Bedingungen und prüft Modelle; SymPy führt
symbolische Ausdrücke unter AST-/Größenlimits aus; Lean 4 kompiliert
Proof-Terms in einem isolierten Prozess. Kein Beweisstatus aus
einem plausiblen Modelltext; Prozesslimit, Timeout, Speicherlimit und
Rückverfolgbarkeit zum Nutzerproblem sind verpflichtend.

**Obsidian Codex** – `codex_sandbox`, `pytest_runner`,
`hypothesis_properties`: generierten Python-Code syntaktisch prüfen,
nur offline und isoliert ausführen, vorab definierte Tests mit `pytest`
und zusätzliche wohldefinierte Eigenschaften mit `Hypothesis` prüfen.
Fremde Netzwerk-/Dateizugriffe, Paketinstallationen und unkontrollierte
Subprozesse im Test-Sandbox-Profil sperren. Ergebnis-Codes trennen
`syntax_invalid`, `tests_failed`, `property_counterexample`,
`timeout`, `sandbox_error` und `passed`.

Die V4.3-Module starten **deaktiviert** und bekommen eigene CPU-Tests und
separate Akzeptanzkriterien. Ein Absturz in Math/Codex darf weder die
eingefrorene V4.2-API noch ihre Web-/Quellenlogik verändern.
