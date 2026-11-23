# NERAL V4.2 → V4.3: Freeze-Gates und vorbereiteter Ausbau

Status: **NERAL V4.2 / NERAL 1.0 eingefroren – 23. November 2026**.
Geprüfter Quellstand: `c0a6be607a0797908e101db460e3ac297a70f088`;
Release-Referenz: `release/neral-1.0-v4.2-frozen-20261123`.
Der Freeze bezieht sich auf diesen Commit, nicht auf den fortlaufenden
`main`-HEAD. Die Release-Referenz ist ein separater Branch, kein
geschütztes Tag. Alle späteren Math-/Codex-Änderungen gehören in V4.3.

### Abnahmeprotokoll (öffentliche synthetische Diagnostik)

[Einmaliger GitHub-Actions-Lauf #45256370458](https://github.com/Cola968/neral-helios/actions/runs/45256370458):
**completed / success**. Trigger-Commit des Pin-only-PR #620:
`28a4f197ee89f2f1951cf79b139b3473eda00e10`.
Alle sechs Checkouts verwenden das Quell-SHA oben. Ubuntu-CPU-Preflight,
Ubuntu-/Windows-Standalone-ZIP und W39-Owner-Deployment waren erfolgreich.

| Modus | Test | Status | Latenz | Peak CUDA reserviert | Peak GPU gesamt |
| --- | --- | --- | ---: | ---: | ---: |
| Quartz | Alltag, Kondensation | `two_sentence_condensation_explanation` | 6,70 s | 7.716 MiB | 9.061 MiB |
| Quartz | Mathematik | `exact_numeric_match` (345) | 12,11 s | 7.866 MiB | 9.219 MiB |
| Quartz | Webwissen Apollo 11 | `exact_year_and_read_https_quote` (1969) | 13,22 s | 7.712 MiB | 9.065 MiB |
| Obsidian | Alltag, Kondensation | `two_sentence_condensation_explanation` | 5,61 s | 7.736 MiB | 9.097 MiB |
| Obsidian | Mathematik | `exact_numeric_match` (345) | 18,45 s | 7.832 MiB | 9.201 MiB |
| Obsidian | Webwissen Apollo 11 | `exact_year_and_read_https_quote` (1969) | 10,30 s | 7.830 MiB | 9.199 MiB |

Fachlich und technisch **6/6 HTTP 200**, keine 409; beide Quellenbelege
wurden serverseitig mit einer tatsächlich gelesenen HTTPS-Seite abgeglichen
(`year_source_pipeline_code=candidate_available`,
`source_year_match=true`, `year_quote_in_final=true`).
Die zwei Alltagsaufgaben wurden nur durch die begrenzte
Kondensations-Heuristik bewertet, nicht durch ein unabhängiges Human-Review.

Die NVIDIA-WDDM-Schnittstelle lieferte für den exakten Prozess-VRAM
`[N/A]`. Der im Protokoll als `cuda_reserved_wddm_fallback` markierte
Wert ist die PyTorch-CUDA-Reservierung, **keine direkte Messung des
gesamten NERAL-Prozess-VRAM**. Der separate, systemweite VRAM blieb in
allen sechs Aufgaben unter 10.752 MiB. Pre-Load-Baseline nach StopOwned:
`free=10855 MiB`, `used=1157 MiB`; Post-Load mindestens 1.000 MiB
frei, keine konkurrierenden Python-/LLM-Compute-Prozesse.

Rollback auf den unveränderten ursprünglichen Besitzer von Port 9690:
**success**, am 23.11.2026 um 20:14:18 UTC; bestätigt durch
`NERAL_V42_SMOKE5_ORIGINAL_OWNER_RESTORED_HTTP200` und
`NERAL_DIAG3_ORIGINAL_OWNER_RESTORED_HTTP200`. Ein anderer GPU-Lauf
oder ein 50-Aufgaben-Vergleich wurde für diesen Freeze nicht ausgeführt.
Die 421 bestehenden Blind200-Ergebniszeilen blieben unangetastet.

Die sechs öffentlichen Proben sind ein Release-Smoke-Test, **kein**
Nachweis allgemeiner GPT- oder Codex-Leistungsparität. 

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
- GPU-Pre-Load nach StopOwned und vor StartOwned: mindestens
  **8.500 MiB frei**. Post-Load nach authentifiziertem NERAL-HTTP-200:
  systemweit **höchstens 10.752 MiB belegt** und **mindestens 1.000 MiB
  frei**; kein konkurrierender Python-/LLM-Compute-Prozess.
  NVIDIA-Prozess-VRAM nur als direkt gemessen ausweisen, wenn verfügbar.
  Unter Windows WDDM `[N/A]` die tatsächliche PyTorch-CUDA-Reservierung
  separat als **enger begrenzten Ersatzindikator** ausgeben und gegen
  9.500 MiB prüfen; System-GPU-VRAM immer getrennt protokollieren.
  Fremde Prozesse niemals automatisch beenden.
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
