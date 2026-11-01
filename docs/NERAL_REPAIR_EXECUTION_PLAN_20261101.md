# NERAL: konkreter Reparaturpfad bis zur geschlossenen Beta

Stand 2026-11-01; dies ist ein **operativer Reparaturplan**, kein Nachweis, dass alle Probleme schon beseitigt sind. Basis: vollständiges Issue-/PR-Inventar in `docs/NERAL_OPEN_PROBLEMS_AUDIT_20261101.md`; der Haupt-Release-Kandidat ist Draft #215 gegen `main`.

## Priorität P0 – ein stabiler Qwen-basierter NERAL-Kern

| Blocker | Wie wir ihn beheben | Abnahmebeleg | Abhängigkeit |
|---|---|---|---|
| 110 offene PRs, 90 gestapelt | **Kein Mass-Merge.** #215 als einzige Core-Integrationsquelle; PR-Änderungen nur selektiv anhand Datei-/Branch-Abhängigkeit übernehmen. Für jeden Kandidaten: Head-SHA, geänderte Dateien, vorhandene Tests, Rechte/Lizenzen, ob bereits in #215 enthalten. Historische Alternativen nach Review archivieren. | Ein signierter integrierter Commit, alle relevanten CI-Jobs grün, menschlicher Review, Entscheidung zu jedem enthaltenen PR. | GitHub-Review; Merge bleibt gesperrt ohne Nutzerfreigabe |
| Qwen-Live-Funktion und Tool-Protokoll | Die echte Codeprüfungs-HTTP400-Ursache und die strikte Metadatenhülle sind **bereits repariert**; Windows-Live #41329964479 3/3. Keine Neuentwicklung. Als Nächstes: 100+ unabhängige DE/EN-Fälle, Deep-Mode, wiederholte Toolnutzung innerhalb erlaubter Limits, 8K Kontext, offlines/Timeout/OOM und Herkunft verifizieren. | Separate Modell-/App-Passraten nach Kategorie und unveränderlichem Modell-Digest; 0 falsch zugeordnete Modell-/Tool-Antworten. | Nutzerfreigabe für zusätzliche lokale GPU-E2E-Jobs, unabhängige Fälle |
| SOLVE-Bereichsgrenzen | Überlauf beim Zusammenfassen linker/rechter linearer Gleichungsseiten **bereits behoben**: 4 RED→GREEN-Boundary-Regressionen, Kompatibilitätsfälle. | CI #41331306141 / #41331306135 grün. | Keine |
| Datenschutz-/Launch-Check | Bereits entwickelte P0-Kontrollen aus #206 selektiv in #215 übernehmen: sechs Risikobereiche, Fail-closed static scan, sessiongeschützte Read-only-API und Text-only-UI; keine rechtliche Scheinsicherheit. | `python -m pytest -q tests/test_launch_check.py tests/test_launch_check_flask.py`, `python launch_check.py --fail-on-blocker`, echte Browser-Netzwerkanalyse. | Menschlicher Rechts-/Datenschutzreview für öffentlichen Betrieb |
| Windows-/Mobile-App-Abnahme | Frische lokale Installation und echte Helios-Interaktionen testen: Chat, Import/Export, Tab/Reload, Historie, Android 320px, Offline-PWA, Tastatur/Screenreader, Sync-/Backup-Restore, GPU-Ausfall, Neustart nach Modellfehler. Keine echten Nutzer-Chats in CI. | Signiertes Testprotokoll pro Gerät/Browser; keine offenen P0-Bugs. | Freigegebener Ziel-PC, Geräte; Sandbox-/Testdaten |
| Quellen/Web/PDF/Video | Nur als **separates Beta-Feature**: untrusted Inhalte isolieren, HTTPS-/Redirect-/SSRF-Kontrolle, Dateigröße/Parsing-Limits, Zitate mit Seiten-/Zeitmarken, Prompt-Injection-Gates. Keinesfalls alle Quellen als Systemprompt oder Trainingsdaten verwenden. | Feste Negativtests für prompt injection, lokale Dateien, Netzwerkziele, unzulässige Datenweitergabe. | Explizite Berechtigung für Internet-/Dateiwerkzeuge |
| Auslieferung und Rollback | Checkpoints/Tokenisierer/Buildartefakte durch SHA festhalten; keine automatische Freigabe über bloße grüne Unit-Tests. Abnahme-Checkliste, DB-Backup-Restore, Release-Notes, Rückrollpfad. | Eindeutige SHA und versionsgebundene Testläufe; manueller Release-Gate. | Nutzerreview/Merge/Deployment-Freigabe |

## Priorität P1 – eigene Arctus350-Fähigkeiten nachweisbar verbessern

**Nicht mit Qwen3.5-9B verwechseln.** Der bestehende Arctus350-SFT-Stand ist kein GPT-3/3.5-Äquivalent. Problem: schwache Neuaufgaben-Generalisierung, frühere Überanpassung und Regressionen nach längeren Trainingsläufen.

1. **Einmalig native Baseline einfrieren:** Checkpoint, Tokenizer, RNG, Lizenz-/Korpus-Manifeste und SHA + 100+ unveröffentlichte Multi-Task-VAL/OOD-Beispiele (nicht im Teacher-Input oder Trainingskorpus).
2. **Hypothesen einzeln vergleichen:** Semantische Train/VAL-Deduplizierung, genaue Assistant-Lossmasken, ausgewogene Chat/DE/EN/Mathe-SFT, Datenqualitätsfilter, EOS-/Prompt-Echo-Diagnose. Ausschließlich CPU-Plan/Review, bevor GPU-Freigabe vorliegt.
3. **Reproduzierbar gepaart messen:** Gleiche Initialgewichte und Eval-Aufgaben, kontrollierte Seeds/Budgets, Score-Vektor (OOD-Mathe, Fakten, Befolgung, DE/EN, EOS, Sicherheit) statt einer einzelnen Loss-Zahl. Kandidat nur bei statistisch belegter Verbesserung *ohne* relevante Regressionssignale.
4. **Größeres Arctus erst bei Erfolg:** Ein 1B-Modell vergrößert zuerst den Rechenbedarf und nicht automatisch die Qualität. RTX4070-12GB als harte Grenze; keine ungefragte GPU-Miete.

Abnahme: konkrete neue native Gewichte, identische gepaarte Baseline, unabhängige OOD-Gewinne mit Kosten-/Risikonachweis. **Nicht aus einem erfolgreichen Qwen-Toolcall auf Arctus-Lernen schließen.**

## Priorität P1 – 24/7-Orchestrierung ohne ausufernde Kosten

- **Heute real:** stündliche ChatGPT-Entwicklungsautomatisierung bei ausgeschalteten Geräten, GitHub-Hosted CPU-CI, Cloudflare-Cron/Heartbeat alle 15 Minuten. Dies ist *periodisch*, kein kontinuierlicher Agentenprozess.
- **Vorhandene Draft-Arbeit:** #214 enthält cooperative Git-CAS-Leases, idempotente Tasks, explizite Review-Handoffs und ein read-only Arbeitsmodell. Vor produktivem Einsatz unabhängige Cold-Start-/Netzwerk-/Rate-/Crash-/Lease-Renewal-Tests; kein eigenmächtiger Deploy.
- **Lösung für echte Dauerkoordination:** persistente, minimal berechtigte Aufgabenqueue + begrenzter Scheduler + Worker mit Rollen, Laufzeit-/Token-/Action-Budget und manueller Review. Ein 24/7 erreichbarer Queue-Koordinator ist nicht dasselbe wie 24/7 kontinuierlich generierter KI-Code.
- **Harte Grenzen:** keine automatischen Merges/Training/Cloud-GPU/paid APIs, keine privaten Checkpoint-Uploads, keine kostenlosen Cloud-Limits als unbegrenzt ausgeben. Budget- und Nutzungsgrenzen täglich protokollieren.

## Priorität P2 – Entwicklungsqualität und neue Funktionen

- Einheitliches Issue-Triage-Schema `P0/P1/P2`, `confirmed bug / unverified / feature / research / superseded`, ein Verantwortlicher je Aufgabe, keine parallelen Konflikt-PRs.
- Regelmäßige Dependency-Prüfung aus #211 **erst nach Review und Aufnahme der Konfiguration in den Default-Branch**. Kein Auto-Merge.
- Zusätzliche Werkzeuge (Recherche, Code-Sandbox, Dateien, Videoanalyse) nur mit unabhängig begrenzten Permissions, Test-Fixtures und echten Herkunftsangaben.
- Geschwindigkeit erst nach P50/P95- und VRAM-Profil messen; keine Optimierung ausschließlich nach einer Einzelfrage.

## Reihenfolge der Abarbeitung

**Jetzt:** #215 einschließlich echter Qwen-Fix-/SOLVE-Suite und Launch-Check; grünes CPU-/Browser-/Windows-Gate.
**Danach:** manuelle Kern-PR-Konsolidierung, vollständige sichere Offline-/Neuinstallation/Restore-Prüfung und breiter echter Qwen-E2E-Test mit Freigabe.
**Vor Beta:** alle P0-Blocker geschlossen; Integrations-Head und Modellversion gepinnt; Datenschutz-/Lizenz-/Sicherheits-Review dokumentiert.
**Nach stabilem Kern:** native Arctus350-Trainingsforschung und volle autonome Cloud-Agentensteuerung getrennt und budgetiert weiterentwickeln.

**Datum:** Ein internes Alpha-Ziel **15.11.2026** und mögliches Closed-Beta-Ziel **13.12.2026** bleiben ausschließlich konditional; bei fehlenden Abnahmebelegen ist **31.01.2027** die konservative Planmarke, keine Garantie. Keine Veröffentlichung nur wegen Kalenderdatum.

**Produktziel:** zuerst ein wirklich benutzbares, lokales NERAL+Qwen; danach ein qualitativ eigenständiger Arctus. Der Nutzer muss keine teuren externen Modelle abonnieren, solange die vorhandene Qwen-Integration für eine lokale Beta genügt.


## Erledigte PWA-Reparaturen: echte Offline-Neustart- und Cache-Duplikat-Tests

**Fehlernachweis 1 (RED):** Der echte Chromium-Browser übernahm die erste neu installierte PWA-Seite nicht mit dem aktiven Service Worker. [CI #41334056636](https://github.com/Cola968/neral-helios/actions/runs/41334056636) scheiterte reproduzierbar an `navigator.serviceWorker.controller === null`, obwohl `navigator.serviceWorker.ready` bereits erfüllt war.

**Fix 1:** Aktivierungsphase des `static/sw.js` wartet auf Cache-Aufräumen und ruft danach `self.clients.claim()` auf. Kein `skipWaiting` bei laufenden Tabs, um keine laufenden Nutzersitzungen mitten in der Nutzung umzuschalten. [Zwischenprüfung #41334329522](https://github.com/Cola968/neral-helios/actions/runs/41334329522) grün, inklusive echter Offline-Reload und lokal erhaltenem Entwurf.

**Fehlernachweis 2 (RED):** Shell-Responses für `/?quick=0` bis `/?quick=3` erzeugten vier zusätzliche, unnötige Cache-Keys (4 statt 0). [CI #41334624157](https://github.com/Cola968/neral-helios/actions/runs/41334624157) bestätigte echte Cache-Vermehrung. API-Antworten sollen weiterhin unter keinen Umständen offline aus dem Shell-Cache geliefert werden.

**Fix 2:** Für bekannte feste App-Shell-Ressourcen wird ausschließlich `u.pathname` als Cache-Schlüssel zum Schreiben/Lesen verwendet. Der Cache-Namensraum `neral-shell-v4` trennt die neue App von Altversion v3; die bestehende `activate`-Bereinigung löscht alte `neral-shell-*`-Caches. Ausgeschlossen bleiben API-Routen, fremde Origins und Nicht-GET-Requests.

**Finales GREEN:** [Helios App CI #41334867455](https://github.com/Cola968/neral-helios/actions/runs/41334867455), [Arctus350 Flask CPU #41334867491](https://github.com/Cola968/neral-helios/actions/runs/41334867491), [Launch-Check #41334867452](https://github.com/Cola968/neral-helios/actions/runs/41334867452), [Offline-Gate #41334867451](https://github.com/Cola968/neral-helios/actions/runs/41334867451), [Syntax #41334867458](https://github.com/Cola968/neral-helios/actions/runs/41334867458) **alle SUCCESS** auf Code-Head `564b313479d5e1c3e53703e798c221b7e92fa8b6`. Helios-App-Workflow: 113 Python-Tests plus 108 Regressionen (2 bewusst abgewählt), 1 Zusatzt​​est, Chromium-Browser und Windows-Launcher bestanden.

**Grenzen:** Der reale Chromium-Lauf ist eine localhost-Fixture mit isolierten, synthetischen Inhalten; echte Android-PWA-Installations-, Netzwerkwechsel-, Telefon-Gateway-, Nutzer-/PC-Neustart- und Daten-Wiederherstellungs-Abnahmen bleiben notwendig. Kein tatsächliches externes Modelltraining und keine Cloud-Produktfreigabe.
