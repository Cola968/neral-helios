# NERAL Tag 1 – Integrationsaudit (28.10.2026)

**Scope:** Bestandsaufnahme gemäß [28-Tage-Plan](../NERAL_28_DAY_PLAN.md), Tag 1, und [W1-Issue #133](https://github.com/Cola968/neral-helios/issues/133). **Nur Dokumentation; keine PRs gemerged, keine Modelle getestet, keine Kosten ausgelöst.**

## Stand der drei offenen Draft-PRs

| PR | Zweck / Backend | Geänderte Pfade | Letzter geprüfter CI-Nachweis (27.10.2026) | Empfehlung |
| --- | --- | ---: | --- | --- |
| [#129](https://github.com/Cola968/neral-helios/pull/129) | Lokaler NERAL Assistant mit **externen Qwen3.5-9B-Gewichten** über Ollama; bestehende Flask/Helios-App | 24 | [Assistant-Contract grün](https://github.com/Cola968/neral-helios/actions/runs/40599363407), [Helios CPU grün](https://github.com/Cola968/neral-helios/actions/runs/40599363467) | Getrennt als **lokalen** Opt-in-Pilot prüfen; nicht als eigenes Arctus-Modell vermarkten |
| [#131](https://github.com/Cola968/neral-helios/pull/131) | Eigenständiges Cloud-Beta-Gateway, **externes Qwen3.5-9B** via HF-Inference-Providers; neue Weboberfläche | 11 | [Cloud-Offline-Gates grün](https://github.com/Cola968/neral-helios/actions/runs/40598778434) | Cloud-Gateway unabhängig reviewen; **kein Deployment / keine Provider-Aktivierung** ohne gesonderte Freigabe |
| [#132](https://github.com/Cola968/neral-helios/pull/132) | NIS2-Belegintegrität, Ablehnung fehlender / nicht zusammenpassender Scores | 5 | [NIS2-Integrität grün](https://github.com/Cola968/neral-helios/actions/runs/40593231075), [Foundation-CI grün](https://github.com/Cola968/neral-helios/actions/runs/40593231531) | Als **logisch unabhängigen** Schutzmechanismus zuerst fachlich prüfen; NIS2 bleibt **nicht gemessen** |

**Datei-Konfliktanalyse:** Die GitHub-Dateilisten der drei PRs wurden am 28.10.2026 vollständig abgefragt. **0 identische geänderte Pfade** für #129/#131, #129/#132 und #131/#132. Das ist **keine** Garantie für konfliktfreies Zusammenführen: GitHub meldet die Merge-Fähigkeit nur jeweils gegenüber der aktuellen Basis, nicht nach dem hypothetischen sequenziellen Merge. Außerdem bestehen mögliche **semantische** Integrationskonflikte zwischen lokalem und Cloud-Chat, Herkunftskennzeichnung, API-Vertrag und Produktnavigation.

## Abhängigkeiten und vorgeschlagene Review-/Merge-Reihenfolge

1. **#132 fachlich zuerst reviewen** (Score-Integrität unabhängig von Produkt-Backends). Erst nach Code-Review und aktualisierten CI-Gates überhaupt über Merge entscheiden.
2. **#129 als lokalen Opt-in-Pfad reviewen** (Ollama/Flask/Helios). Der vorhandene Offline-/Mock-HTTP-Test ist kein Beleg für einen durchgehend echten NERAL-UI-Chat. Ein im PR berichteter echter Qwen-Ollama-One-Shot ist **kein** Arctus-Test.
3. **#131 separat als Cloud-Beta reviewen** (neuer Server, andere Oberfläche). Es gibt **keine technische Merge-Abhängigkeit** zu #129, aber vor gemeinsamer Produktfreigabe müssen Identität, Sitzungen, Datenschutz, Rate-/Budget-Limits und Fehlerzustände konsistent sein. Das prozesslokale Provider-Versuchslimit ist **keine** persistente Ausgabensperre.

Diese Reihenfolge ist ein **Vorschlag für menschliches Review**, keine Freigabe zum automatischen Merge. Alle drei PRs bleiben Draft. Für jeden tatsächlichen Merge sind erneut die aktuellen Checks und eine Integration gegen den dann aktuellen Zielbranch zu prüfen.

## Release-Gates: derzeit gesperrt

- **Modellherkunft:** Qwen ist ein fremdes Basismodell. NERAL/Helios-Software ist nicht mit eigenständig trainierten Arctus-Gewichten gleichzusetzen. NIS2 für Arctus 350M bleibt ohne vollständige eingefrorene Evidenz **NOT MEASURED**.
- **Echte Chat-E2E-Qualität:** Keine in diesem Audit ausgeführte reale Modellinferenz. Bestehende Offline-/HTTP-Tests und simulierte Antworten belegen nur Transport/Vertrag, nicht die Qualität realer generierter Antworten.
- **Cloud-Betrieb:** Keine freigegebene Provider-Abrechnung, kein dauerhaftes, über Prozessneustarts und mehrere Instanzen gültiges Kostenlimit, kein nachgewiesener öffentlicher Produktionsbetrieb.
- **Sicherheit/Produkt:** Provider-Ausfälle, Streaming/Abbruch, Rate-Limits, DSGVO/Datenschutz, reale Browser-Journey, Incident-/Rollback-Plan und Nutzer-/Agentenfreigaben benötigen gesonderte Abnahme.
- **Autonomer Coding-Agent:** [#137](https://github.com/Cola968/neral-helios/issues/137) bleibt ein offener Abnahmeauftrag; bestehende Komponenten/CI beweisen nicht den vollständigen unbeaufsichtigten Issue→Draft-PR-Lauf in einer Cloud-Sandbox.

## Issue-Zuordnung und nächster kleinstmöglicher Schritt

- **Heute (Tag 1):** [#133](https://github.com/Cola968/neral-helios/issues/133), Review-Matrix und Release-Gates. Die weiteren W1-Akzeptanzkriterien sind damit **nicht** erledigt.
- **Später, nicht vorziehen:** [#134](https://github.com/Cola968/neral-helios/issues/134) (W2 Coding-Agent), [#135](https://github.com/Cola968/neral-helios/issues/135) (W3 Weiterlauf), [#136](https://github.com/Cola968/neral-helios/issues/136) (W4 Releaseprüfung), [#137](https://github.com/Cola968/neral-helios/issues/137) (Agentenvertrag) bleiben offen.
- **Nächster Tagesauftrag 29.10.:** Tag 2 – aus einer konkreten Nutzerfrage den **prüfbaren NERAL Chat/Coding-Agent-Produktvertrag** mit eindeutigen Akzeptanzkriterien formulieren. Keine externen Inferenzaufrufe oder Trainingsjobs.

**Prüfgrenze:** Dies ist ein statisches Review von GitHub-Metadaten, PR-Dateilisten und CI-Status vom 28.10.2026. Keine lokalen Tests oder Modelltests in diesem Audit ausgeführt. Kosten durch diese Änderung: **0 €** (keine bezahlten Dienste aufgerufen).
