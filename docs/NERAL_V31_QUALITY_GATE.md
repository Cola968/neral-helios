# NERAL V3.1 — prospektive Qualitätsmessung

Stand: 19. November 2026. Entwicklungszweig, **keine produktive Freigabe**.

## Umfang und Kriterien

Der Generator `scripts/neral_v31_quality_gate.py` erstellt 100 synthetische neue Aufgaben aus einem festgelegten Seed. Das Manifest wird vor dem ersten Modellaufruf gehasht. Frühere 100er-Fragen werden beim Generieren ausgeschlossen; die zusätzlichen Rechenschritte ändern Fragen **und** Soll-Ergebnisse. Die Benchmark-Fragen und die Bewertungsregeln sind im Repository einsehbar: *prospektiv* bedeutet nicht geheim gegenüber Entwicklern und garantiert keine Unbekanntheit im Vortraining.

| Teilbereich | Aufgaben | Prüfmaßstab |
| --- | ---: | --- |
| Mathematik | 20 | Exakte ganze Zahl; unabhängige Python-Orakel |
| Kombinatorische Logik | 20 | Exakte ganze Zahl; deterministische Enumeration |
| Programmcode nachvollziehen | 20 | Exakte ganze Zahl; Python-Auswertung |
| Live-Webvergleiche | 20 | Tatsächlich gelesene HTTPS-Seiten, Zitate beider Entitäten, keine unbelegten Zahlen |
| Statisches Allgemeinwissen | 20 | Exakter überprüfbarer Kurzbegriff |

**Zentrale Einschränkung:** Die 20 Webtests messen nachweisbare *Belegstruktur*, nicht jede inhaltliche Behauptung auf Wahrheit. Ein `agent_checked`-Status darf nicht als unabhängiger Faktenbeweis dargestellt werden. Der aggregierte 90/100-Schwellenwert ist deshalb ein technisches Qualitätsgate, **keine 90-%-Behauptung zur allgemeinen Antwortgenauigkeit**.

## Latenz

Die 100 Anfragen werden auf der lokalen RTX 4070 an den existierenden V3-Dienst (`127.0.0.1:9690`) geschickt. Für jede echte Antwort wird die externe Wanduhrzeit gespeichert. Das Ergebnis enthält Median und p95, nach Aufgabentyp sowie nach tatsächlich ausgeführtem Tier. Webanfragen können von Luna zu Helios eskalieren; sie werden deshalb nicht pauschal als Luna-Leistung bezeichnet.

## Sicherheits- und Datenschutzregeln

Der GPU-Workflow benötigt einen Eigentümer-freigegebenen Commit. Er prüft die exakte Git-SHA, RTX 4070, die bestehende V3-Session und Port 9690. Er startet, beendet oder aktualisiert **keinen** Modell- oder App-Dienst und verwendet keine externen kostenpflichtigen APIs.

Die 100 Fragen sind synthetisch; Chatverläufe des Nutzers werden nicht gelesen. Für die API-Aufrufe wird ausschließlich ein neues temporäres Konto angelegt und im `finally`-Block gelöscht. Ist die Löschung nicht bestätigt, gilt der Lauf als fehlgeschlagen. In GitHub-Artefakten stehen nur Fallkennungen, Ergebnis-Boolesche Werte, HTTP-Status, Werkzeug und Latenz, **keine** Rohausgaben, Passwörter oder Chatverläufe.

Webvergleiche führen ausdrücklich angeforderte Suchanfragen über die vorhandene NERAL-Websuche aus. Die im Workflow vorgesehene Suchpause begrenzt die Anfragehäufigkeit; fehlerhafte oder nicht verfügbare Quellen sind echte Fehlschläge.

## V2.2-Vergleich und weitere Freigabe

V3.1 darf erst dann einen Gewinn gegenüber V2.2 behaupten, wenn V2.2 auf **denselben neuen Aufgaben** mit identischem Score, Zeitbudget und Hardware getrennt gemessen wurde. Dieser Zweig ändert den laufenden V3-Dienst nicht und startet keinen separaten V2.2-Prozess. Deshalb trägt der Bericht ausdrücklich `v22_paired_comparison: NOT_RUN` und `production_promotion_allowed: false` — auch bei einem Score über 90.

Regelmäßige Regressionstests und unerklärte negative Änderungen bleiben vor einem Release zu prüfen.
