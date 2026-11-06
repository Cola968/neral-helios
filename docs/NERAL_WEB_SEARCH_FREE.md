# NERAL-Websuche: kostenlos und nur auf ausdrücklichen Wunsch

## Verfügbarkeit
Websuche ist für **Free, Plus und Pro identisch** freigeschaltet.
Es gibt keinen Websuche-Kauf, keine höheren Abo-Limits und keinen
kostenpflichtigen API-Schlüssel. Technischer Missbrauchsschutz begrenzt
die Suche auf derzeit sechs Anfragen pro Minute und Serverprozess.
Dies ist kein Tarifmerkmal.

## Bedienung
In Helios auf **Websuche · gratis** klicken, Suchbegriff eingeben
(2–160 Zeichen), **Suchen** wählen. Erst dann erfolgt eine Internetanfrage.
Bis zu fünf externe Ergebnisse zeigen Titel, Snippet und HTTPS-Quellen.

Im KI-Agenten zusätzlich **Einmalige Internetrecherche erlauben**
aktivieren und den **exakten Suchbegriff** eingeben. Das Modell kann
einmalig nur mit diesem Text im Web suchen. Andere vom Modell erzeugte
Suchbegriffe werden ignoriert. Die lokale Projektdateisuche bleibt separat.

## Datenschutz
- KI-Inferenz und Modellgewichte bleiben lokal, keine Qwen-Cloud-API.
- Der Suchbegriff wird zunächst an DuckDuckGo, falls nötig an Bing übertragen. Anbieter erfahren dabei
  auch Verbindungsmetadaten wie die IP-Adresse.
- Kein automatisches Senden von Chatverläufen, Projektdateien,
  Modellprompts oder API-Schlüsseln.
- Kein automatischer Abruf von Treffer-Websites durch den Server.
- HTTPS-Links werden geprüft; kein HTML aus Treffern wird ausgeführt.
  Webtreffer gelten ausdrücklich als unvertrauenswürdige Inhalte.
- Serververbindungen verwenden feste Zieladresse, feste Zeit-/Größenlimits,
  deaktivierte Redirects und keine impliziten Proxy-Umgebungsvariablen.

## Grenzen und Tests
DuckDuckGos öffentliches HTML-Frontend und Bings RSS-Suchfeed sind **keine offiziell stabilen APIs**.
CAPTCHAs, Änderungen des HTML-Layouts, Sperren oder Ratenbegrenzungen
können die Websuche jederzeit beeinträchtigen. Bei leeren Ergebnissen
werden keine Fakten erfunden. Ein umfassender Artikelabruf ist noch
nicht enthalten; V1 verarbeitet nur Suchtreffer.

Offline-Tests: python -m pytest -q tests/test_neral_web_search.py

Der Helios-Workflow prüft außerdem HTTP-API, Session-/Origin-Schutz,
Agentenzustimmung und Browseroberfläche. Ein separater Live-Smoke
unterscheidet reale Anbietererreichbarkeit von Fixture-Tests.
