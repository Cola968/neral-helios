# Orbit: echter, begrenzter Recherche-Worker (Pilot)

**Stand:** Code im bestehenden **Draft-PR #176**, keine Aktivierung, kein
Deployment, kein autonomer Hintergrunddienst. Ohne PC konnten die
GitHub-Dateien bearbeitet werden; zum tatsächlichen Ausführen braucht Orbit
einen gestarteten Prozess oder einen manuell gestarteten GitHub-Actions-Job.

## Was dieser Pilot wirklich kann

Ein menschlich angelegter `research`-Auftrag durchläuft:

1. `DRAFT` – Text und Auftrag lokal in SQLite gespeichert.
2. `APPROVED_NO_EXECUTOR` – Freigabe aufgezeichnet, **noch kein Netzwerkaufruf**.
3. Separater Klick auf **„Wikipedia-Recherche jetzt starten“**, gefolgt von
   einer zweiten Bestätigung und HTTP-POST mit aktueller Aufgabenrevision.
4. `RESEARCH_RUNNING` – eine einzige HTTPS-Abfrage an
   `https://de.wikipedia.org/w/api.php` (MediaWiki Action API) für die
   genehmigte, maximal **200 Zeichen lange** Suchfrage.
5. `REVIEW_READY` – maximal drei Artikel-Einleitungen, Wiki-Links und
   Abfragezeit werden in `orbit_research_reports` neben dem ursprünglichen
   Task in SQLite gespeichert. Danach kann der Nutzer den Bericht als
   `REVIEWED` markieren und archivieren, ohne den Verlauf zu löschen.
6. Bei Fehlantwort/-zeit `RESEARCH_FAILED`, **kein erfundener Bericht**.
   Kein stiller Retry. Ein neuer Auftrag ist erforderlich.

Die Methode ist *Quellensuche mit belegbaren Textauszügen*, **keine
generative KI**, kein aktueller News-Aggregator und keine unabhängige
Faktenprüfung. Keine Überwachung, Zeitpläne, Agentenschleifen, Shell,
Dateibearbeitung, private Modelle, Schlüssel oder Kosten.

Der neue HTTPS-Client hat eine feste Ziel-Domain, sperrt Redirects und
Proxy-Nutzung, limitiert die Antwort auf **96 KiB**, die Verbindung auf
**6 Sekunden** und jeden Artikel-Auszug auf **700 Zeichen**. Beliebige
Links im Auftrag werden *nicht* als Abrufziele verwendet. Die Oberfläche
rendert alle Quelltexte mit `textContent`, nicht als HTML, und verlinkt nur
Wikipedia-URLs mit numerischen Artikel-IDs. Der Quelleninhalt gilt als
unvertrauenswürdig und wird nie als Handlungsanweisung ausgeführt.

**Vertraulichkeit:** Suchfragen werden an Wikimedia übertragen; verwende
nur öffentliche Themen und nie Passwörter, private Dokumente oder
personenbezogene Informationen. API-Nutzung und Wikipedia-Inhalte
unterliegen den Regeln/Lizenzen von Wikimedia. Der HTTP-Client identifiziert
sich ausdrücklich als OrbitBot mit Repo-URL als Kontakt.

## Reale Agentenfortschritte und Berichtsexport

Die vier Schritte des lokalen `research`-Auftrags zeigen nun
`NOT_STARTED`, `APPROVED`, `IN_PROGRESS`, `COMPLETED`,
`FAILED` oder `NEEDS_REVIEW` entsprechend dem gespeicherten
SQLite-Zustand. Auch nach `Bericht gelesen` bleibt die unabhängige
Quellenprüfung als `NEEDS_REVIEW` sichtbar. Das ist bewusst keine
Behauptung, dass Wikipedia-Aussagen durch die KI verifiziert wurden.

Ein erfolgreich gespeicherter Quellenbericht ist im lokalen
Orbit-Aufgabenfenster als **JSON herunterladbar**.
`GET /api/orbit/tasks/<id>/report` verlangt dieselbe lokale Sitzung
wie der Rest der App, löst **keinen neuen Abruf** aus, setzt
`Cache-Control: no-store` und steht auch für später archivierte
Rechercheberichte zur Verfügung. Fehlerhafte oder noch nicht
ausgeführte Aufträge besitzen keinen herunterladbaren Quellenbericht.

Schon beim **Erstellen** eines `research`-Auftrags begrenzt Orbit die
Recherchefrage auf **200 Zeichen**, passend zum einzigen realen
Wikipedia-Worker. `coding` und `monitor` bleiben unverändert
Prototypen ohne aktiven Executor; ihre Pläne sind ausdrücklich
nicht als ausgeführt gekennzeichnet.

## Lokal nach einer späteren Code-Freigabe

Bei einer lokal gestarteten Helios-App, aus dem *reviewten*
integrierten Stand:

```bash
HELIOS_ORBIT_ENABLED=1 HELIOS_DEVICE=cpu python app.py
```

Im Orbit-Tab einen **Recherche**-Auftrag mit kurzer Suchfrage erstellen,
freigeben, **separat** einmalig starten und die Quellen prüfen.
Die REST-Route `POST /api/orbit/tasks/<id>/run-research` erfordert
lokale Session/Origin-Schutzmaßnahmen, JSON `{"revision":2,"confirm":true}`
und eine vorherige Freigabe. Dieser Befehl setzt keine PC-GPU voraus,
braucht für den lokalen HTTP-Server jedoch einen laufenden Rechner.

## Schon als Code nutzbar: eigenständiger Android-/Termux-Modus

Der reine Python-Worker benötigt **keine Helios-App, kein PyTorch,
keine Modellgewichte und keine GPU**. Auf einem Android-Gerät mit einer
selbst gewählten Python-Umgebung (beispielsweise Termux) kann die Datei
`neral_orbit_research.py` aus **diesem Draft-Branch** eigenständig
ausgeführt werden, nachdem sie auf das Gerät kopiert wurde.

```bash
python neral_orbit_research.py
# read-only Plan, kein Netzwerk
python neral_orbit_research.py --self-test
# reine Offline-Prüfung mit synthetischen Quellen, ohne Netzwerk
python neral_orbit_research.py --query "Flughafen Berlin Brandenburg" \
  --out ./orbit-ber-report.json --confirm-public
```

Dabei wird **einmalig** eine öffentliche Suchfrage an Wikimedia
gesendet. Die Ausgabe ist eine neue, bereits lokal vorhandene
Verzeichnisdatei im JSON-Format mit bis zu drei verlinkten
Quellenauszügen. Vorhandene Ausgaben werden niemals überschrieben.
Unterstützt die Android-Python-Umgebung HTTPS und Dateizugriff,
kann das unabhängig von einem eingeschalteten PC funktionieren.

**Keine Freigabe zum Teilen privater Daten:** Suche nie nach
Passwörtern, Zugangstokens, privaten E-Mail-Adressen oder
Dokumentinhalten. Ein zusätzlicher Filter blockiert typische
Token- und E-Mail-Muster bereits vor jeder Anfrage. Er ist
nicht vollständig und ersetzt keine bewusste Eingabeprüfung.

Der mobile Kommandozeilenmodus ist **separat** vom lokalen
Flask-/SQLite-Agenten-Board. Sein Bericht wird nur auf dem
Telefon gespeichert; er synchronisiert **nicht** automatisch in Orbit
oder GitHub. Eine getestete Android-Installation bzw. ein realer
Wikipedia-Netzwerkabruf liegt noch **nicht** vor.

---

## Später vom Handy ohne eingeschalteten PC

`.github/workflows/neral-orbit-manual-research.yml` implementiert
eine *separate, ausschließlich manuelle* GitHub-Actions-Variante,
die denselben begrenzten Wikimedia-Client verwendet:

- `workflow_dispatch` – **kein** `push`, `schedule` oder
  `pull_request` als Auslöser;
- öffentliches `query`-Eingabefeld, bewusstes
  `confirm_public_query=YES`;
- Ubuntu-CPU, maximal **3 Minuten**, minimale Leseberechtigung,
  keine Secrets, keine GPU oder Modellgewichte;
- prüfbarer JSON-Bericht als GitHub-Artifact mit **1 Tag**
  Aufbewahrung; keine automatische Wiederholung.

**Gegenwärtiger Blocker:** Dieser Workflow befindet sich **nur im Draft**.
GitHub zeigt den manuellen Startbutton erst, wenn die Workflow-Datei
auf dem *Default-Branch* vorhanden ist. Wir mergen nichts ohne
Nutzerfreigabe. Zudem haben aktuelle GitHub-Actions-Jobs des privaten
Repositories teilweise `failure` mit **0 gestarteten Schritten** gezeigt;
die Funktionsfähigkeit der manuellen Cloud-Ausführung ist daher
**nicht bestätigt**. GitHub-Actions-Minuten könnten je nach
Kontotarif/Kontingent abrechnungsrelevant sein; der Workflow wurde
nicht gestartet.

Ein einmaliger GitHub-Job schreibt **nicht** in die lokale SQLite-Datenbank
des Orbit-Servers. Der JSON-Bericht ist ein separat prüfbarer Artefaktweg.

## Tests und verbleibende Risiken

```bash
python -m unittest tests.test_neral_orbit tests.test_neral_orbit_research -v
node --check static/orbit.js
```

Worker-Tests nutzen synthetische Wikipedia-JSON-Antworten und patchen
die Flask-Integration; CI muss dabei **keinen** Drittanbieter kontaktieren.
Im vorhandenen Browser-Test bleibt die Freigabe noch ohne Netzwerkaufruf.

Nach einem Prozessabsturz während `RESEARCH_RUNNING` kann ein Task
als laufend stehenbleiben; Wiederaufnahme und Timeout-Reconciliation
sind noch nicht implementiert. Das ist ein bekannter Pilot-Blocker für
unbeaufsichtigte Einsätze. Ein produktiver Multi-Agenten-Dienst mit
erweiterbarem Tool-Register, regelmäßigem Monitoring, ausgefeilter
Agentenplanung, Budgetkontrolle und menschlicher Review-GUI ist
ein **separater nächster Entwicklungsschritt**.

Keine echten Nutzerrecherchen, Modellexekutionen, Cloud-Jobs, Uploads,
Merges, PC-Zugriffe oder Kosten wurden durch dieses PR ausgelöst.
