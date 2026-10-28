# NERAL Orbit – persönlicher Agentenbereich v0

**Arbeitstitel:** NERAL Orbit. Nicht als frei verfügbare Marke oder Domain geprüft; keine Exklusivitätsbehauptung.

## Was jetzt tatsächlich implementiert ist

- Erreichbar über NERAL Sidebar → **Orbit · Agenten** oder direkt `http://127.0.0.1:9680/orbit`.
- Eigener lokaler Bildschirm mit echten Formularfeldern, Aufgabenliste und responsiver Darstellung.
- Drei Arten geplanter Aufgaben: `coding`, `research`, `monitor`. Fester, transparenter Vier-Schritte-Plan; **kein KI-erfundener Aktivitätsbericht**.
- Dauerhafte SQLite-Speicherung `local_state/orbit.sqlite3` mit maximal 40 Aufgaben; Revisionen verhindern versehentliche doppelte Änderungen.
- `DRAFT → APPROVED_NO_EXECUTOR → CANCELLED` oder `DRAFT → CANCELLED`. Freigabe wird protokolliert, **führt aber niemals selbst Code, Internetrecherchen oder Cloud-Jobs aus**.
- Cookie-geschützte `GET/POST /api/orbit/tasks` und `POST /api/orbit/tasks/{id}/transition`; gleiche lokale Host-/Origin-/Rate-Gates wie NERAL Chat.
- Darstellung über `textContent` statt benutzereingesetztem HTML (XSS-Schutz). Kein Herunterladen externer Modelle.

## Einschalten, ohne GPU oder Modellgewichte

Für einen lokalen Test der bestehenden Flask-Anwendung (in Entwicklung, nicht in `main`):
```powershell
$env:HELIOS_PREPARATION_MODE = '1'
$env:HELIOS_DEVICE = 'cpu'
$env:HELIOS_ORBIT_ENABLED = '1'
python app.py
```
Dann `http://127.0.0.1:9680/orbit` öffnen. Ohne Opt-in gibt die Aufgaben-API 404 zurück; die Hilfeseite bleibt sichtbar. Task-Daten bleiben ausschließlich im konfigurierten lokalen `HELIOS_STATE_ROOT`. Keine künstlichen Demonstrationsantworten. UI-Benutzer sind die bereits autorisierten lokalen NERAL-Sessions, keine eigenen Cloud-Konten.

## Explizit **noch nicht** vorhanden

- Kein eigener Cloud-Computer, keine ausgeführten Aufgaben, keine automatischen Pull Requests.
- Keine laufende oder zeitgesteuerte Recherche, kein „Always on“, keine Push-Benachrichtigungen.
- Keine dauerhafte Verbindung zu GitHub-Konten, Browsersteuerung, externen Quellen oder LLM-Tooladaptern.
- Keine eigene Authentifizierung für eine öffentlich zugängliche Version. **Nicht öffentlich bereitstellen.**
- Der Freigabestatus `APPROVED_NO_EXECUTOR` ist **kein Startauftrag**. Zukünftige Execution braucht neue, getrennte, explizite Berechtigungen, Sandboxing und E2E-Tests.

## Nächste überprüfbare Ausbaustufe

Erst Task-Review + Widerruf, vertrauenswürdige Cloud-Worker-Adapter mit minimalen Rechten, harten Limits, isolierten Worktrees und echten Red→Green-Tests. Danach ein autorisierter Beispielauftrag über Issue→Patch→Test→Draft-PR; manuelle Freigabe vor Merge. Keine unbeaufsichtigten Kosten, keine privaten Gewichte im Backend.

**Entwicklungsbasis:** gestapelter Draft-PR #175 (optional Arctus350 im Flask-Chat); dieser Orbit-PR bleibt separat und wird nicht automatisch mit #175 oder `main` zusammengeführt.
