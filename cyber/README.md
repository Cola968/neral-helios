# NERAL Cyber Guard – lokale Vorschau (v0)

**NERAL Cyber Guard ist noch kein trainiertes Cyber-LoRA-Modell und keine echte Firewall.**
Es ist ein deterministischer, lokaler, schreibgeschützter Sicherheitscheck für
Windows. Die Grundlage für einen späteren eigenen NERAL-Cyber-Assistenten.

## Sofort ausprobieren (Windows)

1. Repository oder vollständiges Paket lokal bereitstellen.
2. `Start NERAL Cyber.cmd` doppelt anklicken oder
   `python -B -m cyber.guard` ausführen.
3. Das Ergebnis ist ausschließlich eine Momentaufnahme. Es werden keine
   Regeln geändert, keine Ports geöffnet und keine Geräte gescannt.

Optional: `python -B -m cyber.guard --json` liefert nur einen begrenzten
Statusreport ohne IP-Adressen, WLAN-Namen, Prozesse, Passwörter oder DNS-Verlauf.
Es gibt keinen Cloud-API-Aufruf, keine eigene Firewall-Treiberinstallation
und keinen Admin-Zwang. Erhebungen gehen nicht an Helios/NERAL-Cloud.

## Was geprüft wird

- Windows-Firewall-Profile Domain/Private/Public aus **ActiveStore**
- Aktuelle Windows-Netzwerkkategorien (nicht SSID oder WLAN-Passwort)
- Microsoft-Defender-Echtzeitschutz, soweit lesbar; fremder Virenschutz
  wird **nicht** als fehlender Schutz behauptet.

`status=no_critical_findings` bedeutet lediglich: keine Warnung aus
den **wenigen geprüften Eigenschaften**, nicht: Computer oder WLAN sind sicher.

## Sicheres Architekturziel

**Stufe 1 – Cyber Guard (dieser PR):** Datensparsame read-only Abfrage,
deterministische Befunde, klare Begründungen. Keine autonome Änderung.

**Stufe 2 – Cyber Advisor:** Vorhandenes lokales NERAL/Qwen-Fundament
bekommt einen gesonderten defensiven Analysemodus. Das Modell erhält nur
ausgewählte Befunde und formuliert Empfehlungen. Kein ungeprüftes
Ausführen generierter PowerShell-, Firewall- oder Router-Befehle.

**Stufe 3 – Eigenes NERAL-Cyber-LoRA:** Separater Adapter mit überprüftem
defensiven Datensatz, unabhängigen Holdouts für Diagnose, Fehlalarmrate,
Netzwerksicherheit und Policy-Following. Kein Claim auf eigenständige
Cyber-Leistung vor belastbaren Vergleichstests.

**Stufe 4 – Network Shield:** Schutz für *das gesamte Heim-WLAN* erfordert
Unterstützung des **Routers** (z. B. geeignete Router-Firewall-/DNS-Schnittstelle)
oder eine bewusst eingerichtete Gateway-Firewall. Der PC allein sieht nicht
zuverlässig den Traffic anderer Geräte. Auch DNS-Filter blockieren nicht jeden
Traffic. Änderungen am Netzwerk erst nach separater Einwilligung,
Backup, Rollback und Tests; NERAL macht keine selbstständigen Sperrungen.

Nur in eigenen/autorisierten Netzen verwenden. Kein automatisches Scannen,
keine Überwachung anderer Personen, kein Mitlesen verschlüsselter Chats.

## Test

`python -m unittest discover -s tests -p test_neral_cyber_guard.py -v`

Einschränkung: GitHub-CPU-Tests prüfen Regeln und Schnittstellen; ein
realer Windows-PC muss die PowerShell-Abfrage gegen die lokale Firewall
zusätzlich bestätigen.
