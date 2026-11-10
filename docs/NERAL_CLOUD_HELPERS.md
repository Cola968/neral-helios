# NERAL – Cloud-Helfer auch bei ausgeschalteten Geräten

Stand 01.11.2026. Das private Repository bleibt privat.

## Was bereits läuft

**ChatGPT-Automation „NERAL stündliche Entwicklung“:** Die vorhandene, aktivierte Automation schaut stündlich bei GitHub nach den neuesten Issues, Draft-PRs, CI-Ergebnissen und Branches. Wo eine sichere Korrektur begründet ist, schreibt sie einen RED-Test, einen GREEN-Fix und ändert **nur Draft-PRs**, soweit Rechte/Runner das ermöglichen. Jede Ausführung ist ein einzelner Cloud-Lauf; das ist weder ein ununterbrochener Prozess noch dauerhaft kostenlos garantierte dedizierte Rechenleistung. Die Automation braucht keinen laufenden Nutzer-PC und kein Smartphone. Soweit kein tatsächlicher Fortschritt möglich ist, meldet sie einen Blocker statt eines imaginären Commits.

**GitHub Actions:** Existierende Projekt-CI und der neue Qualitäts-Gate aus #210 prüfen bei passenden PR-Ereignissen Python, NERAL-Flask, Frontend-Browser und Qwen-Vertragsgrenzen. Sie führen CPU-Tests oder einen **simulierten** Modellserver aus; weder ein echter Qwen-Qualitätsvergleich noch eigenes Arctus350-Training wird damit automatisch durchgeführt. Private GitHub-Hosted-Actions verbrauchen das begrenzte inklusive Monatskontingent und können bei Überschreitung kostenpflichtig sein. Bestehende Actions-Abrechnung/Budgetobergrenze deshalb auf GitHub prüfen, bevor man weitere geplante Workflow-Trigger einrichtet.

## Dieser kleine zusätzliche Cloud-Helfer: Dependabot

Die Datei `.github/dependabot.yml` bereitet **wöchentliche** automatische Prüfung der vorhandenen Python-Abhängigkeiten (`requirements.txt`) und verwendeten GitHub Actions vor; höchstens ein offener Versionsupdate-PR pro Ökosystem. Keine automatischen Merges, kein eigenmächtiger GPU-Job, keine externen Zugangsdaten. Das ist **Versionspflege**, **kein autonomer KI-Programmierer** und **kein Ersatz für Code-Reviews**.

**Wichtige Aktivierungsgrenze:** Dependabot liest diese Konfiguration nur aus dem **Standardbranch**. Solange der zugehörige PR ein Draft ist, bleibt die neue Versions-Überwachung **vorbereitet, aber nicht eingeschaltet**. Aktivierung ausschließlich nach manueller Review und Merge durch den Nutzer. Dependabot-Sicherheitswarnungen können gesonderte Repository-Einstellungen erfordern.

## Sicherheitsregeln für alle Cloud-Helfer

- Keine automatischen Merges, Deployment, Releases, kostenpflichtigen AI-/GPU-Aufträge oder Trainingsjobs.
- Keine Uploads privater Arctus-Gewichte, Trainingsdaten, Tokenizer und Zugangsdaten.
- Niemand behauptet GPT-3-/GPT-3.5-Niveau ohne echten unabhängigen Vergleich.
- Cloud-Codeprüfung bei ausgeschaltetem PC ist möglich; native Arctus-GPU-Inferenz/Training benötigt weiterhin die freigegebene Hardware.
- GitHub CodeQL-Code-Scanning für private persönliche Repositories ist ohne berechtigte GitHub Code-Security-Entitlements **nicht** einfach gratis aktivierbar: keine stillschweigende Lizenzannahme.
- GitHub Actions und stündliche Automation sind unabhängige Schichten, keine Endlosschleife und keine Garantie für stündliche Codeänderungen.

Quellen: https://docs.github.com/en/code-security/concepts/supply-chain-security/about-the-dependabot-yml-file und https://docs.github.com/en/actions/concepts/billing-and-usage.

## Cloudflare Cloud-Koordinator (01.11.2026)

**Tatsächlich bereitgestellt** im verbundenen Cloudflare-Konto:

- Workers-Script `neral-cloud-coordinator`, Cloudflare Worker-Subdomain aktiviert.
- Zeitsteuerung `*/15 * * * *` (**alle 15 Minuten**, Cloudflare-Zeitzone UTC). Cloudflare bestätigt die gespeicherte Trigger-Konfiguration; die zeitversetzte Ausführung des ersten Cron-Events wurde noch nicht separat nachgewiesen.
- Eigener Cloudflare-KV-Namespace mit EU-Jurisdiktion für einen einzigen Schlüssel `last_tick`, ausschließlich ISO-Zeitstempel, keine Nutzernachrichten, Tokens, GitHub-Quellen oder Modellgewichte.
- Öffentliche, **schreibgeschützte** Diagnoseadresse: `https://neral-cloud-coordinator.a-stad.workers.dev/health`. Eine externe HTTP-Abfrage ließ sich aus der aktuellen Arbeitsumgebung nicht abschließend bestätigen. Worker-Upload, Cron-Eintrag und `workers.dev`-Freigabe sind über die Cloudflare-API bestätigt.
- Der Status sagt bewusst `ai_code_agent_running=false`, `private_github_connected=false`, `gpu_training=false`. Dies ist ein **24/7 terminierter Koordinator/Heartbeat**, **kein zusätzlicher autonomer KI-Programmierer**. Der stündliche ChatGPT-Autopilot führt die echte GitHub-Codearbeit separat durch.

**Sicherheit:** Keine privaten GitHub- oder KI-API-Zugangsdaten wurden hinterlegt. Der Worker kann per HTTP keine Schreib- oder Programmausführungsaufträge annehmen. Nur `GET /health` ist erreichbar; andere Pfade werden abgewiesen. Fünf lokale `node --test`-Kontrakttests und GitHub-Hosted-CI wurden für die geprüfte Quellversion erfolgreich durchgeführt. Quellcode: `cloud/neral-coordinator/worker.mjs` und `worker.test.mjs`.

**Kosten:** Keine GPU-, KI-API- oder Hosting-Zahlung wurde ausgelöst. Der sehr geringe geplante Cron-/KV-Verbrauch ist auf die veröffentlichten Workers-Free-/KV-Free-Limits ausgelegt. **Der tatsächliche Cloudflare-Vertrag und Rechnungsstatus konnten wegen fehlender Billing-Read-Berechtigung nicht abgerufen werden.** Der Nutzer muss deshalb im Cloudflare-Dashboard unter Workers & Pages / Billing das Kontomodell und mögliche Warnlimits prüfen. Dies ist keine Garantie, dass jede künftige Nutzung kostenlos bleibt.

**Offen für echten Cloud-KI-Agenten:** sicherer privater GitHub-Zugriff über ein gezielt eingerichtetes GitHub-App-Berechtigungsmodell, gesondert genehmigter KI-Dienst mit harter Budgetkontrolle oder bereits inkludiertem zulässigem Agenten-Kontingent sowie eine isolierte Sandbox für Tests und Review. Vorher darf dieser Worker ausdrücklich keine Pull Requests erstellen, keine externen AI-Modelle ansprechen und keine Trainingsprozesse starten. Die bestehende stündliche ChatGPT-Automation bleibt der einzige aktivierte KI-Codeentwicklungsagent.
