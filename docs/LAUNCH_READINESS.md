# NERAL Launch-Check — Codeaudit, **keine** Rechtsfreigabe

**Stand:** lokales, noch nicht öffentlich ausgerolltes Entwicklungsprodukt.
Der Button **Einstellungen → 6 Launch-Risiken prüfen** zeigt sechs
kontrollierbare Risikofelder. Die Read-only-API `/api/launch-readiness`
ist wie alle `/api/`-Routen durch die bestehende lokale Session geschützt.

## Geprüftes Verhalten

| Thema | Befund zum aktuellen lokalen Prototyp | Vor öffentlichem Launch |
|---|---|---|
| Minderjährige / COPPA | Kein öffentlicher Account-Signup im aktuellen Flask-UI. **review** | Zielgruppe, unter 13 Jahre (USA), Kinder- und DSGVO-Besonderheiten prüfen. Ein bloßes Altersfeld genügt nicht generell als Einwilligungsersatz. |
| Google Fonts / externe Browser-Assets | Code-Scan von HTML, CSS und JS auf feste Remote-Referenzen. **pass** wenn nichts entdeckt | Fontdateien selbst hosten; Browser-Network-Inspektion gegen dynamische Requests; Rechtsgrundlagen der konkreten Verarbeitung prüfen |
| Session-Replay | Code-Scan auf bekannte Replays (rrweb, Clarity, Hotjar, FullStory, LogRocket). **pass** wenn nichts entdeckt | Tracking, Zweck, Datenweitergabe und Einwilligung je nach Rechtsraum juristisch bewerten; sensible Felder nie mitschneiden |
| Werbe-E-Mails | Kein Marketingversand wird durch NERAL automatisiert freigegeben. **review** | EU: DSGVO/UWG/ePrivacy-Konstellation; USA: CAN-SPAM für kommerzielle Emails inkl. korrekter Absender-/Postadresse und funktionierender Abmeldung |
| Abonnements | Stripe ist im Repo nur als Sandbox-Architektur vorbereitet. **review** | Vor Live-Zahlungen Preise, Laufzeit, Verlängerung, Zustimmung, Kündigung, Widerruf und Rechnungen prüfen; Kalifornien verlangt deutliche Renewal-Bedingungen nahe der Zustimmung |
| Öffentliche Nutzeruploads | Browser bietet lokalen Chat-JSON-Import, keine öffentliche Content-Plattform in dieser geprüften Oberfläche. **review** | Bei öffentlichem UGC-Take-down Verfahren und Rechteinhaber-Meldungen prüfen; US-DMCA-Agent nur soweit einschlägiger Safe-Harbor beansprucht wird, zusätzlich DSA/DSGVO in EU |

**Wichtig:** `pass` heißt ausschließlich: *in den tatsächlich gelesenen
Quelldateien kein Treffer auf die bekannten statischen Signaturen*.
Nicht geprüft werden dynamische Scriptimporte, Drittanbieter-Tags in späteren
Deployments, externe Hosting-Plugins, Laufzeit-Fetches mit zusammengesetzten
URLs, App-Store-Veröffentlichungen oder alle Rechtsräume. Der Bericht setzt
daher **immer** `launch_ready=false` und `is_legal_certification=false`.
Unlesbare oder fehlende Kern-Assets führen zu `unverified`, Remote-
Browser-Assets und Replay-Indikatoren zu `blocker`; CI schlägt in diesen
Fällen fehl.

Es wird **kein** Alter erhoben, kein Cookie-Consent vorgetäuscht, kein
automatischer Versand gestartet, kein Checkout freigeschaltet und keine
DMCA-Registrierung erfunden. Personenbezogene Daten und echte
Geschäftsanschriften müssen vom verantwortlichen Betreiber vor einem
echten Launch eingetragen, geprüft und bereitgestellt werden.

## Lokal prüfen ohne Modelldownload

```bash
python launch_check.py --fail-on-blocker
python -m pytest -q tests/test_launch_check.py
```

Die API-Regressionsprüfung `tests/test_launch_check_flask.py` läuft in
bestehender CPU-only-Flask-CI. Beides verwendet keine GPU, kein externes
Datenmaterial und kein kostenpflichtiges Hosting.

## Rechtsquellen, die beim tatsächlichen Launch neu geprüft werden müssen

- US FTC COPPA: https://www.ftc.gov/business-guidance/resources/complying-coppa-frequently-asked-questions
- FTC (2026) Altersverifikation: https://www.ftc.gov/news-events/news/press-releases/2026/02/ftc-issues-coppa-policy-statement-incentivize-use-age-verification-technologies-protect-children
- LG München I, 20.01.2022, Google Fonts: https://www.gesetze-bayern.de/Content/Document/Y-300-Z-BECKRS-B-2022-N-612
- FTC CAN-SPAM: https://www.ftc.gov/business-guidance/resources/can-spam-act-compliance-guide-business
- California BPC §17602 (Automatic Renewal): https://law.justia.com/codes/california/code-bpc/division-7/part-3/chapter-1/article-9/section-17602/
- U.S. Copyright Office DMCA-Agent: https://copyright.gov/dmca-directory/faq.html

Der X-Post nennt mögliche **Höchstbeträge**, keine pauschal
automatisch bei jedem Entwickler entstehende Strafe. Der deutsche
Google-Fonts-Fall von 2022 ist ein konkretes Urteil, nicht eine
allgemeine 100-Euro-Strafe bei jedem externen Font.
