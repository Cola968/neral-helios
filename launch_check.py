"""NERAL launch-risk review: scoped static evidence, never a legal certification.

Reports six X-post launch topics. Local NERAL has no public account/checkout
workflow; statuses requiring legal/product decisions remain 'review' instead of
promising compliance. Does not fetch external assets or process user content.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_REMOTE_HTML = re.compile(r"""\b(?:src|href|srcset)\s*=\s*["']\s*(?:https?:)?//""", re.I)
_REMOTE_CSS = re.compile(
    r"""(?:@import\s+(?:url\()?\s*["']?\s*|url\(\s*["']?\s*)(?:https?:)?//""",
    re.I,
)
_EXTERNAL_NETWORK_JS = re.compile(
    r"""\b(?:fetch|WebSocket|EventSource|sendBeacon)\s*\(\s*["'`](?:https?:)?//""", re.I
)
_REPLAY = re.compile(
    r"""(?i)(?:\brrweb\b|\bLogRocket\b|\bFullStory\b|(?:^|[^\w])hotjar(?:[^\w]|$)|\bclarity\.ms\b|\bclarity\s*\(|\bsession[-_]?replay\b)"""
)

# Checks are deliberately scoped to the current deployed HTML/CSS/JS surface.
def audit(root: Path) -> dict:
    base = Path(root)
    html_files = sorted((base / "templates").glob("*.html"))
    css_files = sorted((base / "static").glob("*.css"))
    js_files = sorted((base / "static").glob("*.js"))
    expected = [base / "templates" / "index.html",
                base / "static" / "neral.css",
                base / "static" / "neral.js",
                base / "static" / "chat-experience.css"]
    absent = [str(p.relative_to(base)) for p in expected if not p.is_file()]
    files = html_files + css_files + js_files
    problems = []
    replay_signatures = []
    for file in files:
        try:
            content = file.read_text(encoding="utf8")
        except (OSError, UnicodeError):
            absent.append(str(file.relative_to(base)))
            continue
        relative = str(file.relative_to(base))
        if file.suffix == ".html":
            if _REMOTE_HTML.search(content):
                problems.append(relative + ": externally loaded HTML asset")
        elif file.suffix == ".css":
            if _REMOTE_CSS.search(content):
                problems.append(relative + ": externally loaded CSS asset")
        else:
            if _EXTERNAL_NETWORK_JS.search(content):
                problems.append(relative + ": hard-coded external browser request")
            if _REPLAY.search(content):
                replay_signatures.append(relative + ": known replay signature")

    def check(id, title, status, summary, next_step):
        return {"id": id, "title": title, "status": status,
                "summary": summary, "next_step": next_step}

    checks = [
        check(
            "minors", "Kinder und Altersprüfung", "review",
            "Die lokale NERAL-Oberfläche hat keinen öffentlichen Signup-Prozess; "
            "COPPA/DSGVO-Kinderregeln sind nicht automatisch erledigt.",
            "Vor öffentlicher Registrierung Zielgruppe, Mindestalter, Erhebung, "
            "Elterneinwilligung und Datenschutzerklärung rechtlich prüfen."
        ),
        check(
            "remote_fonts", "Externe Fonts und Browser-Ressourcen",
            "unverified" if absent else "blocker" if problems else "pass",
            "Statischer Scan der geprüften HTML/CSS/JS-Dateien: "
            + ("Quelldateien fehlen: " + ", ".join(absent) if absent else
               "; ".join(problems) if problems else "keine erkannten externen Browser-Ressourcen."),
            "Assets lokal hosten; anschließend echtes Browser-Netzwerkprofil "
            "und alle dynamisch konstruierten Requests prüfen."
        ),
        check(
            "session_replay", "Session-Replay und Tracking",
            "unverified" if absent else "blocker" if replay_signatures else "pass",
            "Statischer Scan: " + (
                "Quelldateien unvollständig" if absent else
                "; ".join(replay_signatures) if replay_signatures
                else "keine bekannten Session-Replay-SDKs entdeckt."
            ),
            "Vor Einführung von Analytics/Replays eine separate "
            "Einwilligungs- und Datenschutzprüfung durchführen."
        ),
        check(
            "marketing_email", "Werbe- und Launch-E-Mails", "review",
            "Kein Marketing-Versand wird durch diesen Scanner freigegeben.",
            "Vor Versand Einwilligung/gesetzliche Grundlage, Abmeldung, "
            "Absenderanschrift und länderspezifische Regeln umsetzen."
        ),
        check(
            "subscriptions", "Abos und automatische Verlängerungen", "review",
            "Stripe-Billing ist im Repository als vorbereitet, nicht live dokumentiert.",
            "Vor Live-Checkout klare wiederkehrende Preise und Laufzeiten "
            "nahe der Bestätigung, gesonderte Einwilligung, Kündigungsweg "
            "und etwaige App-Store-Vorgaben juristisch prüfen."
        ),
        check(
            "user_uploads", "Nutzeruploads und Urheberrecht", "review",
            "Lokaler Chat-JSON-Import ist nicht automatisch eine öffentlich "
            "zugängliche Plattform für Nutzerinhalte.",
            "Vor öffentlichen Uploads Moderation, Meldung/Takedown, "
            "Copyright-Regeln und je nach Markt DMCA/DSA prüfen."
        )
    ]
    return {
        "schema_version": 1,
        "scope": "static-local-source-scan",
        "is_legal_certification": False,
        "launch_ready": False,
        "blocker_count": sum(x["status"] == "blocker" for x in checks),
        "unverified_count": sum(x["status"] == "unverified" for x in checks),
        "manual_review_count": sum(x["status"] == "review" for x in checks),
        "files_scanned": len(files),
        "checks": checks,
        "notice": "Nur statischer Entwicklungscheck, keine Rechtsberatung und keine "
                  "Freigabe für öffentlichen Launch. Vollständiger Laufzeit-/Netzwerk- "
                  "und Rechtscheck weiterhin nötig."
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--fail-on-blocker", action="store_true")
    args = parser.parse_args(argv)
    report = audit(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.fail_on_blocker and (report["blocker_count"] or report["unverified_count"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
