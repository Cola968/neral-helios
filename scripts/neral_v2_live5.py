"""Manual RTX 4070 / real localhost 9690 V2 five-question acceptance.

Does NOT start/stop/install a service or execute model-authored code on host.
Creates one synthetic local test account and deletes it. Sends five newly
parameterized tasks via real HTTP; only Docker is permitted to execute scripts.
Never run as GitHub-hosted job (localhost would refer to the wrong machine).
"""
from __future__ import annotations

import http.cookiejar
import json
import re
import secrets
import sys
from decimal import Decimal, ROUND_HALF_UP
from fractions import Fraction
from itertools import permutations
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, Request, build_opener

BASE = "http://127.0.0.1:9690"
CENTS = Decimal("0.01")


def _de_number(token: str) -> Fraction:
    token = token.strip()
    if "/" in token:
        return Fraction(token)
    if "," in token and "." in token:
        token = token.replace(".", "").replace(",", ".")
    elif "," in token:
        token = token.replace(",", ".")
    return Fraction(Decimal(token))


def _scalar_from_stdout(stdout: str) -> Fraction:
    matches = re.findall(r"[-+]?\d+(?:[.,]\d+)*(?:/\d+)?", stdout)
    if len(matches) != 1:
        raise ValueError("Expected exactly one scalar result")
    return _de_number(matches[0])


def _task_cases() -> list[tuple[str, str, object]]:
    # Parameters are generated on the actual user's PC, not hardcoded in repo.
    principal = Decimal(1300 + secrets.randbelow(1700))
    annual_rate = Decimal(3 + secrets.randbelow(6))
    repayment = Decimal(150 + secrets.randbelow(200))
    percent = annual_rate / Decimal(100)
    debt = ((principal * (1 + percent) - repayment) * (1 + percent)).quantize(CENTS, rounding=ROUND_HALF_UP)
    credit = (f"Ein Darlehen von {int(principal)} Euro wird jedes Jahr mit {int(annual_rate)} Prozent "
              f"verzinst. Nach dem ersten Jahr leiste ich eine Sondertilgung von {int(repayment)} Euro. "
              "Wie hoch ist die Restschuld nach dem zweiten Jahr auf den Cent genau?")

    price = Decimal(180 + secrets.randbelow(190))
    d1 = Decimal(8 + secrets.randbelow(8))
    d2 = Decimal(3 + secrets.randbelow(6))
    net = price * (1 - d1/100) * (1 - d2/100)
    retail = (net * Decimal("1.19")).quantize(CENTS, rounding=ROUND_HALF_UP)
    discount = (f"Ein Produkt kostet {int(price)} Euro. Zuerst werden {int(d1)} Prozent Rabatt "
                f"abgezogen, vom neuen Preis weitere {int(d2)} Prozent. Danach kommen 19 Prozent "
                "Mehrwertsteuer hinzu. Was ist der endgültige Preis auf den Cent genau?")

    a = 4 + secrets.randbelow(7)
    delta = 2 + secrets.randbelow(5)
    b = a + delta
    c = 2 * b
    total = a+b+c
    chips = (f"Anna, Ben und Cora haben insgesamt {total} Spielchips. "
             f"Ben besitzt {delta} mehr als Anna. Cora besitzt exakt doppelt so viele wie Ben. "
             "Wie viele Chips hat jede Person?")

    shift_count = sum(
        p[0] != "Ada" and p[1] != "Cora" and p[2] != "Berta"
        for p in permutations(("Ada", "Berta", "Cora")))
    shifts = ("Ada, Berta und Cora müssen Morgen-, Abend- und Nachtschicht besetzen, "
              "jede Person genau eine Schicht. Ada darf nicht morgens arbeiten, "
              "Berta nicht nachts und Cora nicht abends. "
              "Wie viele zulässige vollständige Schichtpläne existieren?")

    x = 2+secrets.randbelow(7)
    a_coeff = Fraction(3,4)
    offset = Fraction(5,6)
    rhs = a_coeff*x+offset
    fraction = (f"Bestimme die Lösung x der Gleichung 3/4 mal x plus 5/6 "
                f"gleich {rhs.numerator}/{rhs.denominator}. Gib den exakten Wert an.")

    return [
        # Put the newly diagnosed combinatorial case first; acceptance
        # still requires every one of the five independent questions.
        ("shifts", shifts, Fraction(shift_count)),
        ("chips", chips, {"Anna":a, "Ben":b, "Cora":c}),
        ("credit", credit, Fraction(debt)),
        ("discount", discount, Fraction(retail)),
        ("fraction", fraction, Fraction(x)),
    ]


def _check_stdout(kind: str, stdout: str, gold: object) -> None:
    if kind == "chips":
        # The prompt fixes the output order as Anna, Ben, Cora. A bare triplet
        # is therefore as objectively checkable as three named assignments.
        # Admit only EXACTLY three unsigned integers and no other text.
        bare = re.fullmatch(r"\s*(\d+)[\s,;]+(\d+)[\s,;]+(\d+)\s*", stdout)
        if bare is not None:
            values = tuple(map(int, bare.groups()))
            if values != tuple(gold[n] for n in ("Anna", "Ben", "Cora")):
                raise ValueError("Incorrect ordered chip quantities")
            return
        found = {}
        for name in ("Anna", "Ben", "Cora"):
            hit = re.search(r"\b"+name+r"\b[^\d]{0,20}(\d+)",stdout,re.I)
            if hit is None:
                raise ValueError("Names or chip quantities missing")
            found[name] = int(hit.group(1))
        if found != gold:
            raise ValueError("Incorrect chip assignment")
        return
    if _scalar_from_stdout(stdout) != gold:
        raise ValueError("Incorrect exact arithmetic / planning output")


class Live:
    def __init__(self):
        self.cookies = http.cookiejar.CookieJar()
        self.opener = build_opener(HTTPCookieProcessor(self.cookies))
        self.csrf = None

    def bootstrap(self) -> None:
        """GET / first: the preview sets its session/CSRF cookie on this page."""
        request = Request(BASE + "/", headers={"Origin": BASE}, method="GET")
        with self.opener.open(request, timeout=15) as response:
            if response.status != 200:
                raise RuntimeError("Local NERAL start page was not ready")
            response.read(8192)

    def call(self, path: str, data: object = None) -> dict:
        raw = json.dumps(data).encode("utf-8") if data is not None else None
        headers = {"Origin": BASE, "Accept": "application/json"}
        if raw is not None:
            headers["Content-Type"] = "application/json"
            if self.csrf:
                headers["X-Neral-CSRF"] = self.csrf
        request = Request(BASE + path, data=raw, headers=headers,
                          method="POST" if raw is not None else "GET")
        with self.opener.open(request, timeout=280) as response:
            if response.status not in (200, 201):
                raise RuntimeError("Unexpected local HTTP status")
            return json.loads(response.read(100000))


def main():
    api = Live()
    completed = []
    current_case = "session_bootstrap"
    last_stdout = ""
    last_metadata = {}
    account_created = False
    cleanup = False
    password = "E2E-V2-" + secrets.token_hex(18)
    try:
        api.bootstrap()
        session = api.call("/api/neral/preview/session")
        if session.get("three_tier_enabled") is not True:
            raise RuntimeError("Three-tier local service disabled")
        api.csrf = session["csrf"]
        user = "e2ev2" + secrets.token_hex(7)
        registration = api.call("/api/neral/account/register",{"username":user,"password":password})
        if registration.get("authenticated") is not True:
            raise RuntimeError("Synthetic account registration failed")
        account_created = True
        api.csrf = registration["csrf"]
        for kind, question, gold in _task_cases():
            current_case = kind
            last_stdout = ""
            last_metadata = {}
            response = api.call("/api/neral/chat/three-tier",{
                "tier":"Luna", "prompt":question, "history":[]})
            last_metadata = {k: response.get(k) for k in (
                "verified", "safe_to_promote", "sandbox_executed",
                "tool_used", "sandbox_attempts", "verification_status",
                "sandbox_failure_trace", "tier", "auto_escalated_from")}
            if not (response.get("verified") is False and
                    response.get("safe_to_promote") is False and
                    response.get("sandbox_executed") is True and
                    response.get("tool_used") == "helios_code_sandbox" and
                    response.get("tool_calls_succeeded") == 1 and
                    response.get("tier") == "Helios" and
                    response.get("auto_escalated_from") == "Luna"):
                raise RuntimeError("Missing real execution and unverified metadata: " + kind)
            label = "Sandbox-Ausgabe (nicht unabhängig geprüft):\n"
            if label not in response.get("text",""):
                raise RuntimeError("Exact sandbox stdout not displayed")
            stdout = response["text"].split(label,1)[1].split("\n\n",1)[0].strip()
            last_stdout = stdout[:220]
            _check_stdout(kind,stdout,gold)
            completed.append(kind)
    except (HTTPError, URLError, ValueError, KeyError, RuntimeError) as exc:
        safe = {"passed":completed, "failed_type":type(exc).__name__,
                "failed_case":current_case}
        if isinstance(exc, RuntimeError):
            safe["check_error"] = str(exc)[:140]
            safe["execution_metadata"] = last_metadata
        if isinstance(exc, ValueError):
            # ValueError messages originate from the local deterministic check.
            safe["check_error"] = str(exc)[:120]
            # Only synthetic test data was sent; stdout is bounded evidence.
            safe["synthetic_stdout_preview"] = last_stdout
        if isinstance(exc, HTTPError):
            safe["http_status"] = exc.code
            safe["endpoint"] = exc.url.removeprefix(BASE).split("?", 1)[0]
        print("NERAL_V2_REAL5_FAILURE",json.dumps(safe,ensure_ascii=False))
    finally:
        if account_created:
            try:
                # HTTP DELETE is intentionally separate from JSON POST.
                request = Request(BASE+"/api/neral/account",
                    data=json.dumps({"password":password}).encode("utf-8"),
                    headers={"Origin":BASE,"Content-Type":"application/json",
                             "X-Neral-CSRF":api.csrf},
                    method="DELETE")
                with api.opener.open(request,timeout=15) as response:
                    cleanup = json.loads(response.read(10000)).get("deleted") is True
            except (HTTPError, URLError, ValueError, KeyError, OSError):
                cleanup=False
        else:
            cleanup=True
    success = len(completed)==5 and cleanup
    print("NERAL_V2_REAL5_RESULT",json.dumps({
        "pass":success, "cases_passed":len(completed),
        "synthetic_account_deleted":cleanup, "real_local_http":True,
        "sandbox_success_required":True, "model_answers_independently_verified":False
    },sort_keys=True))
    return 0 if success else 1


if __name__=="__main__":
    sys.exit(main())
