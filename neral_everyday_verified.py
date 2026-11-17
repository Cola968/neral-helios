"""Fully parsed everyday source proofs for NERAL's local three-tier chat.

No partial-phrase routing: a source must fit the COMPLETE audited grammar.
Python executes only in the existing networkless Docker TIR sandbox.
Schedules are solved with real local Z3 and independently checked as witnesses.
"""
from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
import re


_MONEY = re.compile(
    r"Teile (?P<gross>(?:[1-9]\d{0,2}(?:\.\d{3})+|[1-9]\d{0,7})(?:,\d{1,2})?) "
    r"€ brutto \(inkl\. (?P<vat>\d{1,2}) % MwSt\.\) exakt in Netto und Steuer "
    r"auf und verteile den Nettobetrag im Verhältnis "
    r"(?P<a>\d{1,3})\s*:\s*(?P<b>\d{1,3})\s*:\s*(?P<c>\d{1,3}) "
    r"auf drei Projektphasen\.", re.IGNORECASE,
)
_SCHEDULE = re.compile(
    r"Vier Aufgaben A \((?P<a>\d{1,2}) Std\.\), B \((?P<b>\d{1,2}) Std\.\), "
    r"C \((?P<c>\d{1,2}) Std\.\) und D \((?P<d>\d{1,2}) Std\.\) "
    r"müssen auf (?:zwei|2) Personen verteilt werden\. "
    r"A muss vor B abgeschlossen sein, "
    r"B und C dürfen nicht gleichzeitig laufen, und die Gesamtdauer darf "
    r"(?P<limit>\d{1,2}) Stunden nicht überschreiten\. "
    r"Erstelle einen gültigen Zeitplan\.", re.IGNORECASE,
)
_DISCLAIMER = re.compile(
    r"Grenze:\s*Ein lokales Modell kann keinen mathematischen Beweis liefern\.\s*",
    re.IGNORECASE,
)


def tidy_unverified_model_text(answer: str) -> str:
    """Remove one misleading generic disclaimer, never manufacture proof."""
    text = _DISCLAIMER.sub("", answer)
    return re.sub(r"([.!?])(?=(?:Rechnung|Plan|Ergebnis):)",
                  r"\1\n\n", text).strip()


def _parse_money(source: str):
    match = _MONEY.fullmatch(source)
    if match is None:
        return None
    amount = match["gross"].replace(".", "").replace(",", ".")
    try:
        gross_dec = Decimal(amount)
        if gross_dec != gross_dec.quantize(Decimal("0.01")):
            return None
        gross = int(gross_dec * 100)
        vat = int(match["vat"])
        weights = tuple(int(match[name]) for name in "abc")
    except (ArithmeticError, ValueError):
        return None
    if not (1 <= gross <= 1000000000 and 1 <= vat <= 30 and
            all(1 <= w <= 100 for w in weights)):
        return None
    return gross, vat, weights


def _cents(n: int) -> str:
    whole, tail = divmod(n, 100)
    return f"{whole:,}".replace(",", ".") + f",{tail:02d} €"


def _independent_money(gross: int, vat: int, weights: tuple[int, ...]):
    """Independent decimal/cents check, NOT trusting sandbox-side asserts."""
    net = int((Decimal(gross) * 100 / Decimal(100 + vat)).quantize(
        Decimal(1), rounding=ROUND_HALF_UP))
    tax = gross - net
    parts = [net * weight for weight in weights]
    divisor = sum(weights)
    allocated = [int(Decimal(part) // Decimal(divisor)) for part in parts]
    missing = net - sum(allocated)
    # Largest remainder, deterministic leftmost tie-break, exact cent conservation.
    ranking = sorted(range(3), key=lambda i: (-(parts[i] % divisor), i))
    for i in ranking[:missing]:
        allocated[i] += 1
    if sum(allocated) != net:
        raise ArithmeticError("Unbalanced exact allocation")
    return (net, tax, *allocated)


def _python_program(gross: int, vat: int, weights: tuple[int, ...]) -> str:
    # Only bounded integers from the entire verified input grammar enter code.
    # No model-authored instructions, no shell, no network, no host exec.
    return (
        f"gross={gross}\nvat={vat}\nweights={weights!r}\n"
        "den=100+vat\n"
        "net=(2*gross*100+den)//(2*den)\n"
        "tax=gross-net\n"
        "total=sum(weights)\n"
        "raw=[net*w for w in weights]\n"
        "parts=[v//total for v in raw]\n"
        "missing=net-sum(parts)\n"
        "rank=sorted(range(3), key=lambda i:(-(raw[i]%total),i))\n"
        "for i in rank[:missing]:\n"
        "    parts[i]+=1\n"
        "print('|'.join(str(x) for x in (net,tax,*parts)))\n"
    )


def _unavailable(tool: str) -> dict:
    return {
        "text": ("Die unabhängige " + tool +
                 "-Prüfung ist nicht erfolgreich abgeschlossen.\n\n"
                 "Es wurde deshalb kein Rechen- oder Planungsergebnis als "
                 "verifiziert ausgegeben."),
        "verified": False, "safe_to_promote": False,
        "model_generated": False, "verification_scope": "none",
        "proof_backend": None, "tool_calls_succeeded": 0,
        "response_source": "source_proof_unavailable",
    }


def _gross_verified(parsed, execute_python):
    gross, vat, weights = parsed
    expected = _independent_money(gross, vat, weights)
    program = _python_program(gross, vat, weights)
    try:
        observed = execute_python(program)
        if type(observed) is not dict or observed.get("status") != "PASSED":
            return _unavailable("Python")
        raw = observed.get("output")
        if type(raw) is not str or re.fullmatch(r"\d+(?:\|\d+){4}", raw.strip()) is None:
            return _unavailable("Python")
        actual = tuple(int(x) for x in raw.strip().split("|"))
        if actual != expected:
            return _unavailable("Python")
    except (ValueError, TypeError, ArithmeticError, RuntimeError):
        return _unavailable("Python")
    net, tax, *phases = expected
    return {
        "text": (
            f"Brutto: {_cents(gross)}\n"
            f"Netto (19 % MwSt. bereits im Brutto enthalten)" if vat == 19
            else f"Brutto: {_cents(gross)}\nNetto ({vat} % MwSt. enthalten)"
        ) + (
            f": {_cents(net)}\nSteuer: {_cents(tax)}\n\n"
            "Verteilung des Nettobetrags im Verhältnis "
            f"{weights[0]} : {weights[1]} : {weights[2]}:\n"
            f"- Phase 1: {_cents(phases[0])}\n"
            f"- Phase 2: {_cents(phases[1])}\n"
            f"- Phase 3: {_cents(phases[2])}\n\n"
            "Die drei Phasen ergeben centgenau den Nettobetrag."
        ),
        "verified": True, "safe_to_promote": True,
        "model_generated": False, "verification_scope": "full_source_independent",
        "proof_backend": "trusted_source_arithmetic",
        "verification_status": "source_python_independently_verified",
        "response_source": "independent_source_python",
        "tool_used": "python_sandbox", "tool_calls_succeeded": 1,
    }


def _parse_schedule(source: str):
    match = _SCHEDULE.fullmatch(source)
    if match is None:
        return None
    hours = tuple(int(match[name]) for name in "abcd")
    limit = int(match["limit"])
    if not (1 <= limit <= 24 and all(1 <= h <= 24 for h in hours)):
        return None
    return hours, limit


def _schedule_verified(parsed):
    hours, limit = parsed
    try:
        import z3
        starts = [z3.Int(f"everyday_s_{i}") for i in range(4)]
        people = [z3.Int(f"everyday_p_{i}") for i in range(4)]
        ends = [starts[i] + hours[i] for i in range(4)]
        opt = z3.Optimize()
        opt.set(timeout=3500)
        for i in range(4):
            opt.add(starts[i] >= 0, ends[i] <= limit)
            opt.add(z3.Or(people[i] == 0, people[i] == 1))
        # Same person: no overlap. Different persons may work in parallel.
        for i in range(4):
            for j in range(i+1, 4):
                opt.add(z3.Or(people[i] != people[j],
                              ends[i] <= starts[j], ends[j] <= starts[i]))
        opt.add(ends[0] <= starts[1])  # A must finish before B
        opt.add(z3.Or(ends[1] <= starts[2], ends[2] <= starts[1]))
        makespan = z3.Int("everyday_makespan")
        opt.add(makespan >= 0, makespan <= limit)
        for end in ends:
            opt.add(makespan >= end)
        opt.minimize(makespan)
        for start in starts:
            opt.minimize(start)
        for person in people:
            opt.minimize(person)
        if opt.check() != z3.sat:
            return _unavailable("Z3")
        model = opt.model()
        s = [model.eval(x).as_long() for x in starts]
        p = [model.eval(x).as_long() for x in people]
    except (ImportError, AttributeError, ValueError, ArithmeticError, z3.Z3Exception if "z3" in locals() else RuntimeError):
        return _unavailable("Z3")
    # Independent witness validator; never trust a textual/model claim alone.
    for i in range(4):
        if not (0 <= s[i] <= s[i] + hours[i] <= limit and p[i] in (0, 1)):
            return _unavailable("Z3")
    if s[0] + hours[0] > s[1]:
        return _unavailable("Z3")
    if not (s[1] + hours[1] <= s[2] or s[2] + hours[2] <= s[1]):
        return _unavailable("Z3")
    for i in range(4):
        for j in range(i+1, 4):
            if p[i] == p[j] and not (
                s[i] + hours[i] <= s[j] or s[j] + hours[j] <= s[i]
            ):
                return _unavailable("Z3")
    names = "ABCD"
    lines = ["Z3-geprüfter Zeitplan:"]
    for person in range(2):
        jobs = sorted((i for i in range(4) if p[i] == person),
                      key=lambda i: (s[i], i))
        entries = [f"{names[i]} {s[i]}–{s[i]+hours[i]} Std." for i in jobs]
        lines.append(f"Person {person+1}: " + "; ".join(entries))
    finish = max(s[i]+hours[i] for i in range(4))
    lines.append(f"\nGesamtdauer: {finish} Stunden (Limit: {limit} Stunden).")
    lines.append("A endet vor B; B und C überschneiden sich nicht.")
    return {
        "text": "\n".join(lines), "verified": True, "safe_to_promote": True,
        "model_generated": False, "verification_scope": "full_source_grammar_exact",
        "proof_backend": "z3_sat_witness",
        "verification_status": "source_z3_schedule_witness_verified",
        "response_source": "independent_source_parser_z3",
        "tool_used": "z3_solver", "tool_calls_succeeded": 1,
        "witness": {names[i]: {
            "person": p[i] + 1, "start": s[i],
            "end": s[i]+hours[i], "hours": hours[i],
        } for i in range(4)},
    }


def verify_everyday_source(source: str, execute_python) -> dict | None:
    """Return a proof or explicit unverified tool failure; otherwise no route."""
    if type(source) is not str or not 30 <= len(source.encode("utf-8")) <= 4096:
        return None
    source = source.strip()
    money = _parse_money(source)
    if money is not None:
        return {"preferred_tier": "Terra", **_gross_verified(money, execute_python)}
    schedule = _parse_schedule(source)
    if schedule is not None:
        return {"preferred_tier": "Helios", **_schedule_verified(schedule)}
    return None
