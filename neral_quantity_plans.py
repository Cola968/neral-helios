"""Exact plans for recognized rate conversions and inventory consumption."""

import re


DECIMAL = r"\d+(?:[.,]\d+)?"
PRICE = r"\d{1,3}(?:\.\d{3})+,\d{2}|\d+,\d{2}"


def _packet(values, calculations):
    numbers = {key: {"token": token, "quote": quote, "source": source}
               for key, token, quote, source in values}
    return {"locale": "de", "missing": [], "calculations": [
        {"label": label, "expr": expr, "unit": unit} for label, expr, unit in calculations
    ]}, numbers


def quantity_plan(question, evidence):
    """Return a plan only for a complete recognized source, otherwise defer."""
    mass = re.search(rf"\b({DECIMAL})\s*[- ]?\s*(?:Gramm\b|g\b)", question, re.I)
    conversion = re.search(rf"\b1\s+Feinunze\s*=\s*({DECIMAL})\s*(?:Gramm\b|g\b)", question, re.I)
    metal = re.search(r"\b(Gold|Silber|Platin|Palladium)(?:preis|kurs|barrens?)\b", question, re.I)
    if (mass and conversion and mass.start() < conversion.start() and metal and
            re.search(r"Materialwert|Metallwert", question, re.I) and
            re.search(r"Euro|EUR|€", question, re.I)):
        # Require currency, quote basis and metal on the same published row.
        name = metal[1]
        rate = re.compile(rf"\b{name}\b[^\n]{{0,180}}?\b({PRICE})\s*(?:€|EUR)\s*(?:/\s*(?:oz|Feinunze)\b|pro\s+Feinunze\b)", re.I)
        for source, text in evidence.items():
            if not source.startswith("web"):
                continue
            for row in text.splitlines():
                match = rate.search(row)
                if match and not re.search(r"Silber|Platin|Palladium|Gold", row[:match.start()], re.I):
                    return _packet([
                        ("mass", mass[1], mass[0], "user"),
                        ("grams_per_ounce", conversion[1], conversion[0], "user"),
                        ("price", match[1], row, source),
                    ], [("Kurs je Feinunze", "price", "EUR"),
                        ("Materialwert", "mass / grams_per_ounce * price", "EUR")])
    if set(evidence) != {"user"}:
        return None
    # Complete, bounded grammar: a packed inventory, known loss, fill per
    # surviving item, known liquid stock and exactly the two corresponding asks.
    pattern = (
        rf"(?P<groups>\d+)\s+\w+\s+mit je (?P<per_group>\d+)\s+(?P<item>\w+)"
        rf"[^.!?\d]*\.\s*(?P<lost>\d+)\s+\w+\s+(?:gehen kaputt|zerbrechen)\.\s*"
        rf"In jedes übrige \w+ füllt \w+ (?P<per_item>{DECIMAL}) Liter [^.!?\d]+\.\s*"
        rf"Der \w+ enthält vorher (?P<stock>{DECIMAL}) Liter\.\s*"
        rf"Was bleibt im \w+\?\s*Wie viele gefüllte \w+ stehen bereit\?\s*$"
    )
    match = re.search(pattern, question, re.I)
    if match and not re.search(r"\d", question[:match.start()]):
        values = [(key, match[key], question[max(0, match.start(key)-18): match.end(key)+24], "user")
                  for key in ("groups", "per_group", "lost", "per_item", "stock")]
        return _packet(values, [
            ("Gefüllte " + match["item"], "groups * per_group - lost", "Stück"),
            ("Restmenge", "stock - (groups * per_group - lost) * per_item", "Liter"),
        ])
    return None
