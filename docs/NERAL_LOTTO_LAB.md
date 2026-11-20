# NERAL Lotto Lab – LOTTO 6aus49

Offline statistics and ticket selection, **not** prediction and not automated gambling. No paid APIs, GPU, login or purchase function.

## Run

    python -m neral_lotto_lab generate --fields 6 --mode portfolio --ticket-number 1234567
    python -m neral_lotto_lab generate --fields 4 --mode unpopular --seed 2026 --budget-eur 5.00 --json
    python -m neral_lotto_lab compare --draws 10000 --fields 6 --seed 2026
    python -m neral_lotto_lab audit --csv draws.csv --fields 6 --mode portfolio --seed 2026 --ticket-number 1234567
    python -m pytest -q tests/test_neral_lotto_lab.py

The audit CSV requires columns n1,n2,n3,n4,n5,n6,superzahl. Only verified historical draws should be used. The report is retrospective, not predictive.

The three modes:
- random: uniform random different six-number rows.
- unpopular: searches rows using an **uncalibrated** heuristic about other players' choices (birthdays and obvious patterns).
- portfolio: same heuristic, with a preference for small overlap of numbers between separate rows.

The score weights have NOT been trained or validated against up-to-date German customer selection data. They should not be interpreted as probabilities. Historic frequencies of drawn numbers are not used to predict future draws.

## Mathematics and restrictions

A jackpot for one row has probability 1/(10*C(49,6)) = 1/139,838,160. For n distinct six-number rows on a ticket with ONE shared Superzahl, the events of jackpot success are disjoint, so P(at least one jackpot) = n/139,838,160. The Superzahl is the last digit of the ticket number; it is not chosen independently per field.

The program calculates the official nine prize classes, not prize amounts. Prize pools vary with other winners. Without real ticket-popularity data, numerical expected profit cannot credibly be estimated.

Every field costs 1.20 EUR plus the relevant processing fee. Use --fee-eur and --budget-eur to set strict limits. Actual German state tickets contain up to 18 fields, depending on the state. No tickets can be purchased through this program. Use --seed only for reproducibility, never as a prediction signal.

## Evidence

- Official rules: https://www.lotto.de/lotto-6aus49/spielregeln
- Official odds: https://www.lotto.de/lotto-6aus49/info/gewinnwahrscheinlichkeit
- Official strategy limits: https://www.lotto.de/beste-strategien-fuer-lotto-6aus49
- J. Simon (1997), Expected Value of Lotto when not all Numbers are Equal: http://hdl.handle.net/1814/610
- H. Joe (1990), A winning strategy for lotto games?: https://onlinelibrary.wiley.com/doi/10.2307/3315454

These papers provide theoretical motivation for minimizing jackpot sharing, not evidence that this particular heuristic improves net returns. Never increase stakes to chase past losses.
