"""NERAL Lotto Lab: offline LOTTO 6aus49 statistics, no prediction or betting."""
from __future__ import annotations
import argparse
import csv
import json
import math
import random
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable

DRAW_COMBINATIONS = math.comb(49, 6)
JACKPOT_OUTCOMES = DRAW_COMBINATIONS * 10
FIELD_PRICE_EUR = Decimal('1.20')
WARNING = ('Keine Vorhersage: Alle Zahlenkombinationen sind gleich wahrscheinlich. '
           'Beliebtheitsscores sind unkalibrierte Heuristiken; Gewinne sind nicht garantiert.')


def _valid_six(numbers: Iterable[int]) -> tuple[int, ...]:
    result = tuple(numbers)
    if len(result) != 6 or any(type(n) is not int or not 1 <= n <= 49 for n in result) or len(set(result)) != 6:
        raise ValueError('Erwartet werden sechs verschiedene ganze Zahlen zwischen 1 und 49.')
    return tuple(sorted(result))


def jackpot_probability(fields: int) -> float:
    """Probability of >=1 jackpot for distinct six-number rows, common Superzahl."""
    if type(fields) is not int or not 1 <= fields <= DRAW_COMBINATIONS:
        raise ValueError('fields ausserhalb des gueltigen Bereichs')
    return fields / JACKPOT_OUTCOMES


def ticket_score(numbers: Iterable[int]) -> float:
    """Uncalibrated proxy for OTHER PLAYERS' tastes; NOT a draw prediction."""
    t = _valid_six(numbers)
    high = sum(n > 31 for n in t)
    consecutive = sum(b-a == 1 for a,b in zip(t, t[1:]))
    differences = [b-a for a,b in zip(t, t[1:])]
    score = 1.6 * high - 0.25 * (6-high)
    score -= 2.5 * max(high-4, 0)**2  # avoid only 32..49
    score -= 0.4 * sum(n % 10 == 0 for n in t)
    if len(set(differences)) == 1:
        score -= 7.0  # obvious arithmetic pattern
    elif consecutive >= 3:
        score -= 4.0
    elif consecutive in (1, 2):
        score += 0.35
    if len({n % 10 for n in t}) == 1:
        score -= 3.0
    return round(score, 5)


def generate_tickets(fields: int=6, *, mode: str='portfolio',
                     seed: int|None=None, candidates: int=300) -> list[tuple[int, ...]]:
    """Distinct rows. Seed enables reproducible tests, not foresight."""
    if type(fields) is not int or not 1 <= fields <= 18:
        raise ValueError('Je Spielschein sind 1 bis 18 Felder unterstuetzt.')
    if mode not in ('random', 'unpopular', 'portfolio'):
        raise ValueError('Unbekannter Modus: ' + str(mode))
    if type(candidates) is not int or not 1 <= candidates <= 10000:
        raise ValueError('candidates muss zwischen 1 und 10000 liegen.')
    if seed is not None and type(seed) is not int:
        raise ValueError('seed muss eine ganze Zahl sein.')
    rng = random.Random(seed) if seed is not None else random.SystemRandom()
    chosen = []
    used = set()
    for _ in range(fields):
        options = set()
        goal = 1 if mode == 'random' else candidates
        for _attempt in range(goal*4+30):
            if len(options) >= goal:
                break
            row = tuple(sorted(rng.sample(range(1,50),6)))
            if row not in used:
                options.add(row)
        if not options:
            raise RuntimeError('Keine neue Tippkombination gefunden.')
        if mode == 'random':
            best = next(iter(options))
        elif mode == 'unpopular':
            best = max(options, key=lambda row:(ticket_score(row), row))
        else:
            def priority(row):
                overlaps = [len(set(row)&set(old)) for old in chosen]
                most = max(overlaps,default=0)
                return (1.0 if most <= 2 else 0.0, -float(most),
                        ticket_score(row)-0.1*sum(overlaps), row)
            best = max(options,key=priority)
        chosen.append(best)
        used.add(best)
    return chosen


def classify_prize(ticket: Iterable[int], draw: Iterable[int], super_matches: bool) -> int|None:
    """Official German prize class numbers 1..9; None means no win."""
    hits = len(set(_valid_six(ticket))&set(_valid_six(draw)))
    if hits == 6:
        return 1 if super_matches else 2
    if hits == 5:
        return 3 if super_matches else 4
    if hits == 4:
        return 5 if super_matches else 6
    if hits == 3:
        return 7 if super_matches else 8
    if hits == 2 and super_matches:
        return 9
    return None


def load_draws(csv_path: str|Path) -> list[tuple[tuple[int,...],int]]:
    """CSV headers: n1,n2,n3,n4,n5,n6,superzahl."""
    draws = []
    with Path(csv_path).open('r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        fields = {*(f'n{i}' for i in range(1,7)), 'superzahl'}
        if not reader.fieldnames or not fields.issubset(reader.fieldnames):
            raise ValueError('CSV braucht die Spalten n1,n2,n3,n4,n5,n6,superzahl')
        for line_no,row in enumerate(reader,2):
            try:
                numbers = _valid_six(int(row[f'n{i}']) for i in range(1,7))
                sz = int(row['superzahl'])
                if not 0 <= sz <= 9:
                    raise ValueError('Superzahl ausserhalb 0..9')
            except (ValueError,TypeError) as exc:
                raise ValueError(f'Ungueltige Ziehung in CSV-Zeile {line_no}: {exc}') from exc
            draws.append((numbers,sz))
    if not draws:
        raise ValueError('Die CSV enthaelt keine Ziehungen.')
    return draws


def analyse_draws(tickets: Iterable[Iterable[int]],
                  draws: Iterable[tuple[tuple[int,...],int]],
                  *, ticket_superzahl: int) -> dict:
    """Count historical or simulated outcomes, never prospective probabilities."""
    if type(ticket_superzahl) is not int or not 0 <= ticket_superzahl <= 9:
        raise ValueError('Ungueltige Superzahl')
    rows = [_valid_six(t) for t in tickets]
    if not rows or len(rows) != len(set(rows)):
        raise ValueError('Mindestens eine eindeutige Tippkombination erforderlich.')
    counts = Counter()
    draw_count = winning_draws = 0
    for numbers,sz in draws:
        any_win = False
        for row in rows:
            prize = classify_prize(row,numbers,ticket_superzahl==sz)
            if prize is not None:
                counts[prize] += 1
                any_win = True
        winning_draws += int(any_win)
        draw_count += 1
    if not draw_count:
        raise ValueError('Keine Ziehungen vorhanden.')
    return {'draw_count':draw_count, 'winning_draws':winning_draws,
            'winning_draw_fraction':winning_draws/draw_count,
            'prize_class_counts':{str(k):counts[k] for k in range(1,10)},
            'warning':'Rueckblick oder Simulation, KEINE Prognose fuer die naechste Ziehung.'}


def single_field_any_prize_probability() -> float:
    """Exact, same for every six-number tip."""
    three_plus = sum(math.comb(6,k)*math.comb(43,6-k) for k in range(3,7))
    two_plus_super = math.comb(6,2)*math.comb(43,4)/10
    return (three_plus+two_plus_super)/DRAW_COMBINATIONS


def compare_strategies(*, fields: int=6, draws: int=3000,
                       seed: int=42, candidates: int=300) -> dict:
    """Paired Monte Carlo against the very same random drawings, no leakage."""
    if type(draws) is not int or not 1 <= draws <= 100000:
        raise ValueError('draws muss zwischen 1 und 100000 liegen.')
    modes = ('random','unpopular','portfolio')
    selections = {mode:generate_tickets(fields,mode=mode,seed=seed+i,
                                        candidates=candidates)
                  for i,mode in enumerate(modes)}
    wins = {mode:0 for mode in modes}
    rng = random.Random(seed+123456)
    for _ in range(draws):
        numbers = set(rng.sample(range(1,50),6))
        sz = rng.randrange(10)
        for mode in modes:
            won = any((len(numbers.intersection(row))>=3 or
                       (len(numbers.intersection(row))==2 and sz==0))
                      for row in selections[mode])
            wins[mode] += int(won)
    return {'draws':draws,'fields':fields,
            'single_field_theoretical_any_prize_probability':single_field_any_prize_probability(),
            'single_field_theoretical_jackpot_probability':jackpot_probability(1),
            'results':{mode:{'winning_draws':wins[mode],
                             'winning_draw_fraction':wins[mode]/draws}
                       for mode in modes},
            'disclaimer':('Diese Monte-Carlo-Stichprobe beweist keine Ueberlegenheit. '
                          'Jede Einzelkombination hat dieselbe exakte Gewinnchance; '
                          'Stichprobenschwankungen sind keine Vorhersage.')}


def _money(value: str, label: str) -> Decimal:
    try:
        amount = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError(f'{label}: ungueltiger Eurobetrag') from exc
    if not amount.is_finite() or amount<0 or amount.as_tuple().exponent < -2:
        raise ValueError(f'{label}: nichtnegativer Betrag mit maximal 2 Nachkommastellen erforderlich')
    return amount


def main(argv: list[str]|None=None) -> int:
    parser = argparse.ArgumentParser(description='NERAL Lotto Lab - Mathematik statt Gewinnversprechen')
    sub = parser.add_subparsers(dest='command',required=True)
    gen = sub.add_parser('generate',help='Tippfelder berechnen, keine Spielabgabe')
    gen.add_argument('--fields',type=int,default=6)
    gen.add_argument('--mode',choices=['random','unpopular','portfolio'],default='portfolio')
    gen.add_argument('--seed',type=int)
    gen.add_argument('--candidates',type=int,default=300)
    gen.add_argument('--ticket-number',help='Spielscheinnummer (Superzahl = letzte Ziffer)')
    gen.add_argument('--fee-eur',default='0')
    gen.add_argument('--budget-eur')
    gen.add_argument('--json',action='store_true')
    sim = sub.add_parser('compare',help='Gleiche Zufallsziehungen fuer drei Strategien')
    sim.add_argument('--draws',type=int,default=3000)
    sim.add_argument('--fields',type=int,default=6)
    sim.add_argument('--seed',type=int,default=42)
    audit = sub.add_parser('audit',help='Historische Ziehungen aus CSV auswerten')
    audit.add_argument('--csv',required=True)
    audit.add_argument('--ticket-number',required=True)
    audit.add_argument('--fields',type=int,default=6)
    audit.add_argument('--mode',choices=['random','unpopular','portfolio'],default='portfolio')
    audit.add_argument('--seed',type=int,default=42)
    args = parser.parse_args(argv)
    try:
        if args.command in ('generate','audit') and args.ticket_number is not None:
            if not args.ticket_number.isascii() or not args.ticket_number.isdecimal():
                raise ValueError('Spielscheinnummer muss nur aus Ziffern 0-9 bestehen.')
        if args.command=='generate':
            fee=_money(args.fee_eur,'Bearbeitungsgebuehr')
            total=FIELD_PRICE_EUR*args.fields+fee
            if args.budget_eur is not None and total>_money(args.budget_eur,'Budget'):
                raise ValueError('Kostenlimit ueberschritten; Anzahl der Tippfelder reduzieren.')
            rows=generate_tickets(args.fields,mode=args.mode,seed=args.seed,candidates=args.candidates)
            result={'mode':args.mode,'tickets':[list(t) for t in rows],
                    'ticket_superzahl':int(args.ticket_number[-1]) if args.ticket_number else None,
                    'jackpot_probability':jackpot_probability(len(rows)),
                    'six_main_probability':len(rows)/DRAW_COMBINATIONS,
                    'price_eur_excluding_fee':str(FIELD_PRICE_EUR*len(rows)),
                    'fee_eur':str(fee),'total_eur':str(total),'disclaimer':WARNING}
            if args.json:
                print(json.dumps(result,ensure_ascii=False,indent=2))
            else:
                for i,row in enumerate(rows,1):
                    print(f'{i:2}. {" ".join(f"{n:02}" for n in row)}')
                print(f'Superzahl: {result["ticket_superzahl"]} (nur wenn angegeben)')
                print(f'Kosten: {total} EUR inkl. angegebener Bearbeitungsgebuehr')
                print(WARNING)
        elif args.command=='compare':
            print(json.dumps(compare_strategies(fields=args.fields,draws=args.draws,seed=args.seed),
                             ensure_ascii=False,indent=2))
        else:
            rows=generate_tickets(args.fields,mode=args.mode,seed=args.seed)
            result=analyse_draws(rows,load_draws(args.csv),ticket_superzahl=int(args.ticket_number[-1]))
            result['mode']=args.mode
            print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0
    except (ValueError,OSError) as exc:
        parser.error(str(exc))
    return 2


if __name__=='__main__':
    raise SystemExit(main())
