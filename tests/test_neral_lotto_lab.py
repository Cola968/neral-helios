"""Contract tests for NERAL Lotto Lab."""
import json
import math
import subprocess
import sys
import pytest
from neral_lotto_lab import (
    DRAW_COMBINATIONS, jackpot_probability, ticket_score, generate_tickets,
    classify_prize, analyse_draws, load_draws
)


def test_exact_jackpot_odds():
    assert DRAW_COMBINATIONS == 13_983_816
    assert jackpot_probability(1)==pytest.approx(1/139_838_160)
    assert jackpot_probability(8)==pytest.approx(8/139_838_160)


@pytest.mark.parametrize('mode',['random','unpopular','portfolio'])
def test_valid_tickets_unique_deterministic(mode):
    tickets=generate_tickets(12,mode=mode,seed=25,candidates=150)
    assert len(tickets)==len(set(tickets))==12
    assert all(len(t)==6 and list(t)==sorted(set(t)) and 1<=min(t)<=max(t)<=49 for t in tickets)
    assert tickets==generate_tickets(12,mode=mode,seed=25,candidates=150)


def test_crowd_proxy_score():
    assert ticket_score((32,34,39,41,45,49))>ticket_score((1,3,7,12,19,24))
    assert ticket_score((1,2,3,4,5,6))<ticket_score((1,9,16,23,30,37))


def test_crowd_proxy_avoids_all_high_clusters():
    assert ticket_score((2,33,36,40,45,49))>ticket_score((32,34,39,41,45,49))


def test_portfolio_diversity():
    selected=generate_tickets(10,mode='portfolio',seed=42,candidates=300)
    assert max(len(set(x)&set(y)) for i,x in enumerate(selected) for y in selected[i+1:])<=2


def test_invalid_inputs():
    for fields in [0,19,-1]:
        with pytest.raises(ValueError):
            generate_tickets(fields)
    with pytest.raises(ValueError):
        generate_tickets(3,mode='magic')
    with pytest.raises(ValueError):
        generate_tickets(2,candidates=0)


def test_prize_classes():
    t=(1,2,3,4,5,6)
    assert classify_prize(t,t,True)==1
    assert classify_prize(t,t,False)==2
    assert classify_prize(t,(1,2,7,8,9,10),True)==9
    assert classify_prize(t,(1,2,7,8,9,10),False) is None
    assert classify_prize(t,(1,2,3,7,8,9),False)==8
    assert classify_prize(t,(1,2,3,7,8,9),True)==7


def test_load_and_audit_csv(tmp_path):
    p=tmp_path/'draws.csv'
    p.write_text('n1,n2,n3,n4,n5,n6,superzahl\n1,2,3,4,5,6,7\n11,12,13,14,15,16,4\n',encoding='utf-8')
    draws=load_draws(p)
    assert len(draws)==2
    result=analyse_draws([(1,2,3,4,5,6)],draws,ticket_superzahl=7)
    assert result['prize_class_counts']['1']==1
    p.write_text('n1,n2,n3,n4,n5,n6,superzahl\n1,1,3,4,5,6,7\n',encoding='utf-8')
    with pytest.raises(ValueError):
        load_draws(p)


def test_cli_json_tips_cost_superzahl():
    p=subprocess.run([sys.executable,'-m','neral_lotto_lab','generate','--fields','3',
                      '--seed','7','--mode','portfolio','--ticket-number','1234567',
                      '--json'],capture_output=True,text=True)
    assert p.returncode==0,p.stderr
    data=json.loads(p.stdout)
    assert data['ticket_superzahl']==7
    assert data['jackpot_probability']==pytest.approx(3/139_838_160)
    assert len(data['tickets'])==3
    assert data['price_eur_excluding_fee']=='3.60'
    assert 'Keine' in data['disclaimer']


def test_budget_guard():
    p=subprocess.run([sys.executable,'-m','neral_lotto_lab','generate','--fields','4',
                      '--budget-eur','4','--json'],capture_output=True,text=True)
    assert p.returncode!=0


def test_theoretical_any_prize_matches_hypergeometric():
    from neral_lotto_lab import single_field_any_prize_probability
    exact=(sum(math.comb(6,k)*math.comb(43,6-k) for k in range(3,7))
           +math.comb(6,2)*math.comb(43,4)/10)/DRAW_COMBINATIONS
    assert single_field_any_prize_probability()==pytest.approx(exact)
    assert 0.03<exact<0.04


def test_simulation_is_paired_and_seeded():
    from neral_lotto_lab import compare_strategies
    a=compare_strategies(fields=4,draws=120,seed=2026)
    assert a==compare_strategies(fields=4,draws=120,seed=2026)
    assert set(a['results'])=={'random','unpopular','portfolio'}
    assert all(0<=row['winning_draws']<=120 for row in a['results'].values())
    assert 'keine' in a['disclaimer'].lower()
