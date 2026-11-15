"""24 NEW public development tasks; NOT a held-out blind benchmark.

Six mathematics, logic, programming, mixed. Several wording variants
per family. All golds computed independently from Python stdlib, never from
model replies. Do not add the burned e80c6017... blind100 content here.
"""
from __future__ import annotations
from itertools import product,permutations
import hashlib
import json
from neral_python_post_source import parse_post_source

def build()->list[dict]:
    items=[]
    def add(cat,family,q,gold):
        assert type(gold) is int and 0<=gold<=10**12
        items.append({"id":f"dev24-{cat}-{sum(x['category']==cat for x in items)+1:02d}",
            "category":cat,"family":family,"question":q,
            "gold_integer":gold,
            "question_sha256":hashlib.sha256(q.encode("utf8")).hexdigest()})
    # Mathematics: two modular paraphrases, divisor-count paraphrases,
    # finite congruence-square sum and excluded staircase traversal.
    add("mathematics","modular_exponent",
        "Find the remainder when 19 to the power 63 is divided by 97.",pow(19,63,97))
    add("mathematics","modular_exponent",
        "What's 43 raised to power 58, modulo 97?",pow(43,58,97))
    add("mathematics","divisor_count",
        "How many positive divisors does the integer 840 have?",
        sum(840%i==0 for i in range(1,841)))
    add("mathematics","divisor_count",
        "Determine the number of positive divisors of 1260.",
        sum(1260%i==0 for i in range(1,1261)))
    add("mathematics","congruence_square_sum",
        "Calculate the sum of k squared for k from 1 to 68 with k mod 7 = 3.",
        sum(k*k for k in range(1,69) if k%7==3))
    step=16;blocked=7;dp=[0]*(step+1);dp[0]=1
    for i in range(1,step+1):
        if i!=blocked:dp[i]=sum(dp[i-j] for j in (1,2,3) if i>=j)
    add("mathematics","forbidden_stair_paths",
        "A robot climbs 16 steps, jumping 1, 2 or 3 at a time; it must not land on step 7. How many paths?",
        dp[step])
    # Logic: non-smoke graph for deterministic Z3 (a different topology),
    # ring occupancy and forbidden permutations with two paraphrases each.
    graph_edges=[(0,1),(1,2),(2,3),(3,4),(4,5),(0,4),(1,5)]
    q=("Color vertices 0..5 with red, green or blue. Vertex 0 is red, "
       f"and endpoints of edges {graph_edges} must differ. How many valid colorings?")
    gold=sum(1 for cs in product(range(3),repeat=6)
             if cs[0]==0 and all(cs[a]!=cs[b] for a,b in graph_edges))
    add("logic","graph_coloring_z3",q,gold)
    for q,n,k in [
        ("On a circular ring of 8 switches, how many on/off assignments have exactly 3 switches on and no adjacent on switches?",8,3),
        ("Count binary switch arrangements around a ring of 7 positions with exactly 2 ones and no neighboring ones, including the wraparound.",7,2),
    ]:
        add("logic","ring_switches",q,sum(
            sum(bits)==k and not any(bits[i] and bits[(i+1)%n] for i in range(n))
            for bits in product((0,1),repeat=n)))
    for q,n,forbidden in [
        ("How many placements of 5 distinct rooks on an 5 by 5 board have one rook per row and column, avoiding forbidden squares [(0, 0), (2, 2), (4, 4)]?",5,[(0,0),(2,2),(4,4)]),
        ("Count placements of 4 labelled rooks on a 4x4 chessboard, one in every row and column, excluding zero-based cells [(0, 1), (2, 2)]",4,[(0,1),(2,2)]),
    ]:
        blocked=set(forbidden)
        add("logic","forbidden_rooks",q,sum(
            all((i,j) not in blocked for i,j in enumerate(p))
            for p in permutations(range(n))))
    add("logic","unsupported_free_logic",
        "Among nine named stations, count allocations satisfying a conditional exception that has not been specified.",0)
    # Programming: six different syntax ASTs, paraphrased original-code questions.
    fence=chr(96)*3
    programs=[
      ("What exact integer does this unmodified Python 3 code print?",
       "nums=list(range(7,17))\nprint(sum(nums))",sum(range(7,17))),
      ("Predict the integer output of this Python 3 script:",
       "total=0\nfor x in range(1,12):\n    if x%2==0:\n        total+=x\nprint(total)",
       sum(x for x in range(1,12) if x%2==0)),
      ("What integer is printed by the following Python 3 program?",
       "value=3\nfor i in range(8):\n    value=(value*3+i)%101\nprint(value)",
       None),
      ("What exact integer does this unmodified Python 3 code print?",
       "n=0\nfor a in range(4):\n    for b in range(5):\n        n+=a+b\nprint(n)",
       sum(a+b for a in range(4) for b in range(5))),
      ("Predict the integer output of this Python 3 script:",
       "nums=[2,5,8,11,14]\nprint(sum(nums))",sum((2,5,8,11,14))),
      ("What integer is printed by the following Python 3 program?",
       "acc=1\nfor i in range(1,7):\n    acc*=i\nprint(acc)",720),
    ]
    v=3
    for i in range(8):v=(v*3+i)%101
    programs[2]=(programs[2][0],programs[2][1],v)
    for intro,code,expected in programs:
        add("programming","original_python_program",
            intro+"\n"+fence+"python\n"+code+"\n"+fence,expected)
    # Mixed applied math with domain constraints, not a sentence token ID.
    add("mixed","electric_charge",
        "A circuit carries 19 amperes for 47 seconds. How many coulombs flow?",19*47)
    add("mixed","electric_charge",
        "With constant current 23 A during 37 s, determine the electric charge in coulombs.",23*37)
    add("mixed","airline_load",
        "An airline operates 9 flights with 118 seats each and 23 empty seats per flight. How many passengers?",
        9*(118-23))
    add("mixed","airline_load",
        "On 6 flights of 175 seats each, 18 seats per aircraft stay empty. Compute total passengers.",6*(175-18))
    add("mixed","spring_energy",
        "Calculate the spring energy for a stiffness of 34 N/m and a compression of 7 m, using E=k*x^2/2.",34*7*7//2)
    add("mixed","warehouse_stock",
        "A warehouse begins with 780 packages. For 5 days it receives 80 each day and ships 57 each day. How many remain?",
        780+5*(80-57))
    assert len(items)==24 and len({x["question_sha256"] for x in items})==24
    assert {cat:sum(x["category"]==cat for x in items)
            for cat in ("mathematics","logic","programming","mixed")}=={
                "mathematics":6,"logic":6,"programming":6,"mixed":6}
    return items

def manifest()->dict:
    data="".join(json.dumps(x,sort_keys=True,separators=(",",":"))+"\n" for x in build())
    return {"kind":"nonblind_dev24","cases":24,"sha256":hashlib.sha256(data.encode()).hexdigest(),
            "categories":{"mathematics":6,"logic":6,"programming":6,"mixed":6},
            "no_training":True}

if __name__=="__main__":
    print(json.dumps(manifest(),sort_keys=True))
