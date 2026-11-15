"""Fresh prospective100 v2: new deterministic families and independent exact oracles.

No language model queries; not trained on or selected from model outcomes.
Do not print the generated prompts. Only source hash and serialized corpus SHA
leave the CPU sealer; question text stays on the local runner.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
from itertools import permutations,product

SEED=202611151847
CATEGORIES=("mathematics","logic","programming","mixed")

def mathematics(k,r):
    f=k%5
    if f==0:
        start=r.randint(11,42);step=r.randint(3,19);n=r.randint(9,32)
        answer=sum((1 if i%2==0 else -1)*(start+i*step) for i in range(n))
        return (f"Calculate the alternating signed sum of {n} terms of the arithmetic sequence "
                f"a(i)={start}+{step}*i for i=0,...,{n-1}, with plus for even i and minus for odd i.",answer,"alternating_arithmetic")
    if f==1:
        width=r.randint(5,12);height=r.randint(4,10)
        x=r.randint(1,width-1);y=r.randint(1,height-1)
        total=math.comb(width+height,width)
        blocked=math.comb(x+y,x)*math.comb(width+height-x-y,width-x)
        return (f"On a grid, walk from (0,0) to ({width},{height}), taking only unit steps "
                f"right or up. How many paths avoid the forbidden grid point ({x},{y})?",
                total-blocked,"forbidden_grid_point")
    if f==2:
        a=r.randint(3,31);exp=r.randint(20,110);mod=r.choice((47,53,59,61,67,71,73,79,83,89,97))
        return (f"Find the least nonnegative remainder when {a} to the integer power "
                f"{exp} is divided by {mod}.",pow(a,exp,mod),"modular_power")
    if f==3:
        x=r.randint(1,7);n=r.randint(4,9);offset=r.randint(1,8)
        for i in range(n):x=3*x+2*i+offset
        return (f"An integer sequence begins with v(0)={r.randint(1,1)}. "
                f"Ignore the illustrative value and use v(0)=SPECIAL.",0,"placeholder") if False else (
                f"Starting at v(0)={x//(3**n) if False else 'START'}, iterate "
                f"v(k+1)=3*v(k)+2*k+{offset} for k=0 to {n-1}. Find v({n}).",
                x,"tripled_recurrence")
    # Identical objects, capacity-limited assignments.
    n=r.randint(9,20);caps=(r.randint(4,10),r.randint(4,10),r.randint(4,10))
    ans=sum(1 for a in range(caps[0]+1) for b in range(caps[1]+1)
            for c in range(caps[2]+1) if a+b+c==n)
    return (f"Distribute {n} identical sensors among three labeled boxes A, B and C. "
            f"The maximum numbers are {caps[0]}, {caps[1]} and {caps[2]} respectively. "
            f"How many distributions meet all capacities?",ans,"bounded_distributions")

def logic(k,r):
    f=k%5
    if f==0:
        n=r.randint(8,13);t=r.randint(2,min(6,(n+1)//2))
        ans=math.comb(n-t+1,t)
        return (f"How many binary sequences of length {n} have exactly {t} ones "
                f"and do not contain two consecutive ones?",ans,"no_adjacent_ones")
    if f==1:
        n=r.randint(5,8);target=r.randint(1,3)
        # Four colors, first color fixed to 0, adjacent distinct,
        # last vertex != vertex[target].
        ans=sum(1 for v in product(range(4),repeat=n)
                if v[0]==0 and v[-1]!=v[target]
                and all(v[i]!=v[i+1] for i in range(n-1)))
        return (f"A chain graph has {n} ordered vertices numbered 0..{n-1}. "
                f"Each may use one of four colors. Vertex 0 has color 0, "
                f"every adjacent pair has different colors and the last vertex "
                f"must differ in color from vertex {target}. Count valid colorings.",
                ans,"four_color_chain")
    if f==2:
        n=r.randint(6,8);left=r.randint(1,n-4);right=r.randint(left+1,n-2)
        # Permutations with 0 before right, left before n-1, last != 0.
        ans=sum(1 for p in permutations(range(n)) if p.index(0)<p.index(right)
                and p.index(left)<p.index(n-1) and p[-1]!=0)
        return (f"Count all permutations of the distinct labels 0 through {n-1} "
                f"where label 0 comes before label {right}, label {left} comes "
                f"before label {n-1}, and the final label is not 0.",
                ans,"order_with_final_exclusion")
    if f==3:
        total=r.randint(14,25);mod=r.randint(0,1)
        ans=sum(1 for d in product(range(9),repeat=4)
                if len(set(d))==4 and sum(d)==total
                and d[0]%2==mod and d[1]<d[2])
        return (f"Count four-digit access codes using four distinct digits "
                f"from 0..8; leading zero is allowed, digit sum is {total}, "
                f"the first digit has remainder {mod} modulo 2, "
                f"and the second digit is smaller than the third digit.",
                ans,"nine_digit_access")
    total=r.randint(16,30)
    ans=sum(1 for a,b,c in product(range(13),repeat=3)
            if a+b+c==total and a<b and b!=c and (a+c)%3==0)
    return (f"Count integer triples (A,B,C), each coordinate from 0 through 12, "
            f"for which A+B+C={total}, A<B, B!=C, and A+C is divisible by 3.",
            ans,"bounded_triples_mod3")

def programming(k,r):
    f=k%5
    if f==0:
        n=r.randint(18,50);mod=r.randint(3,11)
        source=f"s=0\nfor i in range(1,{n}):\n s+=(i*i+3*i)%{mod}\nprint(s)"
        ans=sum((i*i+3*i)%mod for i in range(1,n))
    elif f==1:
        m=r.randint(9,18);n=r.randint(6,13);mod=r.randint(3,8)
        source=(f"s=0\nfor i in range({m}):\n for j in range({n}):\n"
                f"  if (i*j)%{mod}==2:\n   s+=i+j\nprint(s)")
        ans=sum(i+j for i in range(m) for j in range(n) if (i*j)%mod==2)
    elif f==2:
        start=r.randint(70,400)
        source=f"x={start}\ns=0\nwhile x>0:\n if x%2==1:\n  s+=x\n x//=2\nprint(s)"
        x=start;ans=0
        while x>0:
            if x%2==1:ans+=x
            x//=2
    elif f==3:
        vals=[r.randint(1,60) for _ in range(r.randint(8,15))]
        source=(f"data={vals}\ns=0\nfor i in range(len(data)):\n"
                f" if data[i]%3==1:\n  s+=2*data[i]\n else:\n  s-=data[i]\nprint(s)")
        ans=sum(2*v if v%3==1 else -v for v in vals)
    else:
        n=r.randint(15,34);offset=r.randint(4,13)
        source=(f"x={offset}\nfor k in range({n}):\n"
                f" x=(x*3+k)%97\nprint(x)")
        x=offset
        for k in range(n):x=(x*3+k)%97
        ans=x
    fence=chr(96)*3
    return (f"What integer does this self-contained Python 3 program print? "
            f"Return the integer output.\n{fence}python\n{source}\n{fence}",
            ans,"fresh_python_trace_"+str(f))

def mixed(k,r):
    f=k%5
    if f==0:
        dist=r.randint(800,5000);speed=r.randint(120,480)
        result=(dist+speed-1)//speed
        return (f"A rescue aircraft travels {dist} km at a constant "
                f"{speed} km per hour. What is the minimum whole number "
                f"of hours required to complete the flight?",result,"ceil_flight_hours")
    if f==1:
        initial=r.randint(3000,9000);flow=r.randint(4,23);burn=r.randint(20,70);eng=r.randint(2,4)
        answer=initial-flow*burn*eng
        return (f"A rocket stage starts with {initial} kg propellant. "
                f"Each of {eng} engines consumes {flow} kg per second for "
                f"{burn} seconds. What integer kilograms remain?",answer,"propellant_remaining")
    if f==2:
        flights=r.randint(9,32);seats=r.randint(85,190);pct=r.choice((50,60,70,80,90))
        value=flights*seats*pct//100
        return (f"An airline operates {flights} flights with {seats} seats each. "
                f"Exactly {pct}% of all available seats are occupied. "
                f"Take the floor if required. How many passengers travel?",value,"airline_occupancy")
    if f==3:
        watts=r.randint(3,18)*250;hours=r.randint(4,20)*4
        energy=watts*hours//1000
        return (f"A ground station draws {watts} watts for {hours} hours. "
                f"How many whole kilowatt-hours of energy does it use?",energy,"groundstation_energy")
    current=r.randint(500,2000);panels=r.randint(2,8);power=r.randint(15,75)
    hours=r.randint(4,14);downlink=r.randint(100,250)
    total=current+panels*power*hours-downlink
    return (f"A solar-powered satellite has {current} Wh stored. "
            f"It receives {panels} panels times {power} watts per panel "
            f"for {hours} hours and then expends {downlink} Wh. "
            f"How many Wh remain?",total,"satellite_energy")

def build(seed:int=SEED)->list[dict]:
    if type(seed) is not int or not 0<=seed<2**31:
        raise ValueError("Invalid prospective seed")
    r=random.Random(seed)
    rows=[];used=set()
    for category,fn in zip(CATEGORIES,(mathematics,logic,programming,mixed)):
        for k in range(25):
            for attempt in range(300):
                q,gold,family=fn(k,r)
                digest=hashlib.sha256(q.encode("utf8")).hexdigest()
                if digest not in used:break
            else:raise RuntimeError("Insufficient unique fresh tasks")
            if type(gold) is not int or abs(gold)>1000000000:
                raise ValueError("Oracle out of 1e9 range")
            used.add(digest)
            rows.append({"id":f"fresh100-{category}-{k+1:02d}",
                "category":category,"family":family,
                "question":q,"gold_integer":gold,
                "question_sha256":digest})
    if len(rows)!=100 or len(used)!=100:
        raise RuntimeError("Not a unique 100-case dataset")
    return rows

def serialize(rows:list[dict])->bytes:
    return "".join(json.dumps(x,ensure_ascii=False,sort_keys=True,
                   separators=(",",":"))+"\n" for x in rows).encode("utf8")
