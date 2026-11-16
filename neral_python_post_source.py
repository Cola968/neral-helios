"""Bounded independent POST-execution source proofs for free wording.

This is NOT proof by model-authored assert. Each recognized family must match
the ENTIRE source; numbers, relations and requested output are interpreted
from the original prompt with independent Python stdlib calculations. Unknown
or extra clauses fail closed to PR537 raw Qwen. No training, no old Blind100.
"""
from __future__ import annotations
import ast
import itertools
import math
import re
from dataclasses import dataclass
from typing import Optional

U=r"(?:0|[1-9]\d{0,5})"
I=r"(?:0|[1-9]\d{0,2})"
FLAGS=re.I|re.ASCII

@dataclass(frozen=True)
class PostTask:
    family:str
    answer:int
    method:str

def _full(pattern:str,source:str):
    return re.fullmatch(pattern,source.strip(),FLAGS)

def parse_post_source(source:str)->PostTask|None:
    if type(source) is not str or not 18<=len(source.encode("utf8"))<=2400:
        return None
    s=source.strip()
    try:
        # Three surface forms but one generic arithmetic semantic family.
        exponent=(
          _full(rf"Find the remainder when ({U}) to the power ({U}) is divided by ({U})\.?",s) or
          _full(rf"What(?:'s| is) ({U}) raised to (?:the )?power ({U}), modulo ({U})\?",s) or
          _full(rf"(?:Calculate|Compute|Evaluate) ({U})\^({U}) mod(?:ulo)? ({U})\.?",s)
        )
        if exponent:
            a,b,m=map(int,exponent.groups())
            if 2<=m<=100000 and b<=10000:
                return PostTask("modular_exponent",pow(a,b,m),"trusted_source_arithmetic")
        divisors=(
          _full(rf"How many positive divisors does (?:the integer )?({U}) have\?",s) or
          _full(rf"(?:Determine|Count|Find) (?:the number of )?positive divisors of ({U})\.?",s)
        )
        if divisors:
            n=int(divisors.group(1))
            if 1<=n<=200000:
                num=sum(n%i==0 for i in range(1,math.isqrt(n)+1))*2
                if math.isqrt(n)**2==n:num-=1
                return PostTask("divisor_count",num,"trusted_source_arithmetic")
        squares=(
          _full(rf"(?:Calculate|Evaluate|Find) the sum of k squared for (?:integers? )?k from 1 to ({U}) (?:where|with) k mod ({I}) = ({I})\.?",s) or
          _full(rf"Sum k\^2 over 1 (?:<=|≤) k (?:<=|≤) ({U}) (?:subject to|if) k % ({I}) == ({I})\.?",s)
        )
        if squares:
            n,m,a=map(int,squares.groups())
            if 0<n<=10000 and 1<=m<=100 and a<m:
                return PostTask("congruence_square_sum",
                    sum(k*k for k in range(1,n+1) if k%m==a),
                    "trusted_source_arithmetic")
        stairs=(
          _full(rf"(?:Count|Find) (?:the number of )?ways to climb ({I}) steps using jumps of 1, 2, or 3, (?:but|while) never landing on step ({I})\.?",s) or
          _full(rf"A robot climbs ({I}) steps, jumping 1, 2 or 3 at a time; it must not land on step ({I})\. How many paths\?",s)
        )
        if stairs:
            n,ban=map(int,stairs.groups())
            if 2<=n<=30 and 1<=ban<n:
                w=[0]*(n+1);w[0]=1
                for i in range(1,n+1):
                    if i!=ban:w[i]=sum(w[i-j] for j in (1,2,3) if i>=j)
                return PostTask("forbidden_stair_paths",w[n],"trusted_source_arithmetic")
        ring=(
          _full(rf"On a circular ring of ({I}) switches, how many on/off assignments (?:have|contain) exactly ({I}) switches on and no adjacent on switches\?",s) or
          _full(rf"Count binary switch arrangements around a ring of ({I}) positions with exactly ({I}) ones and no neighboring ones, including the wraparound\.?",s)
        )
        if ring:
            n,k=map(int,ring.groups())
            if 3<=n<=12 and 0<=k<=n:
                ans=sum(1 for bits in itertools.product((0,1),repeat=n)
                    if sum(bits)==k and not any(bits[i] and bits[(i+1)%n] for i in range(n)))
                return PostTask("ring_switches",ans,"independent_enumeration")
        rooks=(
          _full(rf"How many placements of ({I}) distinct rooks on an ({I}) by ({I}) board have one rook per row and column, avoiding forbidden squares (\[[^\n]{{1,240}}\])\?",s) or
          _full(rf"Count placements of ({I}) labelled rooks on a ({I})x({I}) chessboard, one in every row and column, excluding zero-based cells (\[[^\n]{{1,240}}\])\.?",s)
        )
        if rooks:
            n,n2,n3=map(int,rooks.groups()[:3])
            if n!=n2 or n!=n3 or not 2<=n<=8:return None
            forbidden=ast.literal_eval(rooks.group(4))
            if type(forbidden) is not list or len(forbidden)>24:return None
            if any(type(x) not in (list,tuple) or len(x)!=2
                   or any(type(z) is not int or not 0<=z<n for z in x)
                   for x in forbidden):return None
            blocked={tuple(x) for x in forbidden}
            total=sum(1 for p in itertools.permutations(range(n))
                    if all((row,col) not in blocked for row,col in enumerate(p)))
            return PostTask("forbidden_rooks",total,"independent_enumeration")
        # Applied science and mixed arithmetic, with dimensions explicit.
        charge=(
          _full(rf"A circuit carries ({U}) amperes for ({U}) seconds\. How many coulombs flow\?",s) or
          _full(rf"With constant current ({U}) A during ({U}) s, determine the electric charge in coulombs\.?",s)
        )
        if charge:
            a,b=map(int,charge.groups())
            return PostTask("electric_charge",a*b,"trusted_source_arithmetic")
        airline=(
          _full(rf"An airline operates ({U}) flights with ({U}) seats each and ({U}) empty seats per flight\. How many passengers\?",s) or
          _full(rf"On ({U}) flights of ({U}) seats each, ({U}) seats per aircraft stay empty\. Compute total passengers\.?",s)
        )
        if airline:
            f,c,e=map(int,airline.groups())
            if e<=c:return PostTask("airline_load",f*(c-e),"trusted_source_arithmetic")
        spring=_full(
          rf"Calculate the spring energy for a stiffness of ({U}) N/m and a compression of ({U}) m, using E=k\*x\^2/2\.?",s)
        if spring:
            k,x=map(int,spring.groups())
            if (k*x*x)%2==0:
                return PostTask("spring_energy",k*x*x//2,"trusted_source_arithmetic")
        stock=_full(
          rf"A warehouse begins with ({U}) packages\. For ({U}) days it receives ({U}) each day and ships ({U}) each day\. How many remain\?",s)
        if stock:
            start,days,received,shipped=map(int,stock.groups())
            value=start+days*(received-shipped)
            if 0<=value<=10**12:
                return PostTask("warehouse_stock",value,"trusted_source_arithmetic")
    except (ValueError,TypeError,SyntaxError,RecursionError,MemoryError):
        return None
    return None

_PROGRAM=re.compile(
    r"(?is)\A(?:What exact integer does (?:this|the) (?:unmodified )?"
    r"Python 3 (?:code|program) print\?|Predict (?:the )?integer output of "
    r"(?:this|the) Python 3 script:|What integer is printed by the following "
    r"Python 3 program\?)\s*\n<FENCE>python\s*\n(?P<code>[\s\S]{1,3800}?)\n<FENCE>\s*\Z"
    .replace("<FENCE>",chr(96)*3)
)

def original_python_program(source:str)->str|None:
    if type(source) is not str or len(source.encode("utf8"))>4096:
        return None
    m=_PROGRAM.fullmatch(source.strip())
    if m is None:return None
    code=m.group("code")
    try:
        tree=ast.parse(code,"<original-user-program>","exec")
    except (SyntaxError,TypeError,RecursionError):return None
    # No imports, external calls or access to OS/files/network; still execute
    # ONLY inside the pre-existing Docker sandbox, never on the host.
    if len(tree.body)>40 or any(isinstance(n,(
            ast.Import,ast.ImportFrom,ast.With,ast.AsyncWith,ast.Try,
            ast.Raise,ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef,
            ast.Global,ast.Nonlocal,ast.Lambda,ast.Yield,ast.YieldFrom,
            ast.Attribute,ast.Await,ast.Delete))
            for n in ast.walk(tree)):
        return None
    allowed={"print","sum","len","range","list","min","max","abs",
             "int","enumerate","zip","sorted"}
    if any(isinstance(n,ast.Call) and
           (not isinstance(n.func,ast.Name) or n.func.id not in allowed)
           for n in ast.walk(tree)):
        return None
    return code

def independent_post_certificate(*,source:str,code:str,output:str,
                                canonical:dict|None)->dict:
    rejected={"verified":False,"reason":"source_unsupported"}
    if canonical is not None or type(output) is not str or type(code) is not str:
        return rejected
    if not re.fullmatch(r"(?:0|[1-9]\d{0,11})",output.strip()):
        return rejected
    task=parse_post_source(source)
    if task is not None:
        if output.strip()!=str(task.answer):return {"verified":False,"reason":"source_answer_mismatch"}
        return {"verified":True,"full_source":True,"all_conditions_checked":True,
                "answer":task.answer,"proof_backend":task.method,
                "source_family":task.family}
    original=original_python_program(source)
    if original is None:return rejected
    # A second independent sandbox run of the ORIGINAL program, not the
    # model's generated script, checks the user's actual requested behavior.
    try:
        from neral_adaptive_planner import run_python_in_docker
        result=run_python_in_docker(original)
    except Exception:
        return {"verified":False,"reason":"original_program_execution_failed"}
    if (type(result) is not dict or result.get("status")!="PASSED"
            or result.get("exit_code",0)!=0
            or type(result.get("output")) is not str
            or result["output"].strip()!=output.strip()):
        return {"verified":False,"reason":"independent_program_mismatch"}
    return {"verified":True,"full_source":True,"all_conditions_checked":True,
            "answer":int(output.strip()),
            "proof_backend":"independent_user_source_docker",
            "source_family":"user_supplied_python"}
