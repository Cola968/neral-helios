"""Bounded exact math source proof for two *generic* strict English grammars.

Not an LLM proof. Never accepts extra clauses, arbitrary language, noninteger
divisions, giant numbers, or unchecked generated-code assertions.
"""
from __future__ import annotations
from dataclasses import dataclass
import re

_DIGIT=r"(?:0|[1-9][0-9]{0,3})"
_RE_SINGLE=re.compile(
    r"Compute the exact integer \("+_DIGIT.replace("(?:0|[1-9][0-9]{0,3})",r"((?:0|[1-9][0-9]{0,3}))")+
    r" squared minus "+r"((?:0|[1-9][0-9]{0,3}))"+
    r" squared\) divided by "+r"((?:0|[1-9][0-9]{0,3}))"+
    r", plus "+r"((?:0|[1-9][0-9]{0,3}))"+
    r" times "+r"((?:0|[1-9][0-9]{0,3}))"+
    r", minus "+r"((?:0|[1-9][0-9]{0,3}))"+r" cubed\.",
    re.ASCII|re.IGNORECASE)
_RE_SUM=re.compile(
    r"Compute the sum of \(k cubed minus k\) for integer k from 1 through "
    r"((?:0|[1-9][0-9]{0,3})) inclusive, take the remainder modulo "
    r"((?:0|[1-9][0-9]{0,3})), then add "
    r"((?:0|[1-9][0-9]{0,3})) times "
    r"((?:0|[1-9][0-9]{0,3}))\.",
    re.ASCII|re.IGNORECASE)

@dataclass(frozen=True)
class ExactArithmetic:
    kind:str
    numbers:tuple[int,...]
    answer:int

def parse_strict_arithmetic_source(source:str)->ExactArithmetic|None:
    if type(source) is not str or not 40<=len(source.encode("utf8"))<=1200:
        return None
    try:
        a=_RE_SINGLE.fullmatch(source)
        if a is not None:
            first,second,denom,plus_a,plus_b,minus=map(int,a.groups())
            if denom==0 or (first**2-second**2)%denom!=0:
                return None
            answer=(first**2-second**2)//denom+plus_a*plus_b-minus**3
            if not 0<=answer<=10**12:return None
            return ExactArithmetic("integer_polynomial",tuple(map(int,a.groups())),answer)
        b=_RE_SUM.fullmatch(source)
        if b is not None:
            end,modulus,factor_a,factor_b=map(int,b.groups())
            if not 1<=end<=200 or not 2<=modulus<=10000:
                return None
            answer=sum(k**3-k for k in range(1,end+1))%modulus+factor_a*factor_b
            if not 0<=answer<=10**12:return None
            return ExactArithmetic("finite_modular_sum",tuple(map(int,b.groups())),answer)
    except (TypeError,ValueError,OverflowError):
        return None
    return None

def independent_arithmetic_certificate(*,source:str,code:str,output:str,
                                        canonical:dict|None)->dict:
    refused={"verified":False,"reason":"source_unsupported"}
    if canonical is not None or type(code) is not str or not 1<=len(code.encode("utf8"))<=4096:
        return refused
    parsed=parse_strict_arithmetic_source(source)
    if parsed is None:
        return refused
    if type(output) is not str or output!=str(parsed.answer):
        return {"verified":False,"reason":"independent_proof_missing"}
    return {"verified":True,"full_source":True,"all_conditions_checked":True,
            "answer":parsed.answer,"proof_backend":"trusted_source_arithmetic"}
