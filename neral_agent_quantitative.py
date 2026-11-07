"""NERAL's explicit local quantitative tool, not an LLM benchmark or fine-tune.

Bounded full-match inputs only. Fraction arithmetic avoids float errors.
No eval, exec, network, model loading or arbitrary natural-language guessing.
"""

from __future__ import annotations

from fractions import Fraction
import re


_NUM=r"[+-]?\d{1,6}"
_FRAC=r"([+-]?\d{1,6})/(\d{1,5})"


def _decimal_fraction(value:Fraction)->str:
    """Exact finite decimal if possible, or a reduced rational if not."""
    n,d=value.numerator,value.denominator
    if d==1:return str(n)
    factors=d
    for p in (2,5):
        while factors%p==0:
            factors//=p
    if factors!=1:
        return str(value)
    scale=0
    temp=d
    while temp>1:
        if temp%2==0:temp//=2
        elif temp%5==0:temp//=5
        scale+=1
    power=10**scale
    scaled=n*(power//d)
    sign="-" if scaled<0 else ""
    digits=str(abs(scaled)).zfill(scale+1)
    whole=digits[:-scale] if scale else digits
    decimals=digits[-scale:].rstrip("0") if scale else ""
    return sign+whole+(","+decimals if decimals else "")


def _result(value:str,method:str)->dict:
    return {"text":value,"exact":True,"tool_used":"neral_quant_exact",
            "method":method}


def _money(s:str)->dict|None:
    m=re.fullmatch(r"Berechne Endpreis:\s*(\d{1,6})(?:[.,](\d{1,2}))?\s*EUR,\s*"
                   r"Rabatt (\d{1,3})%,\s*Aufschlag (\d{1,3})%",s,re.IGNORECASE)
    if not m:return None
    whole,cents,disc,tax=m.groups()
    euros=int(whole)
    if euros>999999:return None
    cent=int((cents or "0").ljust(2,"0"))
    disc,tax=int(disc),int(tax)
    if disc>100 or tax>200:return None
    original=euros*100+cent
    exact_cents=Fraction(original*(100-disc)*(100+tax),10_000)
    cents_rounded=(2*exact_cents.numerator+exact_cents.denominator)//(2*exact_cents.denominator)
    return _result(f"{cents_rounded//100},{cents_rounded%100:02d} EUR","decimal_half_up")


def _fuel(s:str)->dict|None:
    m=re.fullmatch(r"Berechne Treibstoff:\s*(\d{1,6}) kg/h,\s*Flug (\d{1,4}) min,\s*"
                   r"Reserve (\d{1,4}) min,\s*Rollen (\d{1,6}) kg",s,re.IGNORECASE)
    if not m:return None
    rate,flight,reserve,roll=map(int,m.groups())
    if not (0<rate<=100000 and 0<=flight<=1440
            and 0<=reserve<=300 and 0<=roll<=100000):
        return None
    total=Fraction(rate*(flight+reserve),60)+roll
    return _result(_decimal_fraction(total)+" kg","rational_fuel")


def _arrival(s:str)->dict|None:
    m=re.fullmatch(r"Berechne Ankunftszeit:\s*(\d{2}):(\d{2}) UTC([+-]\d{1,2}),\s*"
                   r"Flug (\d{1,4}) min,\s*Ziel UTC([+-]\d{1,2})",s,re.IGNORECASE)
    if not m:return None
    hour,minute,src,duration,dst=map(int,m.groups())
    if (hour>23 or minute>59 or not -12<=src<=14 or not -12<=dst<=14
            or not 0<=duration<=2880):
        return None
    day,clock=divmod(60*hour+minute+duration+(dst-src)*60,1440)
    return _result(f"{clock//60:02d}:{clock%60:02d} (Tag {day:+d})","timezone_minutes")


def _fractions(s:str)->dict|None:
    m=re.fullmatch(r"Berechne Bruch:\s*"+_FRAC+r"\s*([+\-*/:])\s*"+_FRAC,s,re.IGNORECASE)
    if not m:return None
    a,b,op,c,d=m.groups()
    a,b,c,d=map(int,(a,b,c,d))
    if (not 1<=b<=10000 or not 1<=d<=10000 or
            abs(a)>1000000 or abs(c)>1000000):
        return None
    x,y=Fraction(a,b),Fraction(c,d)
    if op=="+":value=x+y
    elif op=="-":value=x-y
    elif op=="*":value=x*y
    else:
        if y==0:return None
        value=x/y
    if max(value.numerator.bit_length(),value.denominator.bit_length())>128:
        return None
    return _result(str(value),"rational_arithmetic")


def calculate_explicit(prompt:str)->dict|None:
    if (not isinstance(prompt,str) or not 1<=len(prompt)<=200
            or "\n" in prompt or "\r" in prompt):
        return None
    s=prompt.strip()
    if not s:return None
    for calculator in (_money,_fuel,_arrival,_fractions):
        answer=calculator(s)
        if answer is not None:
            return answer
    return None
